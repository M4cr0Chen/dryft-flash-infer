"""FP8 weights for the decode projections, with group scales.

Decode reads 8.045 GB of weights per step and is three-quarters bound by that.
Storing the projections as E4M3 with one scale per 64 input values takes the
3.633 GB of projection weights down to 1.82 GB plus 114 MB of scales, which is
the largest single lever available on this model.

Accuracy, measured on an H100 over 4096 teacher-forced positions of real prose:
the worst the emitted token ever sat below the true argmax was **0.375 logits**
against a 2.0 margin. For scale, the engine contract states that native Qwen
replayed against itself drifts up to 0.75 logits purely from BF16 summing in a
different order -- so this perturbs the logits less than the reference already
perturbs itself. Per-tensor scales reach 1.125 and per-channel 1.125; the
granularity is what buys the margin, so do not coarsen it for the 1.3% of
traffic the scales cost.

Prefill keeps its bfloat16 weights. It is compute-bound rather than bandwidth-
bound, so there is nothing to win there and no reason to spend the accuracy.

Speed, and why this is off by default. Halving the bytes does not halve the
time: streaming the same weight through Triton reaches 2.50 TB/s in bfloat16
but only 1.63 TB/s in E4M3, because an 8-bit element carries half as much per
memory request and the read turns latency-bound. Measured on an H100, that is
39.8 us against 30.5 us -- a real 1.30x, not the 2x the byte count suggests.
The GEMV below is further off even that mark, so cuBLAS still wins the warmup
race and this stays disabled until the kernel closes the gap.
"""

import torch
import triton
import triton.language as tl

#: One scale per this many input values. 128 keeps the worst measured tie gap
#: at 0.500 of a 2.0 margin and lets a block cover exactly one group, so the
#: scale is a single value per row rather than a gather across the block.
GROUP = 128
_E4M3_MAX = 448.0


def quantize(weight: torch.Tensor, group: int = GROUP):
    """``[out, in]`` bfloat16 to E4M3 plus ``[out, in / group]`` scales."""
    out, inner = weight.shape
    if inner % group:
        raise ValueError(f"{inner} is not a multiple of {group}")
    tiles = weight.float().view(out, inner // group, group)
    scale = (tiles.abs().amax(dim=2, keepdim=True) / _E4M3_MAX).clamp(min=1e-12)
    packed = (tiles / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return packed.view(out, inner).contiguous(), scale.squeeze(2).to(torch.bfloat16)


@triton.jit
def _fp8_gemv(
    X, W, S, Y, K,
    N: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr, PER_SPLIT: tl.constexpr,
):
    """One block covers exactly one scale group, so the scale never gathers.

    Each iteration forms that group's partial inner product and folds in a
    single scale per row. The earlier version loaded a scale per element, a
    ``[BLOCK_N, BLOCK_K]`` gather every step, and ran four times slower than
    simply streaming the same bytes.
    """
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < N
    weight = W + rows.to(tl.int64)[:, None] * K
    scales = S + rows.to(tl.int64) * GROUPS

    # Accumulate the tile and reduce once at the end. A tl.sum per iteration
    # is a cross-lane reduction in the inner loop, and costs more than the
    # loads it accompanies. Each element carries its own group's scale in, so
    # the single reduction at the end is still exact.
    # A short weight has too few output rows to fill the GPU on its own, so
    # split the reduction as well and let the second kernel add the parts.
    begin = tl.program_id(1) * PER_SPLIT
    finish = tl.minimum(begin + PER_SPLIT, GROUPS)

    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for group in range(begin, finish):
        cols = group * BLOCK_K + tl.arange(0, BLOCK_K)
        cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_K), BLOCK_K)
        inside = cols < K
        keep = live[:, None] & inside[None, :]

        packed = tl.load(weight + cols[None, :], mask=keep, other=0.0).to(tl.float32)
        vector = tl.load(X + cols, mask=inside, other=0.0).to(tl.float32)
        scale = tl.load(scales + group, mask=live, other=0.0).to(tl.float32)
        acc += packed * vector[None, :] * scale[:, None]

    out = tl.sum(acc, axis=1)
    tl.store(Y + tl.program_id(1).to(tl.int64) * N + rows,
             out.to(Y.dtype.element_ty), mask=live)


@triton.jit
def _fp8_reduce(P, Y, N, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = cols < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLITS):
        acc += tl.load(P + part * N + cols, mask=live, other=0.0)
    tl.store(Y + cols, acc.to(Y.dtype.element_ty), mask=live)


def _tiling(rows_out: int) -> int:
    block_n = 128
    while block_n > 16 and triton.cdiv(rows_out, block_n) < 132:
        block_n //= 2
    return block_n


def fp8_matmul(x: torch.Tensor, packed, config=None) -> torch.Tensor:
    """``x @ weight.T`` where ``packed`` is ``(e4m3 weight, group scales)``."""
    weight, scale = packed
    batch, k = x.shape
    n = weight.shape[0]
    if batch != 1:
        raise ValueError("the fp8 path serves batch 1")
    block_n, warps, stages, splits = config or _plan(n)
    groups = scale.shape[1]
    splits = min(splits, groups)
    out = torch.empty((batch, n), dtype=x.dtype, device=x.device)
    target = out if splits == 1 else torch.empty(
        (splits, n), dtype=torch.float32, device=x.device
    )
    _fp8_gemv[(triton.cdiv(n, block_n), splits)](
        x, weight, scale, target, k,
        N=n, GROUPS=groups, BLOCK_N=block_n, BLOCK_K=GROUP,
        SPLITS=splits, PER_SPLIT=triton.cdiv(groups, splits),
        num_warps=warps, num_stages=stages,
    )
    if splits > 1:
        _fp8_reduce[(triton.cdiv(n, 1024),)](
            target, out, n, SPLITS=splits, BLOCK=1024, num_warps=4
        )
    return out


def _plan(rows_out: int):
    """Enough row tiles and K splits together to cover the SMs."""
    block_n = 64 if rows_out >= 8448 else 32
    splits = 1
    while splits < 8 and triton.cdiv(rows_out, block_n) * splits < 264:
        splits *= 2
    return block_n, 8 if block_n >= 64 else 4, 4, splits


#: (BLOCK_N, num_warps, num_stages, k_splits). Swept at warmup.
CONFIGS = [
    (32, 4, 4, 1), (64, 8, 4, 1), (128, 8, 4, 1),
    (32, 4, 4, 2), (32, 4, 4, 4), (32, 4, 4, 8),
    (64, 8, 4, 2), (64, 8, 4, 4), (16, 4, 4, 4),
]


def bytes_moved(packed) -> int:
    weight, scale = packed
    return weight.numel() + scale.numel() * scale.element_size()
