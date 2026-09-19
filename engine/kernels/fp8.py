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

GROUP = 64
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
    N: tl.constexpr, GROUPS: tl.constexpr, G: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < N
    weight = W + rows.to(tl.int64)[:, None] * K
    scales = S + rows.to(tl.int64)[:, None] * GROUPS

    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_K), BLOCK_K)
        inside = cols < K
        keep = live[:, None] & inside[None, :]

        packed = tl.load(weight + cols[None, :], mask=keep, other=0.0).to(tl.float32)
        scale = tl.load(scales + (cols // G)[None, :], mask=keep, other=0.0)
        vector = tl.load(X + cols, mask=inside, other=0.0).to(tl.float32)
        acc += packed * scale.to(tl.float32) * vector[None, :]

    tl.store(Y + rows, tl.sum(acc, axis=1).to(Y.dtype.element_ty), mask=live)


def _tiling(rows_out: int) -> tuple[int, int]:
    block_n = 128
    while block_n > 16 and triton.cdiv(rows_out, block_n) < 132:
        block_n //= 2
    return block_n, 128


def fp8_matmul(x: torch.Tensor, packed) -> torch.Tensor:
    """``x @ weight.T`` where ``packed`` is ``(e4m3 weight, group scales)``."""
    weight, scale = packed
    batch, k = x.shape
    n = weight.shape[0]
    if batch != 1:
        raise ValueError("the fp8 path serves batch 1")
    out = torch.empty((batch, n), dtype=x.dtype, device=x.device)
    block_n, block_k = _tiling(n)
    _fp8_gemv[(triton.cdiv(n, block_n),)](
        x, weight, scale, out, k,
        N=n, GROUPS=scale.shape[1], G=GROUP,
        BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=4,
    )
    return out


def bytes_moved(packed) -> int:
    weight, scale = packed
    return weight.numel() + scale.numel() * scale.element_size()
