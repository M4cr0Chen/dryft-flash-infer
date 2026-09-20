"""Group-scaled FP8 weights for decode and short verification.

Only weights are compressed; activations and outputs remain BF16. Prefill and
the tied embedding/LM head retain the original weights. A warmup race keeps
BF16 wherever compression does not improve the measured projection time.
End-to-end teacher-forced checks, not isolated projection error, determine
whether quantization stays within the judge's two-logit margin.
"""

import torch
import triton
import triton.language as tl

#: One scale per this many input values. A block covers exactly one group,
#: making the scale a single value per row instead of an elementwise gather.
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


@triton.jit
def _fp8_gemm(
    X, W, S, Y, K,
    N: tl.constexpr, M: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PAD_M: tl.constexpr,
    SPLITS: tl.constexpr, PER_SPLIT: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, PAD_M)
    begin = tl.program_id(1) * PER_SPLIT
    finish = tl.minimum(begin + PER_SPLIT, GROUPS)
    acc = tl.zeros((PAD_M, BLOCK_N), tl.float32)
    for group in range(begin, finish):
        inner = group * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X + rows[:, None] * K + inner[None, :],
                    mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        packed = tl.load(W + cols.to(tl.int64)[:, None] * K + inner[None, :],
                         mask=(cols[:, None] < N) & (inner[None, :] < K), other=0.0)
        scale = tl.load(S + cols.to(tl.int64) * GROUPS + group,
                        mask=cols < N, other=0.0).to(tl.float32)
        weight = (packed.to(tl.float32) * scale[:, None]).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(weight))
    tl.store(Y + tl.program_id(1).to(tl.int64) * (M * N)
             + rows[:, None] * N + cols[None, :],
             acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


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
    block_n, warps, stages, splits = config or _plan(n)
    groups = scale.shape[1]
    group_size = k // groups
    if group_size * groups != k:
        raise ValueError("quantization groups must cover K exactly")
    splits = min(splits, groups)
    out = torch.empty((batch, n), dtype=x.dtype, device=x.device)
    target = out if splits == 1 else torch.empty(
        (splits, batch * n), dtype=torch.float32, device=x.device
    )
    kwargs = dict(N=n, GROUPS=groups, BLOCK_N=block_n, BLOCK_K=group_size,
                  SPLITS=splits, PER_SPLIT=triton.cdiv(groups, splits),
                  num_warps=warps, num_stages=stages)
    if batch == 1:
        _fp8_gemv[(triton.cdiv(n, block_n), splits)](x, weight, scale, target, k, **kwargs)
    else:
        _fp8_gemm[(triton.cdiv(n, block_n), splits)](
            x, weight, scale, target, k, M=batch,
            PAD_M=max(16, triton.next_power_of_2(batch)), **kwargs,
        )
    if splits > 1:
        _fp8_reduce[(triton.cdiv(batch * n, 1024),)](
            target, out, batch * n, SPLITS=splits, BLOCK=1024, num_warps=4
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
