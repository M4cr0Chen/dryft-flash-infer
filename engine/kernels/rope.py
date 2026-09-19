"""Per-head RMSNorm plus RoPE, fused, writing K and V straight into the cache.

The reference order is q_norm/k_norm over each 128-value head, *then* RoPE with
the absolute cache position; V gets neither. ``rotate_half`` needs the
normalised value at the partner lane, so each program loads its heads twice and
normalises both halves with the one scale each head shares.

Rounding follows the reference: every bfloat16 product rounds where torch would
round it. ``(q * cos) + (rotate_half(q) * sin)`` is three roundings, not one.

Both kernels read the fused QKV projection in place, by row stride and column
offset, so no slice of it is ever made contiguous first. A program covers
``BH`` heads of one row to keep the block large enough to be worth launching.
"""

import torch
import triton
import triton.language as tl

_HEAD_DIM = 128
_HEADS_PER_PROGRAM = 8


@triton.jit
def _norm_rope(x, x_partner, w, w_partner, cos, sin, half, eps, D: tl.constexpr):
    """RMSNorm over the last axis, then RoPE. fp32 in, bfloat16 out."""
    scale = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / D + eps)
    a = ((x * scale).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    b = ((x_partner * scale).to(tl.bfloat16).to(tl.float32) * w_partner)
    b = b.to(tl.bfloat16).to(tl.float32)

    rotated = tl.where(half, -b, b)
    left = (a * cos).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sin).to(tl.bfloat16).to(tl.float32)
    return (left + right).to(tl.bfloat16)


@triton.jit
def _q_kernel(
    QKV, WN, COS, SIN, POS, OUT, T, eps,
    STRIDE: tl.constexpr, D: tl.constexpr, NH: tl.constexpr, BH: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    keep = (heads < NH)[:, None]

    d = tl.arange(0, D)
    partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
    lane = heads.to(tl.int64)[:, None] * D
    src = QKV + row.to(tl.int64) * STRIDE + lane

    x = tl.load(src + d[None, :], mask=keep, other=0.0).to(tl.float32)
    xp = tl.load(src + partner[None, :], mask=keep, other=0.0).to(tl.float32)
    w = tl.load(WN + d).to(tl.float32)[None, :]
    wp = tl.load(WN + partner).to(tl.float32)[None, :]

    p = tl.load(POS + (row % T)).to(tl.int64)
    cos = tl.load(COS + p * D + d).to(tl.float32)[None, :]
    sin = tl.load(SIN + p * D + d).to(tl.float32)[None, :]

    y = _norm_rope(x, xp, w, wp, cos, sin, (d < D // 2)[None, :], eps, D)
    tl.store(OUT + row.to(tl.int64) * (NH * D) + lane + d[None, :], y, mask=keep)


@triton.jit
def _kv_kernel(
    QKV, WN, COS, SIN, POS, KC, VC, T, LMAX, eps,
    STRIDE: tl.constexpr, K_OFF: tl.constexpr, V_OFF: tl.constexpr,
    D: tl.constexpr, NH: tl.constexpr, BH: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    keep = (heads < NH)[:, None]

    d = tl.arange(0, D)
    partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
    lane = heads.to(tl.int64)[:, None] * D
    src = QKV + row.to(tl.int64) * STRIDE + lane

    x = tl.load(src + K_OFF + d[None, :], mask=keep, other=0.0).to(tl.float32)
    xp = tl.load(src + K_OFF + partner[None, :], mask=keep, other=0.0).to(tl.float32)
    w = tl.load(WN + d).to(tl.float32)[None, :]
    wp = tl.load(WN + partner).to(tl.float32)[None, :]

    p = tl.load(POS + (row % T)).to(tl.int64)
    cos = tl.load(COS + p * D + d).to(tl.float32)[None, :]
    sin = tl.load(SIN + p * D + d).to(tl.float32)[None, :]

    y = _norm_rope(x, xp, w, wp, cos, sin, (d < D // 2)[None, :], eps, D)

    seq = (row // T).to(tl.int64)
    slot = ((seq * NH + heads.to(tl.int64)[:, None]) * LMAX + p) * D + d[None, :]
    tl.store(KC + slot, y, mask=keep)
    tl.store(VC + slot, tl.load(src + V_OFF + d[None, :], mask=keep, other=0.0), mask=keep)


def q_norm_rope(
    qkv: torch.Tensor,
    n_heads: int,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    eps: float,
) -> torch.Tensor:
    """Q from the first ``n_heads * 128`` columns of ``qkv``, normed and rotated."""
    rows, stride = qkv.shape
    out = torch.empty((rows, n_heads * _HEAD_DIM), dtype=qkv.dtype, device=qkv.device)
    block = min(_HEADS_PER_PROGRAM, n_heads)
    _q_kernel[(rows, triton.cdiv(n_heads, block))](
        qkv, weight, cos, sin, positions, out, seq_len, eps,
        STRIDE=stride, D=_HEAD_DIM, NH=n_heads, BH=block, num_warps=4,
    )
    return out


def kv_norm_rope_to_cache(
    qkv: torch.Tensor,
    k_offset: int,
    n_heads: int,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
) -> None:
    """Norm and rotate K, copy V, and write both to the slots ``positions`` names.

    ``qkv`` holds K at column ``k_offset`` and V immediately after it. The
    caches are ``[batch, n_heads, capacity, 128]`` contiguous; ``seq_len`` is
    how many rows belong to each sequence, so ``row // seq_len`` is its index.
    """
    rows, stride = qkv.shape
    block = min(_HEADS_PER_PROGRAM, n_heads)
    _kv_kernel[(rows, triton.cdiv(n_heads, block))](
        qkv, weight, cos, sin, positions, k_cache, v_cache,
        seq_len, k_cache.shape[2], eps,
        STRIDE=stride, K_OFF=k_offset, V_OFF=k_offset + n_heads * _HEAD_DIM,
        D=_HEAD_DIM, NH=n_heads, BH=block, num_warps=4,
    )
