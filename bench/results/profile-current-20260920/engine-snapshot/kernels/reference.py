"""Torch implementations of the fused kernels, op for op.

These stand in when Triton is unavailable, which is how the engine's structure
-- cache slots, positions, masks, the residual chain -- gets tested on a
machine with no GPU. They compute exactly what the Triton kernels compute,
including where each bfloat16 rounding lands, so a divergence between the two
is a bug in one of them and not in the arithmetic.
"""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    f = x.float()
    scale = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return ((f * scale).to(x.dtype).float() * weight.float()).to(x.dtype)


def add_rms_norm(
    x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = x + delta
    return residual, rms_norm(residual, weight, eps)


def add_rms_norm_partials(x, partials, weight, eps):
    delta = partials.sum(dim=0).to(x.dtype)
    return add_rms_norm(x, delta, weight, eps)


def _norm_rope(x, weight, cos, sin, positions, seq_len, eps):
    """``x`` is ``[rows, heads, dim]``; row r holds token ``r % seq_len``."""
    rows, _, dim = x.shape
    half = dim // 2

    f = x.float()
    scale = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    normed = ((f * scale).to(x.dtype).float() * weight.float()).to(x.dtype).float()
    rotated = torch.cat([-normed[..., half:], normed[..., :half]], dim=-1)

    index = positions[:seq_len].long().repeat(rows // seq_len)
    c = cos[index].float().unsqueeze(1)
    s = sin[index].float().unsqueeze(1)
    left = (normed * c).to(x.dtype).float()
    right = (rotated * s).to(x.dtype).float()
    return (left + right).to(x.dtype)


def q_norm_rope(qkv, n_heads, weight, cos, sin, positions, seq_len, eps):
    rows = qkv.shape[0]
    dim = weight.shape[0]
    q = qkv[:, : n_heads * dim].reshape(rows, n_heads, dim)
    out = _norm_rope(q, weight, cos, sin, positions, seq_len, eps)
    return out.reshape(rows, n_heads * dim)


def kv_norm_rope_to_cache(
    qkv, k_offset, n_heads, weight, cos, sin, positions, seq_len,
    k_cache, v_cache, eps,
):
    rows = qkv.shape[0]
    dim = weight.shape[0]
    width = n_heads * dim
    k = qkv[:, k_offset : k_offset + width].reshape(rows, n_heads, dim)
    v = qkv[:, k_offset + width : k_offset + 2 * width].reshape(rows, n_heads, dim)

    rotated = _norm_rope(k, weight, cos, sin, positions, seq_len, eps)
    batch = rows // seq_len
    slots = positions[:seq_len].long()
    k_cache[:, :, slots, :] = rotated.view(batch, seq_len, n_heads, dim).transpose(1, 2)
    v_cache[:, :, slots, :] = v.view(batch, seq_len, n_heads, dim).transpose(1, 2)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    inter = gate_up.shape[1] // 2
    gate = gate_up[:, :inter].float()
    up = gate_up[:, inter:].float()
    silu = (gate / (1.0 + torch.exp(-gate))).to(gate_up.dtype).float()
    return (silu * up).to(gate_up.dtype)


def swiglu_partials(partials):
    total = torch.zeros_like(partials[0])
    for part in partials:
        total += part
    return swiglu(total.to(torch.bfloat16))


def qkv_norm_rope_to_cache(
    qkv, n_q, n_kv, q_weight, k_weight, cos, sin, positions, seq_len,
    k_cache, v_cache, eps, k_scale=None, v_scale=None, k_copy=None, v_copy=None,
):
    if k_scale is not None or k_copy is not None:
        raise NotImplementedError("the INT8 cache needs the Triton kernels")
    q = q_norm_rope(qkv, n_q, q_weight, cos, sin, positions, seq_len, eps)
    kv_norm_rope_to_cache(
        qkv, n_q * q_weight.shape[0], n_kv, k_weight, cos, sin, positions,
        seq_len, k_cache, v_cache, eps,
    )
    return q


def qkv_planes_norm_rope_to_cache(
    planes, n_q, n_kv, q_weight, k_weight, cos, sin, positions, seq_len,
    k_cache, v_cache, eps, k_scale=None, v_scale=None, k_copy=None, v_copy=None,
):
    if k_scale is not None or k_copy is not None:
        raise NotImplementedError("the INT8 cache needs the Triton kernels")
    qkv = planes.sum(dim=0).to(torch.bfloat16)
    return qkv_norm_rope_to_cache(
        qkv, n_q, n_kv, q_weight, k_weight, cos, sin, positions, seq_len,
        k_cache, v_cache, eps,
    )
