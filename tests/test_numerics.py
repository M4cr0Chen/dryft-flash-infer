"""Bit-exactness of the fused kernels' arithmetic against Transformers 4.51.3.

The Triton kernels cannot run here, but the function they compute can. Each
test expresses a kernel's semantics in plain torch -- same operand dtypes, same
cast placement -- and demands a bit-identical result from the reference module.
That is the part the tie margin actually cares about: a reordering is free, a
reformulation is not.

Run with a CPU torch and transformers==4.51.3:

    python -m pytest tests/test_numerics.py -q
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

BF16 = torch.bfloat16
HIDDEN = 2560
HEAD_DIM = 128
EPS = 1e-6
THETA = 5_000_000.0


def _config(**kw):
    return Qwen3Config(
        hidden_size=HIDDEN, intermediate_size=9728, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=8, head_dim=HEAD_DIM,
        rms_norm_eps=EPS, rope_theta=THETA, **kw,
    )


def _kernel_rms_norm(x, weight):
    """What norm.py's _rms_norm_kernel computes."""
    f = x.to(torch.float32)
    scale = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
    return ((f * scale).to(BF16).to(torch.float32) * weight.to(torch.float32)).to(BF16)


def test_rms_norm_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(64, HIDDEN, dtype=BF16)
    ref = Qwen3RMSNorm(HIDDEN, eps=EPS).to(BF16)
    ref.weight.data = torch.randn(HIDDEN, dtype=BF16)
    with torch.no_grad():
        assert torch.equal(_kernel_rms_norm(x, ref.weight), ref(x))


def test_add_rms_norm_matches_reference():
    """The residual add rounds to bfloat16 before the norm reads it."""
    torch.manual_seed(1)
    x = torch.randn(64, HIDDEN, dtype=BF16)
    delta = torch.randn(64, HIDDEN, dtype=BF16)
    ref = Qwen3RMSNorm(HIDDEN, eps=EPS).to(BF16)
    ref.weight.data = torch.randn(HIDDEN, dtype=BF16)
    with torch.no_grad():
        residual = (x.to(torch.float32) + delta.to(torch.float32)).to(BF16)
        assert torch.equal(residual, x + delta)
        assert torch.equal(_kernel_rms_norm(residual, ref.weight), ref(x + delta))


def test_head_norm_then_rope_matches_reference():
    """q_norm over 128 values, then RoPE, with the partner lane renormalised."""
    torch.manual_seed(2)
    batch, heads, seq = 2, 8, 6
    q = torch.randn(batch, seq, heads, HEAD_DIM, dtype=BF16)
    norm = Qwen3RMSNorm(HEAD_DIM, eps=EPS).to(BF16)
    norm.weight.data = torch.randn(HEAD_DIM, dtype=BF16)

    rotary = Qwen3RotaryEmbedding(_config())
    positions = torch.arange(seq).unsqueeze(0)
    cos, sin = rotary(q.float(), positions)
    cos, sin = cos.to(BF16), sin.to(BF16)

    with torch.no_grad():
        reference, _ = apply_rotary_pos_emb(
            norm(q).transpose(1, 2), norm(q).transpose(1, 2), cos, sin
        )

        # The kernel's view: normalise the lane and its partner with one scale,
        # build rotate_half from the normalised partner, then three roundings.
        f = q.to(torch.float32)
        scale = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
        w = norm.weight.to(torch.float32)
        normed = ((f * scale).to(BF16).to(torch.float32) * w).to(BF16).to(torch.float32)
        partner = torch.cat(
            [-normed[..., HEAD_DIM // 2 :], normed[..., : HEAD_DIM // 2]], dim=-1
        )
        c = cos.unsqueeze(2).to(torch.float32)
        s = sin.unsqueeze(2).to(torch.float32)
        left = (normed * c).to(BF16).to(torch.float32)
        right = (partner * s).to(BF16).to(torch.float32)
        kernel = (left + right).to(BF16).transpose(1, 2)

        assert torch.equal(kernel, reference)


def test_swiglu_matches_reference():
    torch.manual_seed(3)
    config = _config()
    mlp = Qwen3MLP(config).to(BF16)
    x = torch.randn(32, HIDDEN, dtype=BF16)
    with torch.no_grad():
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        reference = mlp.act_fn(gate) * up

        g = gate.to(torch.float32)
        silu = (g / (1.0 + torch.exp(-g))).to(BF16).to(torch.float32)
        kernel = (silu * up.to(torch.float32)).to(BF16)
        assert torch.equal(kernel, reference)


def test_rope_table_matches_reference():
    """A precomputed table must equal what the module builds per call."""
    config = _config()
    rotary = Qwen3RotaryEmbedding(config)
    seq = 300
    x = torch.zeros(1, seq, HIDDEN, dtype=BF16)
    positions = torch.arange(seq).unsqueeze(0)
    ref_cos, ref_sin = rotary(x, positions)

    half = HEAD_DIM // 2
    inv_freq = 1.0 / (THETA ** (torch.arange(0, half, dtype=torch.int64).float() / half))
    freqs = torch.arange(seq, dtype=torch.int64).float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)

    assert torch.equal(emb.cos().to(BF16), ref_cos[0].to(BF16))
    assert torch.equal(emb.sin().to(BF16), ref_sin[0].to(BF16))


def test_fused_projection_matches_separate():
    """Concatenating q/k/v into one weight must not change any output value."""
    torch.manual_seed(4)
    x = torch.randn(48, HIDDEN, dtype=BF16)
    q = torch.randn(4096, HIDDEN, dtype=BF16)
    k = torch.randn(1024, HIDDEN, dtype=BF16)
    v = torch.randn(1024, HIDDEN, dtype=BF16)
    with torch.no_grad():
        fused = torch.nn.functional.linear(x, torch.cat([q, k, v], dim=0))
        assert torch.equal(fused[:, :4096], torch.nn.functional.linear(x, q))
        assert torch.equal(fused[:, 4096:5120], torch.nn.functional.linear(x, k))
        assert torch.equal(fused[:, 5120:], torch.nn.functional.linear(x, v))
