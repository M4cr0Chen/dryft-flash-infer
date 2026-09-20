"""Direct equivalence checks for the actual Triton Q/K/V fusion."""


def check_rope_fusion():
    import sys

    import torch

    sys.path.insert(0, "/root/engine")
    from kernels.rope import q_norm_rope, kv_norm_rope_to_cache, qkv_norm_rope_to_cache

    torch.manual_seed(31)
    for batch in (1, 4, 16):
        for length in (1, 17):
            capacity = length + 12
            qkv = torch.randn(batch * length, 6144, dtype=torch.bfloat16, device="cuda")
            qn = torch.randn(128, dtype=torch.bfloat16, device="cuda")
            kn = torch.randn_like(qn)
            angles = torch.randn(capacity, 128, device="cuda")
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            positions = torch.arange(5, 5 + length, device="cuda", dtype=torch.int32)
            # Surround the destination with untouched batches to detect bad
            # strides/offsets, and compare every slot, not just written ones.
            shape = (batch + 2, 8, capacity, 128)
            want_k = torch.full(shape, -37.0, dtype=torch.bfloat16, device="cuda")
            want_v = torch.full_like(want_k, 41.0)
            got_k, got_v = want_k.clone(), want_v.clone()
            want_q = q_norm_rope(qkv, 32, qn, cos, sin, positions, length, 1e-6)
            kv_norm_rope_to_cache(
                qkv, 4096, 8, kn, cos, sin, positions, length,
                want_k[1:1 + batch], want_v[1:1 + batch], 1e-6,
            )
            got_q = qkv_norm_rope_to_cache(
                qkv, 32, 8, qn, kn, cos, sin, positions, length,
                got_k[1:1 + batch], got_v[1:1 + batch], 1e-6,
            )
            for label, actual, expected in (
                ("Q", got_q, want_q), ("K", got_k, want_k), ("V", got_v, want_v)
            ):
                if not torch.equal(actual, expected):
                    raise AssertionError(f"fused {label} differs at batch={batch}, tokens={length}")
    print("GPU Q/K/V fusion: bit-exact against separate Triton kernels on all 6 cases", flush=True)


def check_attention_dispatch():
    """Check causal boundaries, empty partitions and direct output against SDPA."""
    import sys
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/engine")
    from kernels.attention import DecodeAttention

    torch.manual_seed(47)
    cases, worst = 0, 0.0
    for batch in (1, 4):
        for tokens in (1, 2, 3):
            capacity, n_kv, group, dim = 521, 8, 4, 128
            q = torch.randn(batch * tokens, n_kv * group * dim,
                            device="cuda", dtype=torch.bfloat16)
            keys = torch.randn(batch, n_kv, capacity, dim, device="cuda", dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            for first in (0, 127, capacity - tokens):
                position = torch.tensor([first], device="cuda", dtype=torch.int32)
                k, v = keys.clone(), values.clone()
                k[:, :, first + tokens:] = float("nan")
                v[:, :, first + tokens:] = float("nan")
                # Slice away the poisoned suffix in the independent reference.
                mask = (torch.arange(first + tokens, device="cuda")[None, :]
                        <= first + torch.arange(tokens, device="cuda")[:, None])
                reference = F.scaled_dot_product_attention(
                    q.view(batch, tokens, n_kv * group, dim).transpose(1, 2),
                    keys[:, :, :first + tokens].repeat_interleave(group, 1),
                    values[:, :, :first + tokens].repeat_interleave(group, 1),
                    attn_mask=mask,
                ).transpose(1, 2).reshape(batch * tokens, -1)
                for config in ((1, 64, 4), (4, 128, 4), (16, 64, 4)):
                    attention = DecodeAttention(batch, n_kv, group, dim, capacity,
                                                "cuda", tokens=tokens, config=config)
                    actual = attention(q, k, v, position)
                    error = (actual.float() - reference.float()).abs().max().item()
                    if not torch.isfinite(actual).all() or error > 0.04:
                        raise AssertionError(f"attention {batch=} {tokens=} {first=} {config=}: {error}")
                    worst = max(worst, error)
                    cases += 1
    print(f"GPU attention: {cases} causal/boundary cases passed; max SDPA error={worst}", flush=True)


def check_fp8():
    """Check weight/group indexing, partial tiles and split reduction against GEMM."""
    import sys
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/engine")
    from kernels.fp8 import quantize, fp8_matmul

    torch.manual_seed(59)
    cases, worst = 0, 0.0
    for group in (64, 128):
        weight = torch.randn(73, 384, dtype=torch.bfloat16, device="cuda")
        weight[0].zero_()
        packed, scales = quantize(weight, group=group)
        restored = (packed.float().view(73, -1, group) * scales.float()[:, :, None])
        restored = restored.reshape_as(weight).bfloat16()
        for batch in (1, 2, 3, 4, 16):
            x = torch.randn(batch, 384, dtype=torch.bfloat16, device="cuda")
            expected = F.linear(x, restored)
            scale = expected.float().abs().max().item()
            for config in ((32, 4, 4, 1), (16, 4, 4, 4)):
                actual = fp8_matmul(x, (packed, scales), config=config)
                relative = (actual.float() - expected.float()).abs().max().item() / scale
                if not torch.isfinite(actual).all() or relative > 0.02:
                    raise AssertionError(f"FP8 {batch=} {group=} {config=}: error={relative}")
                if torch.count_nonzero(actual[:, 0]).item():
                    raise AssertionError("FP8 zero row was not preserved")
                worst = max(worst, relative)
                cases += 1
    print(f"GPU FP8: {cases} indexing/reduction cases passed; max relative error={worst:.5f}", flush=True)
