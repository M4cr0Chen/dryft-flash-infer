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
