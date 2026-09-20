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


def check_fp8_mma():
    """The CUDA FP8 GEMM against the dequantised weight, and the fused plane consumers."""
    import sys
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/engine")
    from kernels import cuda_fp8, fp8
    from kernels.norm import add_rms_norm, add_rms_norm_partials
    from kernels.rope import qkv_norm_rope_to_cache, qkv_planes_norm_rope_to_cache

    if not cuda_fp8.ready():
        raise AssertionError("cuda_fp8 did not compile")
    torch.manual_seed(53)
    cases = 0
    for n, k in ((6144, 2560), (2560, 4096), (2560, 9728), (19456, 2560)):
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        packed = cuda_fp8.quantize(weight)
        prepared = cuda_fp8.prepare(packed)
        exact = cuda_fp8.dequantize(packed)
        for batch in (1, 2, 3, 4, 8, 16, 17, 32):
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            reference = F.linear(x, exact).float()
            scale = reference.abs().max().item()
            for config in cuda_fp8.CONFIGS:
                if not cuda_fp8.applicable(config, batch):
                    continue
                out = cuda_fp8.matmul(x, prepared, config=config).float()
                err = (out - reference).abs().max().item() / scale
                if not torch.isfinite(out).all() or err > 1e-2:
                    raise AssertionError(f"fp8 mma n={n} k={k} b={batch} {config}: rel err {err}")
                cases += 1
            if n == 19456:
                inter = n // 2
                want = (F.silu(reference[:, :inter].to(torch.bfloat16)).to(torch.bfloat16).float()
                        * reference[:, inter:].to(torch.bfloat16).float())
                for config in cuda_fp8.SWIGLU_CONFIGS:
                    if not cuda_fp8.applicable(config, batch, swiglu=True):
                        continue
                    got = cuda_fp8.gate_up_swiglu(x, prepared, config=config).float()
                    err = (got - want).abs().max().item() / want.abs().max().item()
                    if not torch.isfinite(got).all() or err > 2e-2:
                        raise AssertionError(f"fp8 swiglu b={batch} {config}: rel err {err}")
                    cases += 1
            # Planes summed by the consumer must equal the reduced bf16 path exactly.
            for config in ((2, 4), (4, 8), (2, 16)):
                if cuda_fp8.splits(prepared, batch, config) < 2:
                    continue
                planes = cuda_fp8.matmul_partials(x, prepared, config)
                reduced = cuda_fp8.matmul(x, prepared, config)
                # torch sums the planes in its own order; the fused consumers
                # below sum them in the reduce kernel's order and must match it
                # exactly, which is the check that matters.
                if not torch.allclose(planes.sum(0).to(torch.bfloat16).float(), reduced.float(),
                                      rtol=1e-2, atol=1e-2):
                    raise AssertionError(f"planes disagree with the reduce n={n} k={k} b={batch}")
                if n == 2560:
                    residual = torch.randn(batch, n, dtype=torch.bfloat16, device="cuda")
                    gain = torch.randn(n, dtype=torch.bfloat16, device="cuda")
                    r0, y0 = add_rms_norm(residual, reduced, gain, 1e-6)
                    r1, y1 = add_rms_norm_partials(residual, planes, gain, 1e-6)
                    if not (torch.equal(r0, r1) and torch.equal(y0, y1)):
                        raise AssertionError(f"add_rms_norm_partials differs b={batch} {config}")
                    cases += 1
                if n == 6144:
                    capacity, length = 40, 1
                    qn = torch.randn(128, dtype=torch.bfloat16, device="cuda")
                    kn = torch.randn_like(qn)
                    angles = torch.randn(capacity, 128, device="cuda")
                    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
                    positions = torch.tensor([7], device="cuda", dtype=torch.int32)
                    shape = (batch, 8, capacity, 128)
                    k_a = torch.full(shape, -3.0, dtype=torch.bfloat16, device="cuda")
                    v_a = torch.full_like(k_a, 5.0)
                    k_b, v_b = k_a.clone(), v_a.clone()
                    q_a = qkv_norm_rope_to_cache(reduced, 32, 8, qn, kn, cos, sin, positions,
                                                 length, k_a, v_a, 1e-6)
                    q_b = qkv_planes_norm_rope_to_cache(planes, 32, 8, qn, kn, cos, sin,
                                                        positions, length, k_b, v_b, 1e-6)
                    if not (torch.equal(q_a, q_b) and torch.equal(k_a, k_b) and torch.equal(v_a, v_b)):
                        raise AssertionError(f"qkv planes rope differs b={batch} {config}")
                    cases += 1
    print(f"GPU FP8 mma: {cases} cases within tolerance; fused plane consumers bit-exact", flush=True)


def check_kv_int8():
    """INT8 cache: the RoPE kernel's quantised writes and the attention kernel's reads."""
    import sys
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/engine")
    from kernels.attention import DecodeAttention
    from kernels.rope import qkv_norm_rope_to_cache, qkv_planes_norm_rope_to_cache

    torch.manual_seed(61)
    n_kv, group, dim = 8, 4, 128
    cases = 0
    for batch, length in ((1, 1), (4, 1), (2, 9)):
        capacity = 40
        qkv = torch.randn(batch * length, 6144, dtype=torch.bfloat16, device="cuda")
        qn = torch.randn(dim, dtype=torch.bfloat16, device="cuda")
        kn = torch.randn_like(qn)
        angles = torch.randn(capacity, dim, device="cuda")
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        positions = torch.arange(3, 3 + length, device="cuda", dtype=torch.int32)
        shape = (batch, n_kv, capacity, dim)
        k16 = torch.zeros(shape, dtype=torch.bfloat16, device="cuda"); v16 = torch.zeros_like(k16)
        q16 = qkv_norm_rope_to_cache(qkv, 32, 8, qn, kn, cos, sin, positions, length, k16, v16, 1e-6)
        k8 = torch.zeros(shape, dtype=torch.int8, device="cuda"); v8 = torch.zeros_like(k8)
        ks = torch.zeros(shape[:-1], dtype=torch.float32, device="cuda"); vs = torch.zeros_like(ks)
        kc = torch.zeros(batch, n_kv, length, dim, dtype=torch.bfloat16, device="cuda"); vc = torch.zeros_like(kc)
        q8 = qkv_norm_rope_to_cache(qkv, 32, 8, qn, kn, cos, sin, positions, length, k8, v8, 1e-6,
                                    k_scale=ks, v_scale=vs, k_copy=kc, v_copy=vc)
        if not torch.equal(q8, q16):
            raise AssertionError("INT8 cache path changed Q")
        # The bfloat16 copy is exactly what the bfloat16 cache would hold.
        for t, p in enumerate(range(3, 3 + length)):
            if not (torch.equal(kc[:, :, t], k16[:, :, p]) and torch.equal(vc[:, :, t], v16[:, :, p])):
                raise AssertionError("bfloat16 copy differs from the bfloat16 cache")
        # The dequantised cache is within half a step of the bfloat16 cache.
        for c8, sc, c16, label in ((k8, ks, k16, "K"), (v8, vs, v16, "V")):
            deq = c8.float() * sc[..., None]
            written = torch.zeros(capacity, dtype=torch.bool, device="cuda"); written[3:3 + length] = True
            err = (deq - c16.float())[:, :, written].abs()
            bound = (sc[:, :, written] * 0.5 + 1e-6)[..., None]
            if not (err <= bound).all():
                raise AssertionError(f"INT8 {label} quantisation exceeds half a step")
        # planes kernel agrees with the direct kernel on both layouts
        planes = torch.stack([qkv.float() * 0.25, qkv.float() * 0.75])
        k8b = torch.zeros_like(k8); v8b = torch.zeros_like(v8); ksb = torch.zeros_like(ks); vsb = torch.zeros_like(vs)
        kcb = torch.zeros_like(kc); vcb = torch.zeros_like(vc)
        qp = qkv_planes_norm_rope_to_cache(planes, 32, 8, qn, kn, cos, sin, positions, length,
                                           k8b, v8b, 1e-6, k_scale=ksb, v_scale=vsb, k_copy=kcb, v_copy=vcb)
        ref = qkv_norm_rope_to_cache(planes.sum(0).to(torch.bfloat16), 32, 8, qn, kn, cos, sin,
                                     positions, length, k8, v8, 1e-6, k_scale=ks, v_scale=vs, k_copy=kc, v_copy=vc)
        if not (torch.equal(qp, ref) and torch.equal(k8b, k8) and torch.equal(v8b, v8)
                and torch.equal(ksb, ks) and torch.equal(vsb, vs) and torch.equal(kcb, kc)):
            raise AssertionError("planes kernel differs from the direct kernel on the INT8 cache")
        cases += 1

    # Attention over the INT8 cache against SDPA over the dequantised cache.
    worst = 0.0
    for batch in (1, 4):
        for tokens in (1, 2, 3):
            capacity = 521
            q = torch.randn(batch * tokens, n_kv * group * dim, device="cuda", dtype=torch.bfloat16)
            keys = torch.randn(batch, n_kv, capacity, dim, device="cuda", dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            ks = keys.float().abs().amax(-1) / 127.0
            vs = values.float().abs().amax(-1) / 127.0
            k8 = torch.round(keys.float() / ks[..., None]).clamp(-127, 127).to(torch.int8)
            v8 = torch.round(values.float() / vs[..., None]).clamp(-127, 127).to(torch.int8)
            kd = (k8.float() * ks[..., None]).to(torch.bfloat16)
            vd = (v8.float() * vs[..., None]).to(torch.bfloat16)
            for first in (0, 127, capacity - tokens):
                position = torch.tensor([first], device="cuda", dtype=torch.int32)
                mask = (torch.arange(first + tokens, device="cuda")[None, :]
                        <= first + torch.arange(tokens, device="cuda")[:, None])
                reference = F.scaled_dot_product_attention(
                    q.view(batch, tokens, n_kv * group, dim).transpose(1, 2),
                    kd[:, :, :first + tokens].repeat_interleave(group, 1),
                    vd[:, :, :first + tokens].repeat_interleave(group, 1),
                    attn_mask=mask,
                ).transpose(1, 2).reshape(batch * tokens, -1)
                for config in (None, (1, 64, 4), (4, 128, 4), (16, 64, 4)):
                    attention = DecodeAttention(batch, n_kv, group, dim, capacity, "cuda",
                                                tokens=tokens, config=config, quant=True)
                    actual = attention(q, k8, v8, position, ks, vs)
                    err = (actual.float() - reference.float()).abs().max().item()
                    worst = max(worst, err)
                    if not torch.isfinite(actual).all() or err > 0.02:
                        raise AssertionError(f"INT8 attention b={batch} t={tokens} first={first} {config}: err {err}")
                    cases += 1
    print(f"GPU INT8 cache: {cases} cases passed; max attention error vs dequantised SDPA {worst:.5f}", flush=True)


def check_fused_add_norm():
    """GEMM plus residual add plus RMSNorm in one kernel against the two-launch pair."""
    import sys
    import torch

    sys.path.insert(0, "/root/engine")
    from kernels import cuda_fp8
    from kernels.norm import add_rms_norm_partials

    if not cuda_fp8.ready():
        raise AssertionError("cuda_fp8 did not compile")
    torch.manual_seed(71)
    n = 2560
    cases, worst = 0, 0.0
    for k in (4096, 9728):
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        prepared = cuda_fp8.prepare(cuda_fp8.quantize(weight), tiled=False)  # the fused kernel addresses rows
        norm = 1.0 + 0.1 * torch.randn(n, dtype=torch.bfloat16, device="cuda")
        for batch in (1, 3, 4, 8, 16):
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            residual = torch.randn(batch, n, dtype=torch.bfloat16, device="cuda")
            planes = cuda_fp8.matmul_partials(x, prepared, (4, 8))
            want_res, want_norm = add_rms_norm_partials(residual, planes, norm, 1e-6)
            scratch = cuda_fp8.add_norm_scratch(batch, n, "cuda")
            for warps in cuda_fp8.ADD_NORM_CONFIGS:
                for repeat in range(3):   # the ticket counter must reset itself
                    got_res, got_norm = cuda_fp8.matmul_add_norm(x, prepared, residual, norm, 1e-6,
                                                                 scratch, warps=warps)
                    torch.cuda.synchronize()
                    if int(scratch[1].item()) != 0:
                        raise AssertionError("ticket counter did not reset")
                    err_res = (got_res.float() - want_res.float()).abs().max().item() / want_res.float().abs().max().item()
                    err_norm = (got_norm.float() - want_norm.float()).abs().max().item() / want_norm.float().abs().max().item()
                    worst = max(worst, err_res, err_norm)
                    if not torch.isfinite(got_norm).all() or err_res > 1e-2 or err_norm > 1e-2:
                        raise AssertionError(f"fused add-norm k={k} b={batch} w={warps}: {err_res} {err_norm}")
                    cases += 1
    print(f"GPU fused add-norm: {cases} cases; worst relative difference vs planes path {worst:.2e}", flush=True)


def check_separate_decode_norm():
    """Match residual and normalized BF16 outputs across batches and plane counts."""
    import sys
    import torch
    sys.path.insert(0, '/root/engine')
    from kernels.norm import (add_rms_norm, add_rms_norm_partials,
                              add_rms_norm_separate, add_rms_norm_partials_separate)

    torch.manual_seed(812)
    cases = 0
    with torch.inference_mode():
        for batch in (1, 2, 3, 4, 8, 16, 32, 64):
            x = torch.randn(batch, 2560, device='cuda', dtype=torch.bfloat16)
            weight = 1 + .1 * torch.randn(2560, device='cuda', dtype=torch.bfloat16)
            delta = torch.randn_like(x)
            reference = add_rms_norm(x, delta, weight, 1.e-6)
            actual = add_rms_norm_separate(x, delta, weight, 1.e-6)
            assert all(torch.equal(a, b) for a, b in zip(reference, actual)), (batch, 'direct')
            cases += 1
            for splits in (1, 2, 4, 8, 16):
                planes = torch.randn(splits, batch, 2560, device='cuda', dtype=torch.float32)
                reference = add_rms_norm_partials(x, planes, weight, 1.e-6)
                actual = add_rms_norm_partials_separate(x, planes, weight, 1.e-6)
                assert all(torch.equal(a, b) for a, b in zip(reference, actual)), (batch, splits)
                cases += 1
    print(f'GPU separate decode norm: {cases} bit-exact cases passed', flush=True)


def check_partial_swiglu():
    """Compare against the real CUDA reduce followed by the old SwiGLU kernel."""
    import sys
    import torch
    sys.path.insert(0, '/root/engine')
    from kernels import cuda_fp8
    from kernels.swiglu import swiglu, swiglu_partials

    assert cuda_fp8.ready()
    reduce = cuda_fp8._module_for(1, 1).kernel('reduce_partials')
    cases = 0
    with torch.inference_mode():
        for batch in (1, 2, 3, 4, 8, 16, 32, 64):
            for splits in (1, 2, 4, 8, 16):
                planes = torch.randn(splits, batch, 19456, device='cuda', dtype=torch.float32)
                reduced = torch.empty(batch, 19456, device='cuda', dtype=torch.bfloat16)
                width = reduced.numel()
                reduce(-(-width//256), 256, planes, reduced, width, splits)
                expected = swiglu(reduced)
                actual = swiglu_partials(planes)
                assert torch.equal(expected, actual), (batch, splits, (expected-actual).abs().max().item())
                cases += 1
    print(f'GPU partial SwiGLU: {cases} bit-exact cases passed', flush=True)
