"""Controlled prefill-graph and W8A8 arithmetic probes; never submitted."""
import functools
import json
import math
import os
import statistics
import sys
import time
import zlib

os.environ['DRYFT_SHORT_DRAFT'] = '0'
os.environ['DRYFT_PREFILL_GRAPH'] = 'off'
os.environ['DRYFT_INT8_MMA'] = 'off'
os.environ['DRYFT_INT8_NORM_FUSION'] = 'off'
sys.path.insert(0, '/root/engine')
sys.path.insert(0, '/root')

import torch
import triton
import triton.language as tl
from engine import Engine
from harness import _prompts, _replay, _time_stream, load_corpus


@triton.jit
def _integer_reference(X, W, S, Y, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, COMPONENTS: tl.constexpr,
                       BN: tl.constexpr, GROUPS_PER_SPLIT: tl.constexpr):
    # Prepared's column permutation is inverted on load; no weight rounding.
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 64)
    perm = ((k % 8) // 2) * 16 + (k // 16) * 4 + ((k % 16) // 8) * 2 + k % 2
    part = tl.program_id(2)
    acc = tl.zeros((16, BN), tl.float32)
    for g in range(part * GROUPS_PER_SPLIT, tl.minimum((part + 1) * GROUPS_PER_SPLIT, K // 64)):
        x = tl.load(X + m[:, None] * K + g * 64 + k[None, :],
                    mask=m[:, None] < M, other=0)
        w = (tl.load(W + n[None, :] * K + g * 64 + perm[:, None],
                     mask=n[None, :] < N, other=128).to(tl.int32) - 128).to(tl.int8)
        sw = tl.load(S + n * (K // 64) + g, mask=n < N, other=0).to(tl.float32)
        if COMPONENTS == 0:
            block = tl.dot(x, w.to(tl.bfloat16))
        else:
            xf = x.to(tl.float32)
            sx = tl.maximum(tl.max(tl.abs(xf), axis=1) / 127., 1.e-20)
            q = tl.extra.cuda.libdevice.nearbyint(xf / sx[:, None]).to(tl.int8)
            block = tl.dot(q, w).to(tl.float32) * sx[:, None]
            if COMPONENTS == 2:
                residual = xf - q.to(tl.float32) * sx[:, None]
                sr = tl.maximum(tl.max(tl.abs(residual), axis=1) / 127., 1.e-20)
                qr = tl.extra.cuda.libdevice.nearbyint(residual / sr[:, None]).to(tl.int8)
                block += tl.dot(qr, w).to(tl.float32) * sr[:, None]
        acc += block * sw[None, :]
    tl.store(Y + part * M * N + m[:, None] * N + n[None, :], acc,
             mask=(m[:, None] < M) & (n[None, :] < N))


def integer_reference(x, prepared, components):
    m, k = x.shape
    n = prepared.rows
    partials = torch.empty((4, m, n), device=x.device, dtype=torch.float32)
    _integer_reference[(triton.cdiv(m, 16), triton.cdiv(n, 32), 4)](
        x, prepared.weight, prepared.scale, partials, m, n, k, components,
        32, triton.cdiv(k // 64, 4), num_warps=4, enable_fp_fusion=False)
    # Match the current CUDA split-K reduction's sequential addition.
    return ((partials[0] + partials[1]) + partials[2] + partials[3]).to(torch.bfloat16)


@torch.inference_mode()
def check_reference():
    from kernels import cuda_fp8
    torch.manual_seed(71)
    for m in (1, 3, 16, 32):
        x = torch.randn(m, 256, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(64, 256, device='cuda', dtype=torch.bfloat16)
        packed, scale = cuda_fp8.quantize(w)
        prepared = cuda_fp8.prepare((packed, scale))
        wg = (packed.float() - 128).view(64, 4, 64)
        xg = x.float().view(m, 4, 64)
        for components in (0, 1, 2):
            reconstructed = xg
            if components:
                sx = xg.abs().amax(-1, keepdim=True).clamp_min(1.e-20) / 127
                q = (xg / sx).round()
                reconstructed = q * sx
                if components == 2:
                    r = xg - reconstructed
                    sr = r.abs().amax(-1, keepdim=True).clamp_min(1.e-20) / 127
                    reconstructed = reconstructed + (r / sr).round() * sr
            expected = torch.einsum('mgk,ngk,ng->mn', reconstructed, wg, scale.float()).bfloat16()
            actual = integer_reference(x, prepared, components)
            error = (actual.float() - expected.float()).abs().max().item()
            assert error <= 0.25, (m, components, error)
    print('integer reference: 12 independent arithmetic checks passed', flush=True)


@torch.inference_mode()
def check_integer_norm():
    from kernels import cuda_int8
    from kernels.norm import add_rms_norm_partials
    for m in (1, 3, 16, 24, 32):
        for splits in (1, 2, 4, 8):
            x = torch.randn(m, 2560, device='cuda', dtype=torch.bfloat16)
            delta = torch.randn(splits, m, 2560, device='cuda', dtype=torch.float32)
            weight = torch.randn(2560, device='cuda', dtype=torch.bfloat16)
            want_res, want_norm = add_rms_norm_partials(x, delta, weight, 1.e-6)
            expected = cuda_int8.quantize_activation(want_norm)
            got_res, actual = cuda_int8.add_norm_quant(x, delta, weight, 1.e-6)
            assert torch.equal(got_res, want_res), (m, splits, 'residual')
            assert torch.equal(actual[0], expected[0]), (m, splits, 'packed activation')
            assert torch.equal(actual[1], expected[1]), (m, splits, 'activation scales')
    print('integer fused norm: 20 bit-exact residual, activation and scale checks passed', flush=True)


def stats(records, batch, steps):
    total = [r['seconds'] for r in records]
    median = statistics.median(total)
    return {'tps': batch * steps / median, 'seconds': median,
            'ttft': statistics.median(r['ttft'] for r in records),
            'spread': (max(total) - min(total)) / median,
            'worst_gap': max(r['gap'] for r in records),
            'passes_accuracy': all(math.isfinite(r['gap']) and r['gap'] <= 2 for r in records),
            'samples': records}


@torch.inference_mode()
def probe(stage, shape, samples):
    name, batch, context, steps = shape
    load_corpus('/root/corpus.txt', '/weights/qwen3-4b')
    started = time.perf_counter()
    engine = Engine('/weights/qwen3-4b')
    seed = zlib.crc32(name.encode()) % 10**6
    prompts = [_prompts(batch, context, engine.embed.shape[0], seed + i + 1)
               for i in range(samples)]
    list(engine.generate(_prompts(batch, context, engine.embed.shape[0], seed), steps))
    assert engine._fast and engine.graph is not None
    result = {'shape': shape, 'gpu': torch.cuda.get_device_name(), 'stage': stage,
              'load_warmup_seconds': time.perf_counter() - started}
    if stage == 'graph':
        engine._capture_prefill()
        graph = engine.prefill_graph
        records = {'eager': [], 'graph': []}
        for i, prompt in enumerate(prompts):
            outputs = {}
            for label in (('eager', 'graph') if i % 2 == 0 else ('graph', 'eager')):
                engine.prefill_graph = graph if label == 'graph' else None
                ttft, seconds, emitted = _time_stream(engine.generate, prompt, steps)
                outputs[label] = emitted
                gap, exact = _replay(engine._native, engine.device, prompt, emitted, steps)
                records[label].append({'ttft': ttft, 'seconds': seconds, 'gap': gap, 'exact': exact})
            assert outputs['eager'] == outputs['graph'], 'prefill graph changed generated tokens'
            print(name, i, {k: records[k][-1] for k in records}, flush=True)
        result['variants'] = {k: stats(v, batch, steps) for k, v in records.items()}
        result['graph_capture_total_seconds'] = time.perf_counter() - started
    elif stage == 'isolate':
        import engine as engine_module
        from kernels import cuda_int8
        original = {k: getattr(engine, k) for k in
                    ('matmul', 'operand', 'partial_config', 'fused_mlp', 'fused_operand',
                     'integer_mlp', 'integer_head')}
        packed = {n: [cuda_int8.Prepared(w) for w in engine.quantised[n]]
                  for n in ('gate_up', 'lm_head')}
        result['variants'] = {}
        for label, gate, head, components in [('head-only', False, True, 1),
                                              ('gate-only', True, False, 1),
                                              ('gate-residual-head', True, True, 2)]:
            for k, v in original.items():
                setattr(engine, k, v)
            engine.matmul, engine.operand = dict(original['matmul']), dict(original['operand'])
            engine_module.INT8_NORM_FUSION = components == 1
            if gate:
                engine.fused_mlp = functools.partial(cuda_int8.matmul, config=(8, 1), components=components, mode=2)
                engine.fused_operand = packed['gate_up']
                engine.integer_mlp = True
            if head:
                engine.matmul['lm_head'] = functools.partial(cuda_int8.matmul, config=(8, 1))
                engine.operand['lm_head'] = packed['lm_head']
                engine.integer_head = True
            engine._capture()
            records = []
            for i, prompt in enumerate(prompts):
                ttft, seconds, emitted = _time_stream(engine.generate, prompt, steps)
                gap, exact = _replay(engine._native, engine.device, prompt, emitted, steps)
                row = {'ttft': ttft, 'seconds': seconds, 'gap': gap, 'exact': exact}
                records.append(row)
                print(name, label, i, row, flush=True)
            result['variants'][label] = stats(records, batch, steps)
    elif stage == 'ablation':
        import engine as engine_module
        check_integer_norm()
        graphs = {'baseline': engine.graph}
        engine._choose_integer_matmuls(batch)
        engine._capture()
        graphs['integer'] = engine.graph
        engine_module.INT8_NORM_FUSION = True
        engine._capture()
        graphs['integer_fused'] = engine.graph
        records = {name: [] for name in graphs}
        for i, prompt in enumerate(prompts):
            order = list(graphs)
            order = order[i % len(order):] + order[:i % len(order)]
            for label in order:
                engine.graph = graphs[label]
                ttft, seconds, emitted = _time_stream(engine.generate, prompt, steps)
                gap, exact = _replay(engine._native, engine.device, prompt, emitted, steps)
                row = {'ttft': ttft, 'seconds': seconds, 'gap': gap, 'exact': exact}
                records[label].append(row)
                print(name, label, i, row, flush=True)
        result['variants'] = {k: stats(v, batch, steps) for k, v in records.items()}
    elif stage in ('native', 'fused'):
        engine._choose_integer_matmuls(batch)
        if stage == 'fused':
            import engine as engine_module
            check_integer_norm()
            engine_module.INT8_NORM_FUSION = True
        engine._capture()
        import transformers.models.qwen3.modeling_qwen3 as qwen
        result['variants'] = {}
        for corpus, path in [('prose', '/root/corpus.txt'), ('code', qwen.__file__),
                             ('technical', '/root/technical.txt')]:
            load_corpus(path, '/weights/qwen3-4b')
            records = []
            for i in range(samples):
                prompt = _prompts(batch, context, engine.embed.shape[0], 4100+i)
                ttft, seconds, emitted = _time_stream(engine.generate, prompt, steps)
                gap, exact = _replay(engine._native, engine.device, prompt, emitted, steps)
                row = {'ttft': ttft, 'seconds': seconds, 'gap': gap, 'exact': exact}
                records.append(row)
                print(name, corpus, i, row, flush=True)
            result['variants'][corpus] = stats(records, batch, steps)
    else:
        check_reference()
        original = {k: getattr(engine, k) for k in
                    ('matmul', 'operand', 'partial_config', 'fused_mlp', 'fused_operand', 'graph')}
        variants = [('current', None, set()), ('reference-a16', 0, set(engine.quantised)),
                    ('a8-all', 1, set(engine.quantised)),
                    ('a8-attn', 1, {'qkv', 'o'}),
                    ('a8-mlp', 1, {'gate_up', 'down'}),
                    ('a8-residual', 2, set(engine.quantised))]
        result['variants'] = {}
        for label, components, selected in variants:
            for k, v in original.items():
                setattr(engine, k, v)
            if components is not None:
                engine.matmul = dict(original['matmul'])
                engine.operand = dict(original['operand'])
                engine.partial_config = {k: v for k, v in original['partial_config'].items() if k not in selected}
                if 'gate_up' in selected:
                    engine.fused_mlp = None
                for projection in selected:
                    engine.matmul[projection] = functools.partial(integer_reference, components=components)
                    engine.operand[projection] = engine.quantised[projection]
                engine._capture()
            records = []
            for i, prompt in enumerate(prompts):
                ttft, seconds, emitted = _time_stream(engine.generate, prompt, steps)
                gap, exact = _replay(engine._native, engine.device, prompt, emitted, steps)
                row = {'ttft': ttft, 'seconds': seconds, 'gap': gap, 'exact': exact}
                records.append(row)
                print(name, label, i, row, flush=True)
            result['variants'][label] = stats(records, batch, steps)
    result['peak_memory_fraction'] = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
    return result


if __name__ == '__main__':
    print('RESULT_JSON=' + json.dumps(probe(sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]))))
