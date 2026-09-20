"""H100 investigations, outside the submitted engine.

Each invocation owns one engine and one workload. Decode repeats restore their
position and token; prefill repeats overwrite the same complete prompt cache.
"""

import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/engine")
from engine import Engine
from harness import _prompts, load_corpus
from kernels.timing import time_calls
from kernels.swiglu import swiglu
from replay import FixedDecodeReplay


def milliseconds(fn, repeats=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - start))
    return statistics.median(values)


@torch.inference_mode()
def profile_mlp(engine, prompts, steps):
    from torch.profiler import ProfilerActivity, profile, record_function

    batch, context = len(prompts), len(prompts[0])
    list(engine.generate(prompts, steps))
    ids = engine._upload(prompts, batch, context)
    prefill_ms = milliseconds(lambda: engine._prefill(ids))

    # Attribute eager prefill's CUDA work to projections, even when multiple
    # projections dispatch to the same cuBLAS kernel name.
    names = {w.data_ptr(): name for layer in engine.layers
             for name in ("qkv", "o", "gate_up", "down")
             for w in [getattr(layer, name)]}
    names[engine.embed.data_ptr()] = "lm_head"
    linear = F.linear

    def tagged_linear(x, w, *args, **kwargs):
        with record_function("projection/" + names.get(w.data_ptr(), "unknown")):
            return linear(x, w, *args, **kwargs)

    F.linear = tagged_linear
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                engine._prefill(ids)
            torch.cuda.synchronize()
    finally:
        F.linear = linear
    projection_ms = {e.key.removeprefix("projection/"): e.device_time_total / 3000
                     for e in prof.key_averages() if e.key.startswith("projection/")}
    prefill_kernels = sorted(
        [{"name": e.key, "ms": e.self_device_time_total / 3000, "calls": e.count / 3}
         for e in prof.key_averages() if e.self_device_time_total > 0],
        key=lambda e: e["ms"], reverse=True,
    )[:16]

    fixed = FixedDecodeReplay(engine)
    decode_ms = milliseconds(fixed.replay)
    expected = engine.emitted.clone()
    for _ in range(65):
        fixed.replay()
    torch.cuda.synchronize()
    assert engine.pos.item() == fixed.position and torch.equal(engine.emitted, expected)
    decode_projections = {}
    for name in ("qkv", "o", "gate_up", "down", "lm_head"):
        weights = [engine.embed] if name == "lm_head" else [getattr(l, name) for l in engine.layers]
        x = torch.randn(batch, weights[0].shape[1], dtype=torch.bfloat16, device="cuda")
        runner = engine.matmul[name]
        operands = engine.operand[name]
        value = time_calls(lambda w: runner(x, w), operands)
        row = {"ms_all_layers": value * len(weights), "runner": repr(runner)}
        if name == "gate_up":
            row["with_swiglu_ms"] = time_calls(lambda w: swiglu(runner(x, w)), operands) * len(weights)
            row["fused_runner"] = repr(engine.fused_mlp)
            if engine.fused_mlp is not None:
                row["fused_ms"] = time_calls(lambda w: engine.fused_mlp(x, w), weights) * len(weights)
        decode_projections[name] = row
    result = {
        "batch": batch, "context": context, "output": steps,
        "prefill_wall_ms": prefill_ms, "prefill_projection_gpu_ms": projection_ms,
        # CPU parent scopes also carry attributed device time. These events
        # overlap; only the separate projection totals are disjoint.
        "prefill_profiler_events": prefill_kernels, "decode_wall_ms": decode_ms,
        "decode_projections": decode_projections,
        "attention_splits": engine.decode_attention.splits,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    }
    print(f"profile b{batch} s{context}: prefill={prefill_ms:.3f} ms, "
          f"decode={decode_ms:.3f} ms; MLP prefill="
          f"{projection_ms['gate_up'] + projection_ms['down']:.3f} ms", flush=True)
    return result


@torch.inference_mode()
def attention_study(engine, prompts, steps):
    from kernels.attention import DecodeAttention

    batch, context = len(prompts), len(prompts[0])
    list(engine.generate(prompts, steps))
    original = engine.decode_attention
    group = engine.n_q // engine.n_kv
    q = torch.randn(batch, engine.q_width, device="cuda", dtype=torch.bfloat16)
    operands = list(zip(engine.k_cache, engine.v_cache))
    position = torch.tensor([context], device="cuda", dtype=torch.int32)
    positions = [context, context + steps - 1]
    references = []
    for value in positions:
        position.fill_(value)
        references.append(original(q, *operands[0], position).clone())

    def measure(attention):
        times = []
        errors = []
        for value, reference in zip(positions, references):
            position.fill_(value)
            actual = attention(q, *operands[0], position)
            error = (actual.float() - reference.float()).abs().max().item()
            if not torch.isfinite(actual).all() or error > 0.04:
                raise AssertionError(f"attention output mismatch: {error}")
            times.append(time_calls(lambda kv: attention(q, *kv, position), operands,
                                    reps=2, trials=5) * engine.n_layers)
            errors.append(error)
        return times, errors

    incumbent, _ = measure(original)
    rows = []
    for splits in (1, 2, 4, 8, 16):
        for block in (32, 64, 128):
            for warps in (4, 8):
                config = (splits, block, warps)
                attention = DecodeAttention(batch, engine.n_kv, group, engine.head_dim,
                                            engine.capacity, engine.device, config=config)
                times, errors = measure(attention)
                row = {"config": config, "ms_36_layers": times, "max_abs_errors": errors}
                rows.append(row)
                print(f"attention b{batch} {config}: {times}", flush=True)
    rows.sort(key=lambda r: sum(r["ms_36_layers"]))
    best = rows[0]
    # The whole graph is the deciding measurement; isolated kernel speedups
    # need not carry over when mixed with the model's weight traffic.
    engine._prefill(engine._upload(prompts, batch, context))
    before = FixedDecodeReplay(engine)
    before_ms = milliseconds(before.replay, 15)
    engine.decode_attention = DecodeAttention(
        batch, engine.n_kv, group, engine.head_dim, engine.capacity, engine.device,
        config=best["config"],
    )
    after = FixedDecodeReplay(engine)
    after_ms = milliseconds(after.replay, 15)
    # Alternate again to reveal drift and compare model outputs for one exact
    # prefix. The final harness still checks every generated position.
    before.replay()
    reference_token = engine.emitted.clone()
    after.replay()
    torch.cuda.synchronize()
    token_match = torch.equal(reference_token, engine.emitted)
    repeated_after = milliseconds(after.replay, 15)
    repeated_before = milliseconds(before.replay, 15)
    result = {
        "batch": batch, "context": context, "output": steps,
        "baseline_splits": original.splits, "baseline_ms_36_layers": incumbent,
        "candidates": rows, "selected": best["config"],
        "decode_before_ms": [before_ms, repeated_before],
        "decode_after_ms": [after_ms, repeated_after], "token_match": token_match,
    }
    print(f"attention b{batch} s{context}: selected={best['config']}; "
          f"decode {result['decode_before_ms']} -> {result['decode_after_ms']}", flush=True)
    return result


@torch.inference_mode()
def speculation_study(engine, prompts, steps):
    import inspect
    import math
    from pathlib import Path
    import harness
    from harness import _time_stream, _replay, _validate_stream
    from kernels.speculation import ShortVerifier
    from transformers import AutoTokenizer
    from transformers.models.qwen3 import modeling_qwen3

    context = len(prompts[0])
    list(engine.generate(prompts, steps))
    verifier = ShortVerifier(engine)
    tokenizer = AutoTokenizer.from_pretrained("/weights/qwen3-4b", local_files_only=True)
    corpora = {
        "prose": harness.CORPUS,
        "code": tokenizer(inspect.getsource(modeling_qwen3))["input_ids"],
        "technical": tokenizer(Path("/root/technical.txt").read_text(encoding="utf-8"))["input_ids"],
    }
    variants = [
        ("draft1_order2_cap1.15", 1, 2, 1.15),
        ("draft2_order2_cap1.15", 2, 2, 1.15),
        ("draft2_order3_cap1.15", 2, 3, 1.15),
        ("draft2_order3_unlimited", 2, 3, 99.0),
    ]
    rows = []
    for domain, corpus in corpora.items():
        harness.CORPUS = corpus
        for seed in range(5):
            inputs = _prompts(1, context, engine.embed.shape[0], 4100 + seed)
            # Rotate the baseline through the trial order to reduce drift bias.
            order = [("baseline", 0, 0, 0)] + variants
            shift = seed % len(order)
            order = order[shift:] + order[:shift]
            records = []
            for label, draft, ngram_order, target in order:
                fn = (engine.generate if not draft else
                      lambda p, s, d=draft, o=ngram_order, t=target:
                      verifier.generate(p, s, draft=d, order=o, target=t))
                first, total, emitted = _time_stream(fn, inputs, steps)
                _validate_stream(emitted, 1, steps, engine.embed.shape[0])
                records.append((label, first, total, emitted,
                                dict(verifier.stats) if draft else None))
            baseline_tokens = next(r[3] for r in records if r[0] == "baseline")
            # Check after all timed variants so reference execution cannot
            # affect the comparison order or GPU state between candidates.
            for label, first, total, emitted, stats in records:
                gap, exact = _replay(engine._native, engine.device, inputs, emitted, steps)
                passed = math.isfinite(gap) and gap <= 2.0
                failure = None
                if not passed:
                    tokens = torch.tensor(emitted, device=engine.device).T
                    full = torch.cat([torch.tensor(inputs, device=engine.device), tokens], dim=1)
                    logits = engine._native(input_ids=full, logits_to_keep=steps + 1).logits[:, :steps].float()
                    gaps = logits.max(dim=-1).values - logits.gather(2, tokens[:, :, None])[:, :, 0]
                    bad = (gaps[0] > 2.0).nonzero().flatten().tolist()
                    failure = {"positions": bad, "gaps": [gaps[0, i].item() for i in bad],
                               "tokens": [emitted[i][0] for i in bad],
                               "expected": [logits[0, i].argmax().item() for i in bad]}
                    print(f"FAIL {domain}/{seed}/{label}: {failure}; stats={stats}", flush=True)
                row = {"domain": domain, "seed": seed, "variant": label,
                       "ttft_ms": first * 1000, "total_ms": total * 1000,
                       "tps": steps / total, "tie_gap": gap, "argmax_exact": exact,
                       "matches_ordinary_tokens": emitted == baseline_tokens, "stats": stats,
                       "passes": passed, "failure": failure}
                rows.append(row)
            print(f"speculation s{context} o{steps} {domain} seed={seed}: replayed", flush=True)
    summary = []
    for domain in corpora:
        baseline = statistics.median(r["total_ms"] for r in rows
                                     if r["domain"] == domain and r["variant"] == "baseline")
        for label in ["baseline"] + [v[0] for v in variants]:
            timings = [r["total_ms"] for r in rows if r["domain"] == domain and r["variant"] == label]
            median = statistics.median(timings)
            checks = [r for r in rows if r["domain"] == domain and r["variant"] == label]
            summary.append({"domain": domain, "variant": label, "median_ms": median,
                            "speedup": baseline / median,
                            "teacher_forced_passes": all(r["passes"] for r in checks),
                            "worst_tie_gap": max(r["tie_gap"] for r in checks),
                            "spread": (max(timings) - min(timings)) / median})
    for row in summary:
        print(f"{row['domain']:10s} {row['variant']:28s} "
              f"{row['speedup']:.3f}x spread={row['spread']:.3f} "
              f"replay={'pass' if row['teacher_forced_passes'] else 'FAIL'}", flush=True)
    return {"batch": 1, "context": context, "output": steps, "samples": rows,
            "summary": summary}


@torch.inference_mode()
def attention_paired(engine, prompts, steps):
    """Isolate dispatch with identical weights/projections and rotated samples."""
    from harness import _time_stream, _replay, _validate_stream

    batch, context = len(prompts), len(prompts[0])
    list(engine.generate(prompts, steps))
    original, baseline_graph = engine.decode_attention, engine.graph
    selected = original.tuned(engine.k_cache, engine.v_cache, context, context + steps - 1)
    engine.decode_attention = selected
    engine._capture()
    candidate_graph = engine.graph
    rows = []
    for seed in range(5):
        inputs = _prompts(batch, context, engine.embed.shape[0], 6800 + seed)
        order = [("baseline", baseline_graph), ("candidate", candidate_graph)]
        if seed % 2:
            order.reverse()
        records = []
        for label, graph in order:
            engine.graph = graph
            first, total, emitted = _time_stream(engine.generate, inputs, steps)
            _validate_stream(emitted, batch, steps, engine.embed.shape[0])
            records.append((label, first, total, emitted))
        for label, first, total, emitted in records:
            gap, exact = _replay(engine._native, engine.device, inputs, emitted, steps)
            if not gap <= 2.0:
                raise AssertionError(f"{label}: invalid teacher-forced gap {gap}")
            rows.append({"variant": label, "seed": seed, "ttft_ms": first * 1000,
                         "total_ms": total * 1000, "tps": batch * steps / total,
                         "tie_gap": gap, "argmax_exact": exact})
    before = statistics.median(r["total_ms"] for r in rows if r["variant"] == "baseline")
    after = statistics.median(r["total_ms"] for r in rows if r["variant"] == "candidate")
    result = {"batch": batch, "context": context, "output": steps,
              "selected": [selected.splits, selected.block_n, selected.warps],
              "speedup": before / after, "baseline_ms": before, "candidate_ms": after,
              "samples": rows}
    print(f"attention paired b{batch} s{context}: {before / after:.4f}x", flush=True)
    return result


if __name__ == "__main__":
    stage, batch, context, steps = sys.argv[1:]
    batch, context, steps = int(batch), int(context), int(steps)
    torch.manual_seed(42)
    # The dispatch sweep and verifier compare against the original attention
    # heuristic, keeping their measurements independent of a new tuner.
    if stage in ("attention", "attention_paired", "speculation", "speculation_debug", "speculation_mlp"):
        import engine as engine_module
        engine_module.TUNE_ATTENTION = False
        engine_module.SHORT_DRAFT = 0
    load_corpus("/root/corpus.txt", "/weights/qwen3-4b")
    engine = Engine("/weights/qwen3-4b")
    if stage == "speculation_mlp":
        engine.quantised = {name: weights for name, weights in engine.quantised.items()
                            if name in ("gate_up", "down")}
    prompts = _prompts(batch, context, engine.embed.shape[0], 9001)
    if stage == "profile":
        result = profile_mlp(engine, prompts, steps)
    elif stage == "attention":
        result = attention_study(engine, prompts, steps)
    elif stage in ("speculation", "speculation_debug", "speculation_mlp"):
        result = speculation_study(engine, prompts, steps)
    elif stage == "attention_paired":
        result = attention_paired(engine, prompts, steps)
    else:
        raise ValueError(stage)
    print("RESULT_JSON=" + json.dumps(result), flush=True)
