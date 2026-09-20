"""Fresh-process profiler; production engine settings, no engine modifications."""
import gzip
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, "/root")
sys.path.insert(0, "/root/profile_engine")
from engine import Engine
from harness import _prompts, _time_stream, _replay, load_corpus, _validate_stream
from replay import FixedDecodeReplay


def wall_samples(fn, repeats=9, group=1):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(group):
            fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000 / group)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def capture(fn, name, stage, repeats):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    path = f"/tmp/{name}-{stage}.json"
    prof.export_chrome_trace(path)
    trace_bytes = Path(path).read_bytes()
    Path(path + ".gz").write_bytes(gzip.compress(trace_bytes))
    events = json.loads(trace_bytes)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    totals = {}
    for e in kernels:
        row = totals.setdefault(e["name"], {"name": e["name"], "calls": 0, "ms": 0})
        row["calls"] += 1 / repeats
        row["ms"] += e["dur"] / 1000 / repeats
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    union = 0
    if intervals:
        left, right = intervals[0]
        for start, end in intervals[1:]:
            if start > right:
                union += right - left
                left, right = start, end
            else:
                right = max(right, end)
        union += right - left
    return {"kernels": sorted(totals.values(), key=lambda x: -x["ms"]),
            "summed_kernel_ms": sum(r["ms"] for r in totals.values()),
            "kernel_union_ms": union / 1000 / repeats,
            "projection_gpu_ms": {e.key.split("/", 1)[1]: e.device_time_total / 1000 / repeats
                                  for e in prof.key_averages() if e.key.startswith("projection/")}}


@torch.inference_mode()
def main():
    name, batch, context, steps = sys.argv[1:]
    batch, context, steps = int(batch), int(context), int(steps)
    torch.manual_seed(42)
    load_corpus("/root/corpus.txt", "/weights/qwen3-4b")
    start = time.perf_counter()
    engine = Engine("/weights/qwen3-4b")
    warmup = _prompts(batch, context, engine.embed.shape[0], 9000)
    list(engine.generate(warmup, steps))
    assert engine._fast and engine.graph is not None
    load_warmup_ms = (time.perf_counter() - start) * 1000
    samples, records = [], []
    for seed in range(9001, 9006):
        prompts = _prompts(batch, context, engine.embed.shape[0], seed)
        first, total, emitted = _time_stream(engine.generate, prompts, steps)
        _validate_stream(emitted, batch, steps, engine.embed.shape[0])
        samples.append({"ttft_ms": first * 1000, "total_ms": total * 1000,
                        "tpot_ms": (total-first)*1000/(steps-1), "tok_s": batch*steps/total,
                        "speculation": dict(engine.short_verifier.stats) if engine.short_verifier else None})
        records.append((prompts, emitted))
    ids = engine._upload(records[0][0], batch, context)
    prefill_wall = wall_samples(lambda: engine._prefill(ids))
    fixed = FixedDecodeReplay(engine)
    decode_wall = wall_samples(fixed.replay, group=20)
    expected = engine.emitted.clone()
    for _ in range(65):
        fixed.replay()
    torch.cuda.synchronize()
    assert engine.pos.item() == fixed.position and torch.equal(engine.emitted, expected)
    decode_trace = capture(fixed.replay, name, "decode", 20)
    names = {getattr(layer, n).data_ptr(): n for layer in engine.layers
             for n in ("qkv", "o", "gate_up", "down")}
    names[engine.embed.data_ptr()] = "lm_head"
    original = F.linear
    def tagged(x, weight, *args, **kwargs):
        with record_function("projection/" + names.get(weight.data_ptr(), "unknown")):
            return original(x, weight, *args, **kwargs)
    F.linear = tagged
    try:
        prefill_trace = capture(lambda: engine._prefill(ids), name, "prefill", 3)
    finally:
        F.linear = original
    memory = torch.cuda.max_memory_allocated()
    # Validate after all timing so native forwards cannot selectively warm it.
    for sample, (prompts, emitted) in zip(samples, records):
        gap, exact = _replay(engine._native, engine.device, prompts, emitted, steps)
        sample.update(tie_gap=gap, argmax_exact=exact, replay_pass=gap <= 2)
    result = {"name": name, "batch": batch, "context": context, "output": steps,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "load_warmup_ms": load_warmup_ms, "peak_allocated_bytes_before_replay": memory,
              "environment_overrides": {k:v for k,v in os.environ.items() if k.startswith("DRYFT_")},
              "families": engine.families, "integer_mlp": engine.integer_mlp,
              "partial_configs": engine.partial_config,
              "matmuls": {n:repr(f) for n,f in engine.matmul.items()},
              "fused_mlp": repr(engine.fused_mlp),
              "attention": {n:getattr(engine.decode_attention, n) for n in ("splits", "block_n", "warps")},
              "fixed_decode_position": fixed.position,
              "samples": samples, "prefill_wall": prefill_wall, "decode_wall": decode_wall,
              "prefill_trace": prefill_trace, "decode_trace": decode_trace}
    Path(f"/tmp/{name}.json").write_text(json.dumps(result, indent=2))
    print(f"PROFILE {name}: prefill {prefill_wall['median_ms']:.3f} ms, ordinary decode "
          f"{decode_wall['median_ms']:.3f} ms, replay worst {max(s['tie_gap'] for s in samples)}", flush=True)


if __name__ == "__main__":
    main()
