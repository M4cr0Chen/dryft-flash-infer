"""A replica of the judge, to run wherever there is an H100.

Same clock, same gates, same correctness rule: time the token stream, replay it
teacher-forced through native Qwen, and check the tie margin, the latency
ratios, the sample spread and peak memory. Native is measured in the same
process on the same shapes, which is how the latency gates are defined.

Not submitted. ``bench/modal_bench.py`` runs this on Modal; it also runs as-is
on any machine with the pinned runtime and a copy of the checkpoint.
"""

import statistics
import sys
import time
import zlib

import torch

TIE_MARGIN = 2.0
LATENCY_LIMIT = 1.10
SPREAD_LIMIT = 0.25
MEMORY_LIMIT = 0.90

PUBLIC_SHAPES = [
    ("public-0", 1, 512, 32),
    ("public-1", 4, 2048, 32),
    ("public-2", 16, 512, 128),
]


CORPUS = None


def _prompts(batch, length, vocab, seed):
    """Prompt ids the engine has never seen, fresh per sample like the judge's.

    Random ids carry no n-gram structure, which is fine for timing a plain
    decode loop and useless for judging speculation. When a corpus is loaded,
    draw real slices of it instead -- the judge derives its prompts from a
    fixed corpus, so acceptance and its variance only mean anything here.
    """
    generator = torch.Generator().manual_seed(seed)
    if CORPUS is None:
        return torch.randint(0, vocab, (batch, length), generator=generator).tolist()
    starts = torch.randint(
        0, len(CORPUS) - length - 1, (batch,), generator=generator
    ).tolist()
    return [CORPUS[at : at + length] for at in starts]


def load_corpus(path, model_path):
    """Tokenise a text file once, for realistic prompts."""
    global CORPUS
    from transformers import AutoTokenizer

    text = open(path, encoding="utf-8", errors="ignore").read()
    if len(text) < 10000:
        print("corpus unavailable; keeping random prompts", flush=True)
        return
    CORPUS = AutoTokenizer.from_pretrained(model_path)(text)["input_ids"]
    print(f"corpus: {len(CORPUS)} tokens", flush=True)


def _time_stream(generate, prompts, steps):
    """One sample: time to first token, total seconds, and what came back."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    stream = generate(prompts, steps)
    emitted = []
    first = None
    for tokens in stream:
        if first is None:
            first = time.perf_counter() - start
        emitted.append(tokens)
    total = time.perf_counter() - start
    return first, total, emitted


def _measure(generate, batch, length, steps, vocab, samples, seed0):
    """Warm up once, then time the samples, as the judge does per workload."""
    _time_stream(generate, _prompts(batch, length, vocab, seed0), steps)

    records = []
    for index in range(samples):
        prompts = _prompts(batch, length, vocab, seed0 + 1 + index)
        first, total, emitted = _time_stream(generate, prompts, steps)
        records.append((first, total, prompts, emitted))

    totals = [r[1] for r in records]
    firsts = [r[0] for r in records]
    median = statistics.median(totals)
    return {
        "ttft": statistics.median(firsts),
        "tpot": statistics.median(
            (t - f) / max(1, steps - 1) for f, t, _, _ in records
        ),
        "median": median,
        "spread": (max(totals) - min(totals)) / median if median else float("inf"),
        "tps": batch * steps / median,
        "records": records,
    }


def _replay(native, device, prompts, emitted, steps):
    """The judge's rule: our tokens, teacher-forced through native Qwen."""
    ids = torch.tensor(prompts, dtype=torch.int64, device=device)
    tokens = torch.tensor(emitted, dtype=torch.int64, device=device).T
    full = torch.cat([ids, tokens], dim=1)

    with torch.inference_mode():
        logits = native(
            input_ids=full, logits_to_keep=steps + 1, return_dict=True
        ).logits[:, :steps, :].float()

    best = logits.max(dim=-1).values
    chose = logits.gather(2, tokens[:, :, None])[:, :, 0]
    gap = (best - chose).max().item()
    exact = (logits.argmax(dim=-1) == tokens).all().item()
    return gap, exact


def run(model_path, shapes=None, samples=5, verbose=True, engine_path="/root/engine"):
    """Benchmark and check the engine on each shape; returns a row per shape."""
    if engine_path and engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    from engine import Engine

    shapes = shapes or PUBLIC_SHAPES
    torch.cuda.reset_peak_memory_stats()

    build = time.perf_counter()
    engine = Engine(model_path)
    if verbose:
        print(f"load + self-check: {time.perf_counter() - build:.1f}s", flush=True)
        print(f"fast path live: {engine._fast}", flush=True)

    device = engine.device
    vocab = engine.embed.shape[0] if engine._fast else engine._native.config.vocab_size
    total_memory = torch.cuda.get_device_properties(0).total_memory
    rows = []

    for name, batch, length, steps in shapes:
        torch.cuda.reset_peak_memory_stats()
        mine = _measure(
            engine.generate, batch, length, steps, vocab, samples, seed0=zlib.crc32(name.encode()) % 10**6
        )
        peak = torch.cuda.max_memory_allocated() / total_memory

        native = _measure(
            engine._native_generate, batch, length, steps, vocab, samples,
            seed0=zlib.crc32(name.encode()) % 10**6,
        )

        gap, exact = _replay(
            engine._native, device, mine["records"][0][2], mine["records"][0][3], steps
        )

        row = {
            "workload": name,
            "batch": batch, "prompt": length, "output": steps,
            "tps": mine["tps"], "native_tps": native["tps"],
            "speedup": mine["tps"] / native["tps"],
            "ttft_ratio": mine["ttft"] / native["ttft"],
            "tpot_ratio": mine["tpot"] / native["tpot"],
            "spread": mine["spread"], "peak_memory": peak,
            "tie_gap": gap, "argmax_exact": exact,
        }
        row["passes"] = (
            gap <= TIE_MARGIN
            and row["ttft_ratio"] <= LATENCY_LIMIT
            and row["tpot_ratio"] <= LATENCY_LIMIT
            and row["spread"] <= SPREAD_LIMIT
            and peak <= MEMORY_LIMIT
        )
        rows.append(row)
        if verbose:
            _report_row(row)

    if verbose:
        _report_summary(rows)
    return rows


def _report_row(row):
    verdict = "pass" if row["passes"] else "FAIL"
    print(
        f"\n{row['workload']}  b{row['batch']} x {row['prompt']}->{row['output']}  [{verdict}]\n"
        f"  tok/s      {row['tps']:10.1f}   native {row['native_tps']:8.1f}"
        f"   speedup {row['speedup']:.2f}x\n"
        f"  ttft ratio {row['ttft_ratio']:10.3f}   tpot ratio {row['tpot_ratio']:.3f}"
        f"   (gate {LATENCY_LIMIT})\n"
        f"  spread     {row['spread']:10.3f}   (gate {SPREAD_LIMIT})\n"
        f"  peak mem   {row['peak_memory']:10.3f}   (gate {MEMORY_LIMIT})\n"
        f"  tie gap    {row['tie_gap']:10.4f}   (margin {TIE_MARGIN})"
        f"   argmax exact {row['argmax_exact']}",
        flush=True,
    )


def _report_summary(rows):
    product = 1.0
    for row in rows:
        product *= row["tps"]
    geomean = product ** (1.0 / len(rows))
    print(
        f"\ngeometric mean over {len(rows)} shapes: {geomean:.1f} tok/s"
        f"   ({'all pass' if all(r['passes'] for r in rows) else 'GATES FAILED'})",
        flush=True,
    )
