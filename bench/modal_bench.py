"""Run the engine on a real H100 through Modal.

The container pins exactly what the benchmark pins -- Python 3.11, CUDA 12.4,
torch 2.5.1, Triton 3.1.0, Transformers 4.51.3 -- so Triton compiles the same
kernels and CUDA graphs capture the same way. The checkpoint lives in a volume,
downloaded once.

    modal run bench/modal_bench.py::fetch      # once, ~8 GiB
    modal run bench/modal_bench.py             # benchmark the public shapes
    modal run bench/modal_bench.py::check      # correctness only, one sample
    modal run bench/modal_bench.py::shell      # interactive box for poking

Local edits to engine/ are picked up on every run; nothing is baked in.
"""

import os

import modal

#: H100 is what the benchmark runs on and what the numbers must come from.
#: Set DRYFT_GPU=A10G (or L4, A100) to validate Triton and graph capture on a
#: cheaper card -- correctness carries over, timings do not.
GPU = os.environ.get("DRYFT_GPU", "H100")

#: DRYFT_GPU=cpu registers every function without a GPU, so ``fetch`` can run
#: on an account that has no GPU entitlement yet.
_GPU = {} if GPU.lower() in ("", "cpu", "none") else {"gpu": GPU}

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
WEIGHTS = "/weights/qwen3-4b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.51.3",
        "safetensors==0.5.3",
        "tokenizers==0.21.1",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Natural English, for estimating n-gram acceptance. The judge draws its
    # prompts from a fixed corpus; random token ids have no structure at all
    # and would say speculation is worthless when it is not.
    .run_commands(
        "python -c \"import urllib.request as u; "
        "open('/root/corpus.txt','wb').write("
        "u.urlopen('https://www.gutenberg.org/cache/epub/1342/pg1342.txt',"
        "timeout=60).read())\" || echo unavailable > /root/corpus.txt"
    )
    .add_local_dir("engine", "/root/engine")
    .add_local_file("bench/harness.py", "/root/harness.py")
    .add_local_file("bench/probe_kernels.py", "/root/probe_kernels.py")
)

weights = modal.Volume.from_name("dryft-qwen3-4b", create_if_missing=True)
app = modal.App("dryft-engine")


@app.function(image=image, volumes={"/weights": weights}, timeout=3600)
def fetch():
    """Pull the pinned revision into the volume. Run once."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=WEIGHTS,
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    weights.commit()
    print(f"checkpoint in {WEIGHTS}")


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def benchmark(shapes=None, samples: int = 5, detune: str = "",
              draft: int = 0, target: float = 1.25, corpus: bool = False):
    """Time and check the engine the way the judge would.

    ``detune`` switches off stages by name (``graph``, ``triton``, ``matmul``)
    to measure what an earlier build of this engine would score, and with how
    much room under the latency gates.
    """
    import os
    import sys

    if draft:
        os.environ["DRYFT_DRAFT"] = str(draft)
        os.environ["DRYFT_DRAFT_TARGET"] = str(target)
    for stage in filter(None, detune.split(",")):
        os.environ[{"graph": "DRYFT_GRAPH", "triton": "DRYFT_ATTENTION",
                    "matmul": "DRYFT_MATMUL"}[stage]] = "off"
    if detune:
        print(f"detuned: {detune}", flush=True)
    sys.path.insert(0, "/root")
    import harness
    from harness import run

    _describe_gpu(require_h100=True)
    if corpus:
        harness.load_corpus("/root/corpus.txt", WEIGHTS)
    if draft:
        print(f"speculation: draft={draft} target={target}", flush=True)
    return run(WEIGHTS, shapes=shapes, samples=samples)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def check():
    """One sample per shape: does it load, agree with native, and stay in budget."""
    import sys

    sys.path.insert(0, "/root")
    from harness import run

    _describe_gpu()
    return run(WEIGHTS, samples=1)


def _describe_gpu(require_h100: bool = False):
    import torch

    prop = torch.cuda.get_device_properties(0)
    print(
        f"{prop.name}  {prop.total_memory / 2**30:.0f} GiB  "
        f"{prop.multi_processor_count} SMs  torch {torch.__version__}",
        flush=True,
    )
    # Modal substitutes an H200 when no H100 is free. It has 4.8 TB/s against
    # the H100's 3.35, so any timing taken on one is not comparable to the
    # benchmark's hardware. Token statistics are unaffected.
    if "H100" not in prop.name:
        print(f"WARNING: {prop.name} is not the benchmark's H100 -- "
              f"timings from this run are not comparable", flush=True)
        if require_h100:
            raise RuntimeError(f"needed an H100, got {prop.name}; retry")


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def profile(shape: str = "public-0"):
    """Per-stage timing for one shape: where the milliseconds actually go."""
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from harness import PUBLIC_SHAPES, _prompts

    _describe_gpu()
    name, batch, length, steps = next(s for s in PUBLIC_SHAPES if s[0] == shape)
    engine = Engine(WEIGHTS)
    vocab = engine.embed.shape[0]

    list(engine.generate(_prompts(batch, length, vocab, 0), steps))  # warm

    prompts = _prompts(batch, length, vocab, 1)
    ids = engine._upload(prompts, batch, length)
    torch.cuda.synchronize()

    start = time.perf_counter()
    engine._prefill(ids)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(steps - 1):
        engine.graph.replay()
    torch.cuda.synchronize()
    decode = time.perf_counter() - start

    per_step = decode / max(1, steps - 1)
    print(
        f"\n{name}  b{batch} x {length}->{steps}\n"
        f"  prefill     {prefill * 1e3:8.2f} ms\n"
        f"  decode      {decode * 1e3:8.2f} ms   ({per_step * 1e3:.3f} ms/step)\n"
        f"  prefill is  {prefill / (prefill + decode) * 100:6.1f}% of the clock\n"
        f"  implied     {batch * steps / (prefill + decode):8.1f} tok/s\n"
        f"  8.045 GB/step at {8.045e9 / per_step / 1e12:.2f} TB/s effective",
        flush=True,
    )


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def shell():
    """Keep a box alive to poke at: modal run ... ::shell"""
    import subprocess

    subprocess.run(["/bin/bash", "-l"])


@app.local_entrypoint()
def main(samples: int = 5, detune: str = "", draft: int = 0,
         target: float = 1.25, corpus: bool = False):
    benchmark.remote(samples=samples, detune=detune, draft=draft,
                     target=target, corpus=corpus)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def micro(batch: int = 1, context: int = 512):
    """Achieved bandwidth per stage of a decode step, against the real weights."""
    import sys
    import time

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from harness import _prompts

    _describe_gpu()
    engine = Engine(WEIGHTS)
    list(engine.generate(_prompts(batch, context, engine.embed.shape[0], 0), 4))

    def timed(fn, bytes_moved, label, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        each = (time.perf_counter() - start) / reps
        print(
            f"  {label:26s} {each * 1e3:8.3f} ms   {bytes_moved / each / 1e12:5.2f} TB/s",
            flush=True,
        )
        return each

    layers = engine.layers
    hidden = torch.randn(batch, engine.hidden, dtype=torch.bfloat16, device="cuda")
    attn_in = torch.randn(batch, engine.q_width, dtype=torch.bfloat16, device="cuda")
    mlp_in = torch.randn(batch, layers[0].gate_up.shape[0] // 2, dtype=torch.bfloat16,
                         device="cuda")

    print(f"\nbatch {batch}, context {context}")
    total = 0.0
    for label, weight_of, source in [
        ("qkv proj  x36", lambda ly: ly.qkv, hidden),
        ("o proj    x36", lambda ly: ly.o, attn_in),
        ("gate_up   x36", lambda ly: ly.gate_up, hidden),
        ("down proj x36", lambda ly: ly.down, mlp_in),
    ]:
        moved = sum(weight_of(ly).numel() * 2 for ly in layers)
        total += timed(lambda w=weight_of: [F.linear(source, w(ly)) for ly in layers],
                       moved, label)

    kv = engine.k_cache[0]
    group = engine.n_q // engine.n_kv
    q4 = torch.randn(batch, engine.n_kv, group, engine.head_dim,
                     dtype=torch.bfloat16, device="cuda")
    kv_bytes = 2 * 36 * kv[:batch].numel() * 2
    total += timed(
        lambda: [
            F.scaled_dot_product_attention(
                q4, engine.k_cache[i][:batch], engine.v_cache[i][:batch],
                attn_mask=engine.mask.view(1, 1, 1, -1),
            )
            for i in range(36)
        ],
        kv_bytes, "attention x36",
    )
    total += timed(lambda: F.linear(hidden, engine.embed),
                   engine.embed.numel() * 2, "lm_head")

    step = timed(lambda: engine.graph.replay(), 8.045e9, "whole graphed step")
    print(f"  {'stages accounted':26s} {total * 1e3:8.3f} ms of {step * 1e3:.3f} ms"
          f"   ({(step - total) * 1e3:.3f} ms unexplained)", flush=True)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def attn_probe(batch: int = 1, context: int = 552):
    """Which SDPA formulation and backend is costing us the decode step."""
    import sys
    import time

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sys.path.insert(0, "/root")
    _describe_gpu()

    n_kv, group, dim = 8, 4, 128
    kv_bytes = 2 * batch * n_kv * context * dim * 2
    k = torch.randn(batch, n_kv, context, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    q4 = torch.randn(batch, n_kv, group, dim, dtype=torch.bfloat16, device="cuda")
    q32 = torch.randn(batch, n_kv * group, 1, dim, dtype=torch.bfloat16, device="cuda")
    bool_mask = torch.ones(context, dtype=torch.bool, device="cuda").view(1, 1, 1, -1)

    def timed(fn, label, reps=200):
        try:
            for _ in range(5):
                fn()
        except Exception as exc:
            print(f"  {label:38s} unsupported: {type(exc).__name__}", flush=True)
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        each = (time.perf_counter() - start) / reps
        print(f"  {label:38s} {each * 1e6:8.1f} us   {kv_bytes / each / 1e12:5.2f} TB/s",
              flush=True)

    print(f"\nbatch {batch}, context {context}, kv {kv_bytes / 2**20:.1f} MiB")
    timed(lambda: F.scaled_dot_product_attention(q4, k, v, attn_mask=bool_mask),
          "q_len=4 + bool mask (current)")
    timed(lambda: F.scaled_dot_product_attention(q4, k, v),
          "q_len=4, no mask")
    timed(lambda: F.scaled_dot_product_attention(q32, k, v, enable_gqa=True),
          "q_len=1, 32 heads, enable_gqa")
    for backend, name in [
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "mem_efficient"),
        (SDPBackend.MATH, "math"),
    ]:
        timed(
            lambda b=backend: _forced(b, q4, k, v, bool_mask),
            f"q_len=4 + bool mask, forced {name}",
        )
        timed(lambda b=backend: _forced(b, q4, k, v, None),
              f"q_len=4, no mask, forced {name}")


def _forced(backend, q, k, v, mask):
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def trace(batch: int = 1, context: int = 512):
    """Per-kernel time inside one graphed decode step, and the gap around it."""
    import sys
    import time
    from collections import defaultdict

    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from harness import _prompts

    _describe_gpu()
    engine = Engine(WEIGHTS)
    list(engine.generate(_prompts(batch, context, engine.embed.shape[0], 0), 8))

    reps = 50
    for _ in range(5):
        engine.graph.replay()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        engine.graph.replay()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / reps

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(reps):
            engine.graph.replay()
        torch.cuda.synchronize()

    totals = defaultdict(float)
    counts = defaultdict(int)
    for event in prof.key_averages():
        if event.self_device_time_total > 0:
            totals[event.key] += event.self_device_time_total
            counts[event.key] += event.count

    grand = sum(totals.values()) / reps
    print(f"\nbatch {batch}  step wall {wall * 1e3:.3f} ms   "
          f"kernel time {grand:.1f} us   gap {wall * 1e6 - grand:.1f} us", flush=True)
    print(f"  {'kernel':52s} {'calls':>6s} {'us/step':>9s} {'us/call':>8s}", flush=True)
    for key in sorted(totals, key=totals.get, reverse=True)[:18]:
        per_step = totals[key] / reps
        per_call = totals[key] / max(1, counts[key])
        print(f"  {key[:52]:52s} {counts[key] // reps:6d} {per_step:9.1f} {per_call:8.2f}",
              flush=True)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def layout():
    """Does cuBLAS prefer the weight transposed? Median of repeated trials."""
    import statistics
    import sys

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    _describe_gpu()
    from kernels.gemm import _CONFIGS, skinny_matmul

    shapes = [
        ("qkv", 6144, 2560), ("o", 2560, 4096),
        ("gate_up", 19456, 2560), ("down", 2560, 9728),
        ("lm_head", 151936, 2560),
    ]

    def clock(fn, reps=100, trials=5):
        for _ in range(10):
            fn()
        runs = []
        for _ in range(trials):
            torch.cuda.synchronize()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(reps):
                fn()
            b.record()
            torch.cuda.synchronize()
            runs.append(a.elapsed_time(b) / reps)
        return statistics.median(runs)

    for batch in (1, 4, 16):
        print(f"\nbatch {batch}", flush=True)
        for name, n, k in shapes:
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            wt = w.t().contiguous()
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            moved = w.numel() * 2

            rates = {
                "linear [n,k]": clock(lambda: F.linear(x, w)),
                "mm     [k,n]": clock(lambda: torch.mm(x, wt)),
            }
            best_cfg, best_ms = None, float("inf")
            for cfg in _CONFIGS:
                try:
                    skinny_matmul(x, w, config=cfg)
                except Exception:
                    continue
                ms = clock(lambda c=cfg: skinny_matmul(x, w, config=c), reps=50, trials=3)
                if ms < best_ms:
                    best_cfg, best_ms = cfg, ms
            rates[f"triton {best_cfg}"] = best_ms

            line = "  ".join(
                f"{label} {moved / (ms * 1e-3) / 1e12:5.2f}" for label, ms in rates.items()
            )
            winner = min(rates, key=rates.get)
            print(f"  {name:8s} [{n},{k}]  {line}   -> {winner}", flush=True)
            del w, wt
            torch.cuda.empty_cache()


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def trace_prefill(batch: int = 16, context: int = 512):
    """Where prefill's milliseconds go. TTFT ratio says this half is neglected."""
    import sys
    import time
    from collections import defaultdict

    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from harness import _prompts

    _describe_gpu()
    engine = Engine(WEIGHTS)
    vocab = engine.embed.shape[0]
    list(engine.generate(_prompts(batch, context, vocab, 0), 4))
    ids = engine._upload(_prompts(batch, context, vocab, 1), batch, context)

    for _ in range(3):
        engine._prefill(ids)
    torch.cuda.synchronize()
    reps = 10
    start = time.perf_counter()
    for _ in range(reps):
        engine._prefill(ids)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / reps

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            engine._prefill(ids)
        torch.cuda.synchronize()

    totals, counts = defaultdict(float), defaultdict(int)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            totals[e.key] += e.self_device_time_total
            counts[e.key] += e.count
    grand = sum(totals.values()) / reps

    tokens = batch * context
    flops = 2 * 3.633e9 * tokens
    print(f"\nprefill b{batch} x {context} = {tokens} tokens")
    print(f"  wall {wall * 1e3:.2f} ms   kernels {grand / 1e3:.2f} ms"
          f"   gap {wall * 1e3 - grand / 1e3:.2f} ms")
    print(f"  dense GEMM work {flops / 1e12:.1f} TFLOP -> {flops / wall / 1e12:.0f} TFLOPS"
          f"  ({flops / wall / 989e12 * 100:.0f}% of peak)")
    print(f"  {'kernel':50s} {'calls':>6s} {'us':>9s} {'share':>6s}")
    for key in sorted(totals, key=totals.get, reverse=True)[:12]:
        per = totals[key] / reps
        print(f"  {key[:50]:50s} {counts[key] // reps:6d} {per:9.1f} {per / grand * 100:5.1f}%")


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def accept_study(prompt_len: int = 512, steps: int = 192, prompts: int = 6):
    """How many tokens would an n-gram draft get right, on real text?

    No kernels: generate greedily, then replay the sequence asking what a
    prompt-lookup draft would have proposed at each step. Mean accepted length
    per step is the whole speculation payoff, so measure it before building it.
    """
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from transformers import AutoTokenizer

    _describe_gpu()
    text = open("/root/corpus.txt", encoding="utf-8", errors="ignore").read()
    if len(text) < 10000:
        print("corpus unavailable; aborting")
        return
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    print(f"corpus: {len(text)} chars -> {len(ids)} tokens", flush=True)

    engine = Engine(WEIGHTS)

    sequences = []
    for index in range(prompts):
        start = 2000 + index * 4000
        prompt = ids[start : start + prompt_len]
        out = [row[0] for row in engine.generate([prompt], steps)]
        sequences.append((prompt, out))
        print(f"  prompt {index}: generated {len(out)} tokens", flush=True)

    def simulate(prompt, generated, order, draft, use_generated=True):
        """Mean tokens emitted per forward pass under an n-gram draft."""
        context = list(prompt)
        pos, passes, produced = 0, 0, 0
        while pos < len(generated):
            key = tuple(context[-order:])
            proposal = []
            haystack = context if use_generated else list(prompt)
            for j in range(len(haystack) - order - 1, -1, -1):
                if tuple(haystack[j : j + order]) == key:
                    proposal = haystack[j + order : j + order + draft]
                    break
            taken = 0
            for a, b in zip(proposal, generated[pos:]):
                if a != b:
                    break
                taken += 1
            emitted = taken + 1  # the model's own token always lands
            emitted = min(emitted, len(generated) - pos)
            context.extend(generated[pos : pos + emitted])
            pos += emitted
            produced += emitted
            passes += 1
        return produced / passes

    print("\nmean tokens per forward pass (1.00 = no speculation)")
    print(f"  {'order':>5} {'draft':>5} {'alpha':>7} {'vs now':>8}")
    for order in (2, 3, 4):
        for draft in (2, 4, 8):
            alphas = [simulate(p, g, order, draft) for p, g in sequences]
            mean = sum(alphas) / len(alphas)
            lo, hi = min(alphas), max(alphas)
            print(f"  {order:5d} {draft:5d} {mean:7.3f} {mean:7.2f}x"
                  f"   per-prompt {lo:.2f}-{hi:.2f}", flush=True)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def quant_probe(prompt_len: int = 512, prompts: int = 4):
    """Would FP8 weights survive the judge's tie-margin replay?

    No kernels and no speedup: quantise the projections, dequantise straight
    back to bfloat16, and run the same forward. That isolates the accuracy cost
    of the format from every implementation question. The number that matters
    is the worst tie gap -- how far below the true argmax our token would sit.
    """
    import sys

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from transformers import AutoTokenizer

    _describe_gpu()
    text = open("/root/corpus.txt", encoding="utf-8", errors="ignore").read()
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    corpus = tokenizer(text)["input_ids"]

    engine = Engine(WEIGHTS)
    engine._ensure(1, prompt_len, 8)

    def logits_for(ids):
        with torch.no_grad():
            hidden = F.embedding(ids, engine.embed)
            normed = engine._blocks(
                hidden.view(prompt_len, engine.hidden), engine.arange[:prompt_len],
                prompt_len, 1, 0, engine._attend_prefill,
            )
            return F.linear(normed, engine.embed).float()

    batches = []
    for index in range(prompts):
        at = 5000 + index * 9000
        ids = torch.tensor(
            [corpus[at : at + prompt_len]], dtype=torch.int64, device="cuda"
        )
        batches.append((ids, logits_for(ids)))
    print(f"reference logits: {prompts} x {prompt_len} positions", flush=True)

    def quantise(w, dtype, group=0):
        """Round-trip through the format. group=0 is per output channel."""
        limit = 127.0 if dtype == torch.int8 else torch.finfo(dtype).max
        out, inner = w.shape
        if group and inner % group == 0:
            tile = w.view(out, inner // group, group)
            scale = (tile.abs().amax(dim=2, keepdim=True) / limit).clamp(min=1e-12)
            back = (tile / scale).clamp(-limit, limit).to(dtype).to(w.dtype) * scale
            return back.view(out, inner).to(w.dtype)
        scale = (w.abs().amax(dim=1, keepdim=True) / limit).clamp(min=1e-12)
        return ((w / scale).clamp(-limit, limit).to(dtype).to(w.dtype) * scale).to(w.dtype)

    originals = [
        (layer, name, getattr(layer, name).clone())
        for layer in engine.layers
        for name in ("qkv", "o", "gate_up", "down")
    ]

    F8 = torch.float8_e4m3fn
    plans = [
        ("fp8 per-channel   all",      F8, 0,   ("qkv", "o", "gate_up", "down")),
        ("fp8 group-128     all",      F8, 128, ("qkv", "o", "gate_up", "down")),
        ("fp8 group-64      all",      F8, 64,  ("qkv", "o", "gate_up", "down")),
        ("fp8 group-32      all",      F8, 32,  ("qkv", "o", "gate_up", "down")),
        ("fp8 group-128 mlp only",     F8, 128, ("gate_up", "down")),
        ("fp8 group-64  mlp only",     F8, 64,  ("gate_up", "down")),
        ("int8 group-128    all", torch.int8, 128, ("qkv", "o", "gate_up", "down")),
    ]
    saved = {"qkv": 1.134, "o": 0.755, "gate_up": 3.586, "down": 1.793}
    for label, dtype, group, targets in plans:
        for layer, name, original in originals:
            setattr(layer, name,
                    quantise(original, dtype, group) if name in targets else original)
        head = engine.embed
        shrink = sum(saved[n] for n in targets) / 2

        worst = 0.0
        flips = total = 0
        for ids, reference in batches:
            mine = logits_for(ids)
            chosen = mine.argmax(dim=-1, keepdim=True)
            gap = reference.max(dim=-1, keepdim=True).values - reference.gather(1, chosen)
            worst = max(worst, gap.max().item())
            flips += (chosen[:, 0] != reference.argmax(dim=-1)).sum().item()
            total += chosen.shape[0]
        engine.embed = head
        speed = 8.045 / (8.045 - shrink)
        print(f"  {label:24s} gap {worst:6.3f}  flips {flips:5d}/{total}"
              f"  bytes -{shrink:.2f} GB  decode x{speed:.2f}", flush=True)

    for layer, name, original in originals:
        setattr(layer, name, original)
    print("  (weights restored)", flush=True)


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def utilisation(batch: int = 1, context: int = 512, seconds: int = 8):
    """Is the GPU saturated, starved, or throttled while we decode?

    Utilisation percentages alone are misleading -- "100% GPU" only means a
    kernel was resident, not that it was moving bytes. So sample the memory
    controller and the clocks too, and put the achieved bandwidth next to them.
    """
    import subprocess
    import sys
    import threading
    import time

    import torch

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    from harness import _prompts

    _describe_gpu()
    engine = Engine(WEIGHTS)
    list(engine.generate(_prompts(batch, context, engine.embed.shape[0], 0), 8))

    fields = ("utilization.gpu,utilization.memory,clocks.sm,clocks.mem,"
              "power.draw,temperature.gpu,clocks_throttle_reasons.active")
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out:
                samples.append(out.split(", "))
            time.sleep(0.05)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()

    torch.cuda.synchronize()
    start = time.perf_counter()
    replays = 0
    while time.perf_counter() - start < seconds:
        for _ in range(200):
            engine.graph.replay()
        replays += 200
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stop.set()
    watcher.join(timeout=2)

    step = elapsed / replays
    moved = 8.045e9 + batch * (context + 8) * 147456
    print(f"\n{replays} decode steps in {elapsed:.2f}s -> {step * 1e6:.0f} us/step")
    print(f"  achieved {moved / step / 1e12:.2f} TB/s of 3.35 peak "
          f"({moved / step / 3.35e12 * 100:.0f}%)")

    if samples:
        def column(i, cast=float):
            values = []
            for row in samples:
                try:
                    values.append(cast(row[i]))
                except (ValueError, IndexError):
                    pass
            return values

        for index, (name, unit) in enumerate([
            ("sm utilisation", "%"), ("memory controller", "%"),
            ("sm clock", "MHz"), ("memory clock", "MHz"),
            ("power", "W"), ("temperature", "C"),
        ]):
            values = column(index)
            if values:
                print(f"  {name:20s} mean {sum(values) / len(values):7.1f} "
                      f"max {max(values):7.1f} {unit}")
        throttle = {row[6] for row in samples if len(row) > 6}
        print(f"  throttle reasons     {throttle}")
        print(f"  ({len(samples)} samples)")


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def fp8_probe():
    """Correctness and speed of the FP8 GEMV against cuBLAS, per shape."""
    import statistics
    import sys

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    _describe_gpu()
    from kernels import fp8

    def clock(fn, operands, reps=3, trials=5):
        for w in operands[:4]:
            fn(w)
        runs = []
        for _ in range(trials):
            torch.cuda.synchronize()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(reps):
                for w in operands:
                    fn(w)
            b.record()
            torch.cuda.synchronize()
            runs.append(a.elapsed_time(b) / (reps * len(operands)))
        return statistics.median(runs)

    for name, n, k in [("qkv", 6144, 2560), ("o", 2560, 4096),
                       ("gate_up", 19456, 2560), ("down", 2560, 9728)]:
        every = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
                 for _ in range(8)]
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        try:
            packed = [fp8.quantize(w) for w in every]
        except Exception as exc:
            print(f"  {name:8s} quantize failed: {type(exc).__name__}: {exc}")
            continue

        reference = F.linear(x, every[0])
        try:
            mine = fp8.fp8_matmul(x, packed[0])
        except Exception as exc:
            print(f"  {name:8s} kernel failed: {type(exc).__name__}: {exc}"[:200])
            continue
        scale = reference.float().abs().max().item()
        err = (mine.float() - reference.float()).abs().max().item()

        bf16_ms = clock(lambda w: F.linear(x, w), every)
        fp8_ms, best_cfg = float("inf"), None
        for cfg in fp8.CONFIGS:
            try:
                fp8.fp8_matmul(x, packed[0], config=cfg)
            except Exception:
                continue
            ms = clock(lambda p, c=cfg: fp8.fp8_matmul(x, p, config=c), packed)
            if ms < fp8_ms:
                fp8_ms, best_cfg = ms, cfg
        bf16_bytes = every[0].numel() * 2
        fp8_bytes = fp8.bytes_moved(packed[0])
        print(f"  {name:8s} rel_err {err / scale:8.5f}   "
              f"bf16 {bf16_ms * 1e3:6.1f}us {bf16_bytes / (bf16_ms * 1e-3) / 1e12:.2f}TB/s   "
              f"fp8 {fp8_ms * 1e3:6.1f}us {fp8_bytes / (fp8_ms * 1e-3) / 1e12:.2f}TB/s   "
              f"speedup {bf16_ms / fp8_ms:.2f}x  {best_cfg}", flush=True)
        del every, packed
        torch.cuda.empty_cache()


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def fp8_isolate():
    """Is Triton's FP8 load slow, or is it my scale gather?"""
    import statistics
    import sys

    import torch
    import triton
    import triton.language as tl

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    _describe_gpu()

    from probe_kernels import drain as _drain

    def clock(fn, reps=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        runs = []
        for _ in range(5):
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(reps):
                fn()
            b.record()
            torch.cuda.synchronize()
            runs.append(a.elapsed_time(b) / reps)
        return statistics.median(runs)

    n, k = 19456, 2560
    src = (torch.randn(n, k, device="cuda") * 0.02)
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    print(f"  pure streaming read of a [{n},{k}] weight, no scales, no activation")
    for label, w in [("bf16", src.to(torch.bfloat16)),
                     ("fp8 e4m3", src.to(torch.float8_e4m3fn)),
                     ("int8", (src * 100).to(torch.int8))]:
        moved = w.numel() * w.element_size()
        for block_n, block_k, warps in [(32, 128, 4), (64, 128, 8), (128, 64, 8),
                                        (64, 256, 8), (32, 512, 8)]:
            try:
                ms = clock(lambda: _drain[(triton.cdiv(n, block_n),)](
                    w, out, k, N=n, BLOCK_N=block_n, BLOCK_K=block_k,
                    num_warps=warps, num_stages=4))
            except Exception as exc:
                print(f"    {label:9s} {block_n:4d}x{block_k:4d} w{warps}  "
                      f"failed: {type(exc).__name__}: {exc}"[:400]); continue
            print(f"    {label:9s} {block_n:4d}x{block_k:4d} w{warps}  "
                  f"{ms * 1e3:7.1f}us  {moved / (ms * 1e-3) / 1e12:5.2f} TB/s", flush=True)
        del w
        torch.cuda.empty_cache()


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def compiler_probe():
    """Can we compile CUDA at load time, and how? This gates the whole plan.

    Three routes, cheapest-to-verify first. nvcc via cpp_extension is easiest
    but needs a toolkit the runtime image may not carry. NVRTC ships inside the
    torch wheel itself, so it is present wherever torch is -- that is the route
    that survives an unknown container.
    """
    import ctypes
    import os
    import shutil
    import subprocess
    import sys
    import time

    import torch

    _describe_gpu()
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")

    print("\n1. nvcc on PATH?")
    nvcc = shutil.which("nvcc") or (
        f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin/nvcc"
    )
    if os.path.exists(nvcc):
        print(f"   found {nvcc}")
        print("  ", subprocess.run([nvcc, "--version"], capture_output=True,
                                   text=True).stdout.strip().splitlines()[-1])
    else:
        print("   NOT FOUND")

    print("\n2. a PTX toolchain anywhere in site-packages?")
    roots = {os.path.dirname(os.path.dirname(torch.__file__))}
    try:
        import triton
        roots.add(os.path.dirname(os.path.dirname(triton.__file__)))
    except Exception:
        pass
    hits = {}
    for base in roots:
        for root, _, files in os.walk(base):
            for f in files:
                for want in ("libnvrtc.so", "ptxas", "libcuda.so", "nvdisasm"):
                    if f.startswith(want) and want not in hits:
                        hits[want] = os.path.join(root, f)
    for want in ("libnvrtc.so", "ptxas", "libcuda.so", "nvdisasm"):
        print(f"   {want:14s} {hits.get(want, 'NOT FOUND')}")
    found = hits.get("libnvrtc.so")

    if found:
        try:
            nvrtc = ctypes.CDLL(found)
            major, minor = ctypes.c_int(), ctypes.c_int()
            nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
            print(f"   nvrtc {major.value}.{minor.value} loads via ctypes")

            src = b"extern \"C\" __global__ void touch(float* x){ x[0]=1.0f; }\n"
            prog = ctypes.c_void_p()
            rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"t.cu", 0, None, None)
            opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_90")
            rc2 = nvrtc.nvrtcCompileProgram(prog, 1, opts)
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
            print(f"   compile rc={rc},{rc2}  ptx bytes={size.value}  -> "
                  f"{'WORKS' if rc2 == 0 and size.value > 0 else 'FAILED'}")
        except Exception as exc:
            print(f"   ctypes route failed: {type(exc).__name__}: {exc}")

    print("\n3. cooperative launch available on the driver?")
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        attr = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
        libcuda.cuInit(0)
        dev = ctypes.c_int()
        libcuda.cuDeviceGet(ctypes.byref(dev), 0)
        libcuda.cuDeviceGetAttribute(ctypes.byref(attr), 95, dev)
        print(f"   cooperative launch supported: {bool(attr.value)}")
        print(f"   cuLaunchCooperativeKernel present: "
              f"{hasattr(libcuda, 'cuLaunchCooperativeKernel')}")
    except Exception as exc:
        print(f"   {type(exc).__name__}: {exc}")

    print("\n4. cpp_extension.load_inline end to end (times the compile)")
    try:
        from torch.utils.cpp_extension import load_inline

        start = time.perf_counter()
        mod = load_inline(
            name="probe_mk",
            cpp_sources="torch::Tensor go(torch::Tensor x);",
            cuda_sources=(
                "#include <torch/extension.h>\n"
                "__global__ void k(float* x){ x[threadIdx.x] += 1.0f; }\n"
                "torch::Tensor go(torch::Tensor x){ k<<<1,32>>>(x.data_ptr<float>());"
                " return x; }\n"
            ),
            functions=["go"], verbose=False,
        )
        took = time.perf_counter() - start
        x = torch.zeros(32, device="cuda")
        mod.go(x)
        torch.cuda.synchronize()
        print(f"   WORKS in {took:.1f}s, result {x[0].item()}")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {str(exc)[:300]}")


@app.function(
    image=image, **_GPU, volumes={"/weights": weights}, timeout=3600
)
def cuda_probe():
    """Does the JIT route work, is the GEMV correct, and does it beat cuBLAS?"""
    import statistics
    import sys
    import time

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/engine")
    _describe_gpu()
    from kernels import cuda_gemv

    start = time.perf_counter()
    ok = cuda_gemv.ready()
    print(f"  nvrtc compile: {'OK' if ok else 'FAILED'} in {time.perf_counter() - start:.1f}s")
    if not ok:
        return

    def clock(fn, operands, reps=3, trials=5):
        for w in operands[:4]:
            fn(w)
        runs = []
        for _ in range(trials):
            torch.cuda.synchronize()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(reps):
                for w in operands:
                    fn(w)
            b.record()
            torch.cuda.synchronize()
            runs.append(a.elapsed_time(b) / (reps * len(operands)))
        return statistics.median(runs)

    for name, n, k in [("qkv", 6144, 2560), ("o", 2560, 4096),
                       ("gate_up", 19456, 2560), ("down", 2560, 9728),
                       ("lm_head", 151936, 2560)]:
        every = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
                 for _ in range(8)]
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        reference = F.linear(x, every[0])
        scale = reference.float().abs().max().item()

        try:
            mine = cuda_gemv.cuda_matmul(x, every[0])
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"  {name:8s} launch failed: {type(exc).__name__}: {exc}"[:200])
            continue
        err = (mine.float() - reference.float()).abs().max().item() / scale

        bf16_ms = clock(lambda w: F.linear(x, w), every)
        best, best_cfg = float("inf"), None
        for cfg in cuda_gemv.CONFIGS:
            ms = clock(lambda w, c=cfg: cuda_gemv.cuda_matmul(x, w, config=c), every)
            if ms < best:
                best, best_cfg = ms, cfg
        moved = every[0].numel() * 2
        print(f"  {name:8s} rel_err {err:8.5f}  cublas {bf16_ms * 1e3:6.1f}us "
              f"{moved / (bf16_ms * 1e-3) / 1e12:.2f}TB/s   "
              f"cuda {best * 1e3:6.1f}us {moved / (best * 1e-3) / 1e12:.2f}TB/s   "
              f"speedup {bf16_ms / best:.2f}x  {best_cfg}", flush=True)
        del every
        torch.cuda.empty_cache()
