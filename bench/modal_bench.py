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
    .add_local_dir("engine", "/root/engine")
    .add_local_file("bench/harness.py", "/root/harness.py")
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
def benchmark(shapes=None, samples: int = 5):
    """Time and check the engine the way the judge would."""
    import sys

    sys.path.insert(0, "/root")
    from harness import run

    _describe_gpu()
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


def _describe_gpu():
    import torch

    prop = torch.cuda.get_device_properties(0)
    print(
        f"{prop.name}  {prop.total_memory / 2**30:.0f} GiB  "
        f"{prop.multi_processor_count} SMs  torch {torch.__version__}",
        flush=True,
    )


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
def main(samples: int = 5):
    benchmark.remote(samples=samples)


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
