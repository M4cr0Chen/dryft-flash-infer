# Engine log

Not submitted. What was measured, what it cost, and what it bought.

## Where it stands

**Ranked: 919.67 tok/s** on the six hidden workloads. Leaderboard top was 1130.6.

Three submissions of progressively faster engines scored 915.87, 919.67 and
906.98. That is not a ranking of the engines; see the next section.

## The measurement lesson, first, because it governs everything else

**Run-to-run variance is comparable to every improvement made here.** The
native reference is identical code on every run and measured 37.5, 39.0, 44.2,
46.1, 47.4 and 57.1 tok/s on the same shape across the day. Platform scores
moved ±3% on engines that only got faster.

So the geometric mean of a single run cannot resolve a 3-5% change, and several
conclusions drawn from it during the day were wrong in both directions.

What *is* reliable:

- **Traced step time** (`modal_bench.py::trace`), same method every time.
- **Ratios inside one run** -- latency gates, or a candidate against cuBLAS in
  the same warmup race.
- **Per-kernel GPU time** from the profiler.

Judge changes on those. Treat a single submission score as ±3%.

## The roofline

Qwen3-4B, H100 SXM: 3.35 TB/s, 989 TFLOPS BF16, 132 SMs.

Weights streamed per decode pass: 3.633 B non-embedding + 0.389 B tied LM head
= **8.045 GB**. KV per token = `2 * 8 * 128 * 2 B * 36` = **144 KiB**.

```
decode tok/s  <=  a * B * BW / (8.045 GB + B * L * 144 KiB)
```

At batch 1 that is a hard 417 tok/s with no speculation: 8 GB per pass has no
reuse in 50 MB of L2.

Measured against the absolute bound we sit at **58%**, consistently across all
three public shapes (57 / 58 / 60%). The leaderboard top is at about 72%.

## Decode step, batch 1

```
5.597 ms   SDPA decode attention
4.348 ms   + Triton flash-decoding            (-22%)
4.446 ms   re-traced after projection selection
4.111 ms   + FP8 gate_up, CUDA GEMV, fused SwiGLU   (-8%)
```

At 4.111 ms, traced: GEMMs 3388 us (82%), small kernels 585 us, launch gap
128 us. Effective bandwidth over the whole step is 1.96 TB/s of 3.35.

An accidental controlled experiment: Modal sometimes substitutes an **H200**,
which has 43% more bandwidth and returned only 30% more throughput. So roughly
**three quarters of the step is bandwidth and one quarter fixed latency** --
matching the traced split almost exactly.

GPU diagnostics during decode: SMs 97.6% busy, **memory controller 62.3%**,
clocks pinned at maximum, no throttling, 489 W, 49 C. The SMs are busy waiting.

## What worked

| change | effect | where |
| --- | --- | --- |
| Custom forward, static cache, CUDA graph | 5-7x over native | `engine.py` |
| Triton flash-decoding, split along KV | step 5.60 -> 4.35 ms | `kernels/attention.py` |
| Per-shape kernel and layout race at warmup | +6% | `kernels/gemm.py` |
| FP8 E4M3, group-128 scales | wins gate_up at batch 1 | `kernels/fp8.py` |
| CUDA JIT via NVRTC, no toolkit | enables all of the below | `kernels/cuda_jit.py` |
| Fused gate_up + SwiGLU in CUDA | 1.24x at batch 1, 1.07x at 4 | `kernels/cuda_mlp.py` |

The single largest find was that **SDPA costs ~20 us a call in fixed overhead**
no matter how little cache it reads, and an explicit mask doubles it by forcing
it off the flash backend. Thirty-six launches a step of pure latency.

## What did not work, and why

Each of these is a real measurement, not an abandoned attempt.

**A Triton GEMV for the projections.** Two rewrites, both lost to cuBLAS.
`o_proj` moves 21 MB, which is 7 us of work at 3 TB/s against a 3-5 us kernel
launch -- it is launch-bound, and split-K makes it worse because the reduce is
a second launch. cuBLAS is near the practical floor for these shapes.

**FP8 for bandwidth.** Accuracy is fine: group-64 scales leave the emitted
token at worst **0.375 logits** below the true argmax against a 2.0 margin,
less than the 0.75 the contract says native drifts from itself. But halving the
bytes buys **1.30x, not 2x** -- streaming the same weight reaches 2.50 TB/s in
bfloat16 and 1.63 in E4M3, because an 8-bit element carries half as much per
memory request and the read turns latency-bound. Byte-counting overpromised.

**Speculative decoding.** Exact by construction and verified token-identical.
n-gram acceptance on real prose is a mean **1.68 tokens per pass** -- but it
ranges **1.05 to 3.20 across prompts**, and sample time goes as its reciprocal,
so five samples spread about 100% against a 25% gate. Governing the rate down
to 1.25 fixes the spread and leaves less than the per-pass cost of drafting.
Left in, off by default: `DRYFT_DRAFT=8`.

**A megakernel QKV stage.** Correct to one bfloat16 ULP and one launch instead
of four, using a grid barrier under cooperative launch. Slower anyway: the
projection inside it runs at ~1 TB/s where cuBLAS gets 2.15, and 6144 output
rows cannot hide that. Not wired in; see `kernels/cuda_qkv.py`.

**The rule these keep returning:** fusion pays exactly where the projection
inside it is already competitive with cuBLAS, which is the large-N shapes. The
MLP fusion won on a 19456-row weight; the QKV fusion lost on a 6144-row one.

## The toolchain, all verified on the benchmark's runtime

The container has no nvcc and no ninja, so `cpp_extension.load_inline` cannot
run. It does have `libnvrtc` (a torch wheel dependency, so it is wherever torch
is) and the driver. That is enough:

- CUDA C++ from a Python string -> NVRTC -> PTX -> driver. **Compiles in <1 s.**
- Driver-API launches **capture into CUDA graphs** and replay correctly.
- **Cooperative launch with a hand-rolled grid barrier also captures**, with
  1056 resident blocks available (8/SM x 132).
- NVRTC has no toolkit headers, so kernel sources include nothing and carry
  bfloat16 as `unsigned short` with two bit helpers.

Quantisation and JIT CUDA were both cleared by the organiser; the written rules
still say otherwise, so check before relying on either.

## Gates, with the margins we actually have

| gate | limit | ours | headroom |
| --- | ---: | ---: | ---: |
| TPOT ratio | 1.10 | 0.16 | 6.9x |
| TTFT ratio | 1.10 | 0.55 | 2.0x |
| tie margin | 2.0 logits | 0.25 | 8x |
| peak memory | 90% | 28% | 3.2x |
| load budget | 300 s | 22 s | 13x |
| **sample spread** | **0.25** | **0.15** | **1.7x** |

Spread is the only one that is close, and it is what made speculation
unshippable. Note 0.15 is on a 32-token workload *without* speculation -- short
workloads are intrinsically noisy, and that is a standing risk.

## Settled facts about the benchmark

From `GET /api/v1/challenges`, against the starter docs which are stale:

- **Six** private workloads decide the score, not three.
- Public runs exist and never rank; public shapes carry `regime: 1, 2, 3`.
- Budgets: 2100 GPU-seconds, 2400 run-seconds, 600 compile-seconds.
- Hardware is H100 80GB HBM3 SXM, 132 SMs.
- A push is picked up **only when auto-run is on**. With it off the delivery is
  accepted and then *ignored*, and no submission is created at all. "Check
  delivery" reports the last delivery, it does not replay it.
- Submissions pin a commit hash, and a multi-commit push submits only the tip.
- `POST /api/v1/submissions` is 405; the CLI's `submit` cannot work. The CLI
  also needs `DRYFT_API=https://htn.dryft.ai` set explicitly despite the docs.

## If picking this up again

1. **Measure on the traced step, not the geomean.** Three hours were spent
   reading ±3% noise as signal.
2. The only untouched pool is the 585 us of small kernels plus 128 us of gap,
   and fusion is the only thing that reaches it -- but only around projections
   where a hand-written GEMM already matches cuBLAS.
3. Extending FP8 and the CUDA kernels above batch 1 would apply what already
   works to the other two thirds of the workloads. `fp8_matmul` and
   `cuda_matmul` both raise above batch 1 today.
4. Getting to 1130 means a GEMM that beats cuBLAS on *every* shape first.
   Three attempts did not. That is the real blocker, not the fusion.

## Reproducing

```sh
modal run bench/modal_bench.py::fetch        # once, checkpoint to a volume
modal run bench/modal_bench.py --corpus      # judge's gates, real prompts
modal run bench/modal_bench.py::trace        # per-kernel time in the graph
modal run bench/modal_bench.py::utilisation  # throttling, memory controller
modal run bench/modal_bench.py::quant_probe  # FP8 accuracy vs the tie margin
modal run bench/modal_bench.py::accept_study # n-gram acceptance on prose
python -m pytest tests/ -q                   # 20 tests, no GPU needed
```

The CPU tests matter: every fused kernel's arithmetic is checked bit-exactly
against Transformers 4.51.3, and the whole engine against native Qwen, through
torch stand-ins for the Triton kernels. They are mutation-tested -- every
mutation observable in bfloat16 is caught.
