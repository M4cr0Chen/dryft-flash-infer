# Engine log

Not submitted. One entry per change, with what it measured.

## Roofline (H100 SXM, 3.35 TB/s, 989 TFLOPS BF16)

Weights streamed per forward pass: 3.633 B non-embedding + 0.389 B tied LM head
= **8.045 GB**. KV per token = `2 * 8 * 128 * 2 B * 36` = **144 KiB**.

```
decode tok/s  <=  a * B * BW / (8.045 GB + B * L * 144 KiB)
```

`a` is mean accepted tokens per pass, 1 without speculation. At batch 1 that is
a hard 417 tok/s: 8 GB per pass has no reuse in 50 MB of L2, so nothing but
speculation breaks it.

Geometric mean over the public shapes: a strong engine reaches ~950, the
physical ceiling is ~1170. Leaderboard top was 1095.1 on the hidden three,
which puts it at roughly 85-88% of a non-speculative roofline.

Prefill share of the clock at the ceiling: 8% for `1x512->32`, **50%** for
`4x2048->32`, 18% for `16x512->128`. Short-output workloads are prefill
problems, not decode problems.

## Measured on H100 (Modal, pinned runtime)

`modal run bench/modal_bench.py` — five samples per shape, judge's own gates.

| shape | native | stage 1 | + flash decode | gates |
| --- | ---: | ---: | ---: | --- |
| public-0  b1 x 512->32   |  37.0 |  175.6 |  **211.9** | pass |
| public-1  b4 x 2048->32  | 118.3 |  349.6 |  **471.3** | pass |
| public-2  b16 x 512->128 | 560.2 | 2345.9 | **2760.4** | pass |
| **geomean** | ~157 | 524.1 | **650.8** | |

Then per-shape projection selection: **691.4** (230.9 / 491.5 / 2911.9).

Spread 0.003-0.007 (gate 0.25). TTFT ratio 0.40-0.58, TPOT 0.17-0.18 (gate
1.10) -- no latency pressure at all. Peak memory 0.18 (gate 0.90). Tie gap
0.0000 on two shapes and 0.1250 on public-2, against a 2.0 margin; the one
non-exact argmax is the bf16 near tie the margin exists for.

### Where the time goes, batch 1

| stage | before | after |
| --- | ---: | ---: |
| prefill | 9.97 ms | 9.79 ms |
| decode step | 5.537 ms | **4.348 ms** |
| effective bandwidth | 1.45 TB/s | **1.85 TB/s** |

Prefill is already near roofline and is not the problem. Decode is: 4.35 ms
against a 2.75 ms bandwidth floor.

Attention was 46% of the decode step. SDPA costs ~20 us per call in fixed
overhead no matter how little cache it reads (0.11 TB/s on 2.2 MiB), and the
mask doubled that to 40 us by forcing the mem-efficient backend off flash.
Thirty-six launches per step is pure latency. The split flash-decode kernel
replaced it and took the step from 5.54 to 4.35 ms.

Remaining, measured per 36 layers at batch 1: `o_proj` 1.45 TB/s and
`down_proj` 1.79 TB/s are the weak projections, against `gate_up` at 2.48 and
`lm_head` at 2.93. Plus ~180 small elementwise launches a step.

## Stage 1 — custom forward, static cache, graphed decode

- Own forward, no `Qwen3ForCausalLM` dispatch. Fused QKV and gate/up weights,
  built once in `__init__`.
- Triton: RMSNorm, residual-add fused into the norm that reads it, per-head
  norm + RoPE writing K and V straight to their cache slots, SwiGLU.
- Preallocated `[layers, batch, kv_heads, capacity, 128]` K and V.
- One CUDA graph over the whole decode step. Position and mask live on the
  device so the graph needs no host input.
- Decode attention is one SDPA call: the four query heads of a KV group become
  four query positions, so grouped-query attention needs no `repeat_interleave`
  copy of the cache.
- Prompt upload through `array.array` and a pinned buffer, not
  `torch.tensor(nested_list)`, which is ~15 ms of pure Python at 128 K ids.
- Token stream stays one step ahead of the host: replay, enqueue the D2H, and
  yield the *previous* step while the current one is still in flight.

### What is tested, and how

No local GPU, so the suite runs on CPU with torch stand-ins for the Triton
kernels (`kernels/reference.py`, selected automatically when Triton is absent).

- `tests/test_numerics.py` — every fused kernel's arithmetic against the real
  Transformers 4.51.3 modules, **bit-exact**. This is the cast-placement rule.
- `tests/test_engine.py` — the whole engine against native Qwen: token
  equality on four shapes, prefill logits **bit-identical**, the judge's own
  teacher-forced replay rule, state reset between calls, batch-split prefill,
  and that the native fallback survives the weight relayout.
- Both suites are mutation-tested. Every mutation that is observable in
  bfloat16 is caught; `residual * 1.001` is not, because 0% of bf16 values
  change under it (resolution is 0.39%).

Untested until the platform runs it: Triton codegen and CUDA graph capture.

### Safety nets

- `Engine.__init__` runs `_self_check`: a real generation on a small shape,
  compared to native Qwen, applying the judge's tie-margin rule at every
  prompt position. It prints the logit delta, the worst tie gap, and whether
  Triton and the graph are live. On failure the engine serves native Qwen for
  the rest of the process instead of crashing the workload.
- **`SELF_CHECK_TOLERANCE` is a guess (1.0).** Read the delta the first run
  prints and tighten it. A correct engine should be well under 0.1.
- The native model is kept alive as the fallback, costing ~4.7 GiB of
  duplicated q/k/v/gate/up. Drop it if a workload ever returns `memory_limit`.

## Where the decode step actually goes (traced, batch 1)

`modal run bench/modal_bench.py::trace`. Kernel time 4207 us, launch gap only
239 us. **The GEMMs are 84% of the step**; every Triton kernel together is 15%.
So the small-kernel count was never the problem, and fusing them further is not
where the time is.

Per projection, measured streaming all 36 layers' weights (not one in a loop --
qkv is 31 MiB and o_proj 21 MiB, both inside a 50 MiB L2, so a tight loop
reports L2 bandwidth and picks the wrong kernel):

| projection | GB/step | `linear [n,k]` | `mm [k,n]` | triton | taken |
| --- | ---: | ---: | ---: | ---: | --- |
| qkv | 1.13 | 2.08 | **2.30** | 1.57 | cublas-t |
| o | 0.76 | 1.66 | **1.76** | 1.00 | cublas-t |
| gate_up | 3.59 | 2.47 | 2.46 | **2.65** | triton |
| down | 1.79 | 1.83 | **2.07** | 1.45 | cublas-t |
| lm_head | 0.78 | 2.94 | 2.99 | **3.04** | triton |

cuBLAS is layout-sensitive: transposing the weight to `[in, out]` and using
`torch.mm` is worth 10-13% on three of the five. The engine races cuBLAS in
both layouts against a swept Triton config during warmup and keeps the winner,
requiring a 3% margin before moving off cuBLAS so a noisy measurement cannot
make the engine slower.

`o_proj` and `down_proj` are still the floor at 1.4-2.1 TB/s. Both have N=2560,
which is only 2560 outputs to spread over 132 SMs against a long K reduction.
Neither cuBLAS layout nor my split-K kernel cracks it. Getting those two to 2.9
TB/s is worth about 13% of the step and is the next real target.

## Done

1. ~~Measure.~~ Modal H100 harness replicating the judge: `bench/harness.py`.
2. ~~Flash-decoding attention in Triton.~~ +26% geomean.
3. ~~Per-shape projection selection.~~ +6%.

## Next, in order of expected payoff

3. **`o_proj` and `down_proj`.** The last weak projections, ~13% of the step.
   Needs a better split-K Triton kernel than mine: deeper pipelining, async
   copies, `tl.max_contiguous` hints. Beating cuBLAS at a skinny reduction is
   genuinely hard and my first attempt lost.
4. ~~Fewer launches.~~ Not worth it: the trace says launch gap is 239 us of
   4446, and all the Triton kernels together are 625 us. Dropped.
5. **Prefill** is already near roofline; leave it.
6. **Speculative decoding.** Last. No draft model exists in the sandbox, so it
   is n-gram or Jacobi self-speculation. Watch the **25% spread gate**:
   acceptance varies per prompt, and a workload that runs 1.3x on one sample
   and 1.9x on another scores nothing. Current spread is 0.003, so there is
   room to spend.

## Settled, from `GET /api/v1/challenges`

- **Six** private workloads decide the score, not three. `AGENTS.md` and
  `QWEN_ENGINE_CONTRACT.md` both say three; they are stale.
- **Public runs exist** and never rank: one sample per public shape.
- The public shapes carry `regime: 1, 2, 3`. Six private workloads over three
  regimes suggests two private shapes per public regime, which would make the
  public three representative proxies.
- Budgets: 2100 GPU-seconds, 2400 run-seconds, 600 compile-seconds.
- Hardware is **H100 80GB HBM3 SXM**, 132 SMs, 3.35 TB/s. Confirmed on Modal.

## Still unknown

- `SELF_CHECK_TOLERANCE` is 1.0 and the real engine reports a 0.7500 logit
  delta on the self-check shape. That is close. The delta is the max over every
  prompt position against a full native forward, and the contract says native's
  own replay drifts up to 0.75, so this is the expected floor rather than our
  error -- the worst *tie gap*, which is what the judge grades, is 0.0000.
  Leave the tolerance at 1.0; do not tighten it below 0.9.
