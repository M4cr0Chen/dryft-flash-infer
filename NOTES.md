# Engine log

Not submitted. What was measured, what it cost, and what it bought.

## 8-bit weights on the tensor cores, every batch, September 19 (evening)

The earlier FP8 kernels moved half of cuBLAS's bytes and took longer: 0.84 to
1.63 TB/s against cuBLAS's 2.0 to 2.5. Their dequantisation chain and 128-wide
K tiles left the memory system idle. `kernels/cuda_fp8.py` is the replacement,
CUDA C++ through the existing NVRTC route:

- one warp owns sixteen output rows; each lane streams two rows with 16-byte
  loads and keeps a whole 128-wide group in flight while consuming the last;
- bytes convert to bfloat16 pairs and go straight into `mma.sync.m16n8k16`
  as A fragments -- the weight columns are permuted once at load so that a
  lane's contiguous 16 bytes *are* its fragments, and the activation is
  permuted the same way into shared memory;
- the block scale is applied to the fp32 partial once per 64 values, so the
  dequantisation is exact and costs four FMAs per block;
- three epilogues: bf16, fp32 split-K planes, and SwiGLU (gate row `i` and up
  row `i + I` land in the same lane, so the pair needs no exchange).

Every projection including the LM head is 8-bit in decode. The warmup race
still decides per shape, with cuBLAS as the incumbent to beat by 3%.

### E4M3 failed the margin; INT8 replaced it

The first version stored E4M3 with 128-wide fp32 scales and measured +23.7%
on the public geomean. Then one public-2 sample put the emitted token
**4.25 logits** below native's argmax. The same prompt passed at 1.125 under
two other kernel configurations and failed at 3.625 under a third: E4M3's
three mantissa bits give every weight about 3% error, enough that a near-tie
early in a sequence flips, and the diverged prefix eventually meets a
position where the rounded model disagrees with native outright.

INT8 with one fp16 scale per 64 values has about 3.5x less RMS error at the
same byte count (+3.1% for scales). Bytes are stored offset by 128; the kernel
recovers the integer with a byte permute into the mantissa of 2^23 and one
subtraction, which is exact, and the integer is exact in bfloat16. The scale
per block is chosen from a short grid for least squared error instead of
taken from the block maximum. The kernel is about 5% slower than the E4M3
version at batch 1 (one more instruction per pair) and still 1.4 to 2.1x
cuBLAS on every shape at batches 1 to 16.

Same five public-2 prompts, worst gap per sample:

| weights | sample gaps | worst |
| --- | --- | ---: |
| bfloat16 | 0.250 0.125 0.375 0.125 0.250 | 0.375 |
| E4M3, all projections | 1.000 0.500 **4.250** 0.500 1.125 | 4.250 |
| INT8, all projections | 0.625 0.125 0.250 0.250 1.375 | 1.375 |
| INT8, qkv and o only | 0.250 0.125 0.500 0.250 0.188 | 0.500 |
| INT8, gate_up and down only | 0.500 0.625 0.250 0.250 1.375 | 1.375 |

The remaining 1.375 comes from the MLP weights, on one sample. It passes,
but it is 69% of the margin on ten thousand judged positions. Knobs, in the
order to pull them if an official run reports gaps above 1.5:
`DRYFT_FP8_PROJECTIONS=qkv,o,lm_head` keeps the MLP in bfloat16 and gives
back roughly half the gain; `DRYFT_FP8=off` is the previous engine.

| projection, batch | cuBLAS BF16 | Triton E4M3 (old) | CUDA INT8 (new) |
| --- | ---: | ---: | ---: |
| qkv b1 / b4 / b16 | 15.8 / 16.5 / 15.7 us | 13.5 / 20.5 / 19.0 | **11.5 / 11.9 / 13.5** |
| o b1 / b4 / b16 | 14.2 / 14.3 / 14.3 | 12.0 / 17.0 / 15.9 | **10.0 / 10.2 / 11.3** |
| gate_up b1 / b4 / b16 | 40.7 / 41.2 / 41.3 | 31.6 / 48.9 / 48.4 | **24.8 / 25.3 / 32.3** |
| down b1 / b4 / b16 | 25.7 / 26.6 / 27.2 | 22.5 / 30.8 / 29.6 | **16.6 / 16.9 / 21.3** |
| lm_head b1 / b4 / b16 | 333 / 283 / 299 | 218 / 338 / 328 | **158 / 163 / 193** |

At batch 32 the kernel loses to cuBLAS on every shape (0.64 to 0.86x): B
fragments for four n-tiles exceed what one warp can keep resident. The race
keeps cuBLAS there. A hidden workload at batch 32 or above gets no 8-bit gain
from this kernel; a two-warp-per-row-block variant is the next step if the
hidden shapes turn out that wide.

### Fewer launches around it

- **Split-K planes flow into their consumer.** `o` and `down` write fp32
  planes; the fused add-norm (`add_rms_norm_partials`) sums them. `qkv`
  writes planes; the fused norm/RoPE/cache kernel (`qkv_planes_norm_rope_to_cache`)
  sums them. Sums are sequential in the reduce kernel's order and round to
  bf16 once, so the result is bit-identical to running the reduce kernel; the
  GPU checks assert that on 712 cases. Saves 108 launches a step.
- **Single-split attention writes its output directly**, no merge kernel.
- **Attention layout race on by default.** Attention is a sixth of a batch-16
  step now that the projections are 8-bit; the race chose `(1, 64, 4)` there
  for 1.09x on attention alone.

### Weight families, not weight reconstruction

Speculative verification widths race only the family (8-bit or BF16) that
ordinary decode chose per projection. The old approach reconstructed FP8
weights to bf16 for the verification GEMMs; it is gone, along with its 2 GiB.
`_choose_matmuls(rows, families=...)` raises rather than silently switching
families, and `_ensure` then runs without short verification.

### What did not work

**FP8 prefill through `torch._scaled_mm`.** Per-token activation scales and
per-channel weight scales, MLP only or all four projections. Slower on every
shape (prefill 9.6 -> 19.5 ms at 1 x 512, 105 -> 219 ms at 16 x 512: the
quantisation pass over the activation costs more than the GEMM saves at
these sizes in torch 2.5.1) and outside the margin: worst tie gap 4.25 at
16 x 512. Removed.

**A division-free staging loop** for the activation (one row per iteration)
was 10% slower than eight scattered 16-byte loads in flight per thread.
Reverted. An unstaged variant reading B fragments straight from L1 never
won a race and is gone too.

### Measured, paired on one H100, five corpus samples per shape, against `8914520`

| shape | before | after | change | worst tie gap |
| --- | ---: | ---: | ---: | ---: |
| public-0 (1 x 512 -> 32) | 234.9 | 312.2 | **+32.9%** | 0.125 |
| public-1 (4 x 2048 -> 32) | 493.7 | 573.6 | **+16.2%** | 0.125 |
| public-2 (16 x 512 -> 128) | 2995.6 | 3385.4 | **+13.0%** | 1.375 |
| coverage-long (1 x 4096 -> 65) | 196.4 | 263.6 | **+34.2%** | 0.125 |

Public geometric mean 703 -> 846 tok/s, **+20.4%**. Every sample passes every
local gate; peak memory 27%; load plus warmup 69 to 97 s (the race now covers
more kernels, and the 300 s budget has room). Ten samples of public-2 hold
the worst gap at 1.375, on that one prompt. The coverage shapes (3 x 257,
8 x 1024, 32 x 256) pass with worst gaps of 0.19 to 0.25.

`paired-int8-final-20260919.json` and `coverage-int8-20260919.json` are the
sources. The E4M3 numbers are kept in `paired-cuda-fp8-20260919.json` and
`paired-cuda-fp8-final-20260919.json` (the failing run) for the record.

The two thirds of the step that is still not projections at batch 16 --
attention at 0.57 ms, the two add-norms at 0.27 ms, the launch gap at
0.12 ms -- is the next pool. The batch-32 GEMM is the other.

## Organizer clarification and FP8/short speculation, September 19

The user explicitly confirmed that the organizer allows quantization and that
the published prohibition is stale. This supersedes the earlier opt-in policy
and the historical note below about an unverified exception. The native
teacher-forced two-logit margin still applies at every emitted position.

The working engine enables group-128 E4M3 weight compression for decode,
selected per projection by the warmup race. Activations, prefill, and the tied
embedding/LM head retain BF16. Packed weights are cached once and shared across
ordinary and verification graphs. Batched FP8 now has a tensor-core path;
BF16 remains available whenever it wins the timing race.

Short verification has moved from `bench/short_spec.py` into the submitted
`engine/kernels/speculation.py`. Batch one keeps separately tuned one-, two-,
and three-row graphs. Two draft tokens, order-two prompt lookup, and a 1.15
tokens/pass governor are the default. Empty proposals use ordinary decode.
Only the accepted prefix advances the live cache position. Each generation
resets the prompt lookup and overwrites the prompt cache. Verification checks
the chosen target implementation; it does not undo quantization error.

`DRYFT_FP8=off` and `DRYFT_SHORT_DRAFT=0` independently disable these features.
Attention tuning remains opt-in.

### What the first combined comparison established

`bench/results/paired-fp8-short-20260919.json` compares against `0949421` on one
H100, with five samples per public shape. Batch one rose from 249.1 to 255.2
tok/s (+2.4%); batch four fell from 496.4 to 492.3 (-0.8%); batch sixteen fell
from 2997.1 to 2987.8 (-0.3%). The geometric-mean change is **+0.4%, not a
convincing overall gain**. Every public sample passed; this did not establish
correctness on other prefixes, as the failure below demonstrates.

`bench/results/fp8-kernel-study-20260919.json` compares expanding each weight
before the dot product against applying group scales after each group dot.
Both use BF16 tensor-core products. The expanded-weight implementation won
every tested projection/batch pair, but still lost to cuBLAS at batches
2, 3, 4, and 16. This rejects these two implementations; it is not an FP8
hardware ceiling. Activation quantization and native FP8 tensor-core products
were not implemented or measured in this round.

### Precision consistency across speculative passes

The initial full corpus run failed on prose seed 4100, prompt length 512,
output length 128. The failure reproduced with both one- and two-token drafts:
output position 111 selected token 944 instead of native's token 29, a
**22.8125-logit** gap. `fp8-speculation-diagnostic-20260919.json` retains the
75-trial diagnostic, including both failures. The short public examples did
not expose it.

`fp8-speculation-trace-20260919.json` and `fp8-speculation-ablation-20260919.json`
show that graph and eager execution agreed at the bad step. Ordinary decode
used compressed QKV/gate-up weights, whereas verification switched to original
BF16 weights while retaining the cache produced by earlier passes. Disabling
either compressed projection changed the generated prefix and removed the
failure. The MLP-only ablation passed all 225 corpus trials, worst gap 0.875.
Forced-prefix checks also show that original BF16 custom decode can disagree
strongly on that particular quantized-generated prefix; a passing unquantized
generation on another prefix is not sufficient evidence.

The fix preserves the ordinary path's weight choices in every verification
width. Compressed weights are reconstructed into BF16 once, permitting the
faster BF16 verification GEMMs without switching back to original weights.
Original-weight fused MLP kernels are excluded when the gate/up weights are
compressed. This adds weight storage during warmup, not per-token conversion.

`fp8-consistent-speculation-20260919.json` rechecks 75 trials at 512/128:
**all passed**, worst gap **1.0**. For the default capped two-token policy,
speedups over ordinary FP8 decode were 1.037x prose, 1.107x code, and 1.122x
technical text; maximum spread was 11.3%. Uncapped speculation still exceeded
the 25% spread gate. The final full-corpus and paired measurements follow below.

`quantized-speculation-final-20260919.json` repeats the complete 225-trial
corpus study with the corrected implementation. All teacher-forced trials pass.
Worst gap is 1.0 logits; the default policy's maximum sample spread is 12.6%.
The default policy's speedups over ordinary FP8 decode are:

| Prompt / output | Prose | Code | Technical |
| --- | ---: | ---: | ---: |
| 512 / 32 | 1.000x | 1.088x | 1.000x |
| 2048 / 32 | 0.996x | 1.088x | 1.039x |
| 512 / 128 | 1.035x | 1.106x | 1.112x |

This supports a conditional gain on repetitive code/technical continuations,
with approximately flat short prose results. It does not imply the same gain
on the unknown hidden corpus.

### Final paired comparison

`paired-fp8-consistent-20260919.json` uses the final engine source, verified by
its SHA-256, and five samples per shape against `0949421` on the same H100:

| Batch / prompt / output | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 / 512 / 32 | 250.6 | 254.7 | +1.6% |
| 4 / 2048 / 32 | 501.4 | 500.5 | -0.2% |
| 16 / 512 / 128 | 3042.9 | 3045.6 | +0.1% |
| 1 / 4096 / 65 | 199.2 | 208.0 | +4.4% |

The public-shape geometric mean improved approximately **0.5%**, which remains
too small to call a robust general-throughput gain. Every baseline/candidate
sample passes the local gates. The long-context candidate's spread is 15.5%
and peak memory is 35.7% of the device. The extra reconstructed verification
weights explain the memory increase over the first prototype.

Final checks: 40 CPU tests, 20 FP8 GPU indexing/reduction cases, 54 attention
cases, six bit-exact RoPE cases, the full corpus replay, and Dryft archive
validation pass. The initial expanded-shape coverage additionally passed
batches 3, 8, and 32, where the tuner retained BF16 and speculation was inactive;
batch 32 reached a native gap of 1.875, demonstrating why the tolerance cannot
be assumed from the public examples alone.

The latest user-confirmed Dryft score remains **940 tok/s**. No leaderboard
gain has been established by these local experiments. The next substantial
unmeasured direction is native FP8 tensor-core matrix multiplication, including
MLP prefill; further attention tuning has shown little total-time benefit.

## MLP, attention, and short verification, September 19

Starting point: commit `0949421`; the user reports **940 tok/s** on Dryft.
The measurements below are local H100 studies, not hidden leaderboard scores.

### MLP profile

`bench/results/mlp-profile-20260919.json` attributes CUDA time to each prefill
projection and times the selected decode projections across all 36 weights.

| Batch / prompt | Prefill wall, ms | Gate/up + down, ms | Decode wall, ms | Gate/up + SwiGLU + down, ms |
| --- | ---: | ---: | ---: | ---: |
| 1 / 512 | 9.38 | 5.09 | 3.98 | 2.17 |
| 4 / 2048 | 122.96 | 63.16 | 4.72 | 2.28 |
| 16 / 512 | 113.25 | 64.04 | 4.66 | 2.29 |

MLP work is about half the stage time. Prefill already dispatches to Hopper
cuBLAS tensor-core kernels and PyTorch FlashAttention. On the 8192-row prefills,
SwiGLU itself takes about 5.8 ms across all layers; eliminating only its launch
and intermediate traffic cannot eliminate the much larger matrix-product cost.
This supports targeting the MLP projections, without establishing their limit.
No new fused MLP implementation was benchmarked in this study.

The profiler event table includes nested scopes with attributed CUDA time and
must not be summed. The projection breakdown is disjoint. Decode microtimings
include two state-restoration copies and are not generation throughput.

### Attention dispatch

`bench/results/attention-study-20260919.json` sweeps 30 configurations at the
start and end of generation. Winners were `(splits, block, warps)` of
`(16,64,4)`, `(4,128,4)`, and `(1,64,4)` for the three public shapes. Isolated
attention gains translated to roughly 0-1% of complete decode, not a large
throughput improvement. Some token choices change with the reduction layout;
teacher-forced gaps, rather than token identity, determine correctness.

The candidate adds a direct normalized-output path for one partition, plus a
warmup-only dispatch race over eight configurations. It times all layers at
both ends of generation, checks finite outputs, and keeps the original unless
the attention measurement improves by at least 5%. No prompt-dependent tuning
runs during measured generation. The final default retains the original
dispatch; `DRYFT_ATTENTION_TUNE=on` enables the experimental race.

`bench/results/paired-attention-20260919.json` compares fresh baseline/candidate
processes on one H100, with five corpus samples per public shape:

| Shape | Baseline tok/s | Tuned tok/s | Change |
| --- | ---: | ---: | ---: |
| public-0 | 248.8 | 248.7 | approximately flat |
| public-1 | 500.9 | 497.7 | -0.6% |
| public-2 | 3003.9 | 3037.0 | +1.1% |

The geometric-mean change is **+0.14%, inconclusive at this scale**. All 15
candidate samples pass; worst gap 0.375, longest load plus warmup 35.83 seconds.
The archive's subsequent MLP row-count guard does not affect these power-of-two
batches. A separate controlled comparison holds projection choices fixed and
alternates baseline/candidate per prompt to isolate the smaller attention effect.
`bench/results/attention-controlled-20260919.json` measured +0.68%, -0.84%, and
+0.97% on the three shapes. Batch four retained the original attention in both
graphs, making its difference a useful indication of measurement/graph-layout
variation. The overall change remains too small to call a reliable gain;
attention tuning stays opt-in. All 30 controlled-comparison samples also pass.

GPU validation checks 54 attention cases against SDPA, including 1/2/3 query
positions, first/last cache positions, empty partitions, non-power-of-two
capacity, and poisoned unused cache slots. Maximum output error was 0.00390625.
The six existing fused RoPE checks remain bit-exact.

### Short speculative verification

`bench/short_spec.py` retains the ordinary one-row graph and captures separately
tuned two- and three-row graphs. Empty proposals use ordinary decode; rejected
drafts emit the target correction and roll the live cache position forward by
only the verified output count. The prototype is outside the submitted engine.

`bench/results/short-speculation-20260919.json` contains **225 trials**: baseline
and four policies, five prompts in each of three domains, at three batch-one
prompt/output shapes. Every trial passed teacher-forced replay, worst gap
0.375. Different BF16 forward shapes sometimes change a near-tie token, so
the artifact records ordinary-stream token equality separately from validity.

For two draft tokens, order-two lookup, and a 1.15 tokens/pass governor:

| Prompt / output | Prose speedup | Code speedup | Technical speedup |
| --- | ---: | ---: | ---: |
| 512 / 32 | 0.990x | 1.103x | 1.014x |
| 2048 / 32 | 0.996x | 1.099x | 1.067x |
| 512 / 128 | 1.025x | 1.123x | 1.128x |

Maximum spread for that policy was 12.6%; across all capped policies, 14.0%.
Uncapped order-three proposals reached 1.31x median in one code case but broke
the 25% spread gate in multiple code/technical cases (up to 50.5%). This is a
useful conditional gain, not evidence for a corpus-independent default. The
prose corpus is Pride and Prejudice, code is the pinned Qwen implementation,
and technical text is this repository's implementation guide. None is a claim
about the judge's hidden corpus.

The first prototype run exposed an existing memory-safety bug in `cuda_mlp.py`:
the wrapper rounded three rows up to a four-row specialization, while the CUDA
kernel unconditionally loaded and stored all four rows. The same issue affected
other non-power-of-two batches. The wrapper now rejects unsupported row counts
before launch; the existing tuner then retains a general GEMM. Four CPU
regressions cover 3, 5, 9, and 17 rows. The corrected three-row verification
path completed the full GPU study. A separate benchmark text-encoding error
was also corrected before the final recorded run.

Modal runs: [MLP profile](https://modal.com/apps/macrochen05/main/ap-kROpMoIwtKOeK3z9Qziphr),
[attention sweep](https://modal.com/apps/macrochen05/main/ap-c2GpIm52SG8mWESJZBEpoe),
[paired attention](https://modal.com/apps/macrochen05/main/ap-qcUwIBAMxMhOzHSllRelU6),
[controlled attention](https://modal.com/apps/macrochen05/main/ap-rFqFAWol6Qwo6UhFsUyOdm),
[short verification](https://modal.com/apps/macrochen05/main/ap-m0mc6cJojSkvuVZ9NDbTQt).

Final validation: 37 CPU tests pass; 54 GPU attention cases and six fused RoPE
cases pass; the Dryft CLI accepts the source archive. The safety fix is enabled.
The attention tuner is opt-in, and short verification remains a benchmark
prototype. No official run was triggered by this investigation.

## Re-audit, September 19

The historical conclusions below are hypotheses, not hardware limits. In
particular, the old `trace` and `utilisation` replay loops overran their cache
and RoPE allocations; those traced step times need to be remeasured.

The repaired tuner races captured graphs, compares MLP fusion against the
already selected projection, and chooses verification kernels for the actual
number of draft rows. The local harness now checks every measured sample and
uses fresh processes per workload. FP8 is opt-in because the published rules
still forbid quantization; the claimed organizer exception below is unverified.

A paired H100 run against commit `9b95be8`, with BF16 on both sides and five
corpus samples per public shape, measured the tuning repair:

| Shape | Before, tok/s | Repaired, tok/s | Change |
| --- | ---: | ---: | ---: |
| public-0 | 230.3 | 240.7 | +4.5% |
| public-1 | 490.8 | 496.9 | +1.2% |
| public-2 | 2928.2 | 2972.6 | +1.5% |

The public geometric mean increased 2.4%. All 15 candidate samples passed
replay; the worst gap was 0.375 logits. The largest sample spread was 1.1%,
and the longest load plus warmup was 39.3 seconds. This is not a hidden score
or an official evaluation. Source hashes and full sample metrics are in
`bench/results/paired-bf16-20260919.json`.

The valid batch-one trace was 3.906 ms including two state-restoration copies.
It selected existing Triton/CUDA projections that eager timing had passed over.
For public-1, prefill accounts for 44.1% of total generation time, making it a
substantial remaining target. See `bench/README.md` for the measurement method.

The next edit combined Q normalization/RoPE and K/V cache writes into one
Triton launch per layer, leaving the projection kernels intact. Direct GPU
checks were bit-exact against the two old kernels for batches 1, 4 and 16,
both decode and multi-token inputs, and nonzero cache batch offsets.

A second paired H100 run, against the repaired engine above, measured:

| Shape | Separate Q/KV, tok/s | Fused Q/KV, tok/s | Change |
| --- | ---: | ---: | ---: |
| public-0 | 240.2 | 245.9 | +2.4% |
| public-1 | 494.1 | 498.5 | +0.9% |
| public-2 | 2964.0 | 3005.9 | +1.4% |

That is another 1.6% in the public geometric mean. All 15 candidate samples
passed, with a worst replay gap of 0.375 logits. Small per-shape differences
remain subject to noise. `bench/results/paired-rope-20260919.json` records the
final source hash and all samples. No official evaluation was submitted.

Validation: 33 CPU tests pass, including the tuner regression and all-sample
replay checks; the actual fused GPU kernel passes six bit-exact cases; fixed
decode replay stays at its original position/output over 65 replays on every
paired workload. The CPU bit-exact prefill test now uses native's explicit
KV-head expansion so it compares the same attention backend; the unmodified
commit failed that test on this Mac with a 0.005859375 logit difference.

The following sections preserve the earlier record; their strong claims about
cuBLAS, speculation, and the only remaining optimization pool are not accepted
as established limits.

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
