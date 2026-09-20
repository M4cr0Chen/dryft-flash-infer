# Engine log

Not submitted. What was measured, what it cost, and what it bought.

## Applying the ablations, September 20

Starting point: `8a4f75d` (engine identical to the ring version `43822c0`).
The changes implement the passing findings in `ABLATION_REPORT.md`:

- **Separate split-K residual work from decode RMSNorm.** A pointwise kernel
  sums FP32 planes sequentially, rounds the projection to BF16, adds and rounds
  the residual, then the existing RMSNorm runs. Direct BF16 projections retain
  their fused add/norm: separating those lost about 1% at batch 32. Prefill and
  native INT8 normalization/activation packing also retain their existing
  combined kernels. `DRYFT_DECODE_NORM=fused` restores the old plane consumer.
- **Attention target 128 instead of 132 programs.** Avoid doubling 128 programs
  merely to occupy the H100's last four SMs. At batch 16 this also removes the
  merge kernel. `DRYFT_ATTENTION_PROGRAM_TARGET=132` restores the prior rule.
- **Speculation target 1.20 for outputs of at least 64 tokens.** Shorter outputs
  keep 1.15; their higher-target ablations showed no median benefit and less
  spread headroom. `DRYFT_SHORT_LONG_TARGET=1.15` restores the old long-output
  governor. An explicit `DRYFT_SHORT_TARGET` still overrides both defaults.
- **Consume split-K planes directly in SwiGLU.** The new consumer replaces
  reduction-to-BF16 plus a separate activation launch. It retains the projection,
  SiLU and multiply BF16 casts. It is selected only for INT8-weight projections
  when it beats the existing complete operation by 3% and matches its output.
  It preserves BF16 activations; this does not force the native W8A8 MLP that
  failed the ablations. `DRYFT_PARTIAL_SWIGLU=off` disables it.
- **Skip unused mask updates** in custom decode and verification attention;
  SDPA fallback paths still construct their masks.

The new plane/SwiGLU consumer passed 40 GPU comparisons bit-exact against the
real CUDA plane reduction followed by the existing SwiGLU. The separated norm
passed 48 bit-exact cases, including batches 1--64 and 1--16 planes. All 42 CPU
tests pass, including a new check that shorter requests reusing a long warmup
cache return to the short-output governor.

### Final paired H100 comparison

Fresh processes, five corpus samples per shape; INT8 weights and the existing
native INT8 policy enabled in both versions. Source hashes are recorded in
`bench/results/paired-findings-final-20260920.json` and the tested candidate
hash was checked against the working engine before packaging.

| Shape | Ring baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| public-0, 1 x 512 -> 32 | 351.3 | 361.5 | +2.9% |
| public-1, 4 x 2048 -> 32 | 587.0 | 603.4 | +2.8% |
| public-2, 16 x 512 -> 128 | 3541.2 | 3685.2 | +4.1% |
| 1 x 4096 -> 65 | 259.1 | 272.1 | +5.0% |

Every local gate passes. Load plus warmup is 11--17 seconds in the development
comparison; candidate peak allocation is about 25.6% of the H100. Local memory
and latency checks are diagnostics, not substitutes for official eligibility.

The final prose/code/technical sweeps cover batches 1, 3, 8, 12, 16 and 32.
All candidate generations pass teacher-forced replay. Largest observed gaps:
1.375 at batch 16, 1.875 on the unchanged batch-32 projection path, and 0.5 on
the long batch-one outputs. Maximum candidate sample spread is 13.6%.
`findings-final-core-20260920.json` and `findings-final-coverage-20260920.json`
retain these results. Their within-process controls hold the selected projection
families and the partial-SwiGLU implementation fixed; the fresh-process paired
file above measures the complete change against the old engine.

`bench/findings_modal.py` runs the production implementations as component,
combined, and coverage experiments. `partial_swiglu_study.py` isolates the new
consumer. The first broad norm-separation candidate's batch-32 regression is
retained in `findings-wide-20260920.json`; it is not the final dispatch policy.

Official result pending for this candidate.

## Where the decode GEMM's time goes, and a tiled layout with a shallow ring, September 20

Starting point: `07110ef`, official 1177.3 tok/s. A fresh per-kernel trace of
one graphed decode step (`bench/modal_bench.py::trace`) on the current engine:

| Shape | Step | Projections + LM head | Attention | Add-norm + RoPE | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 x 512 | 2.84 ms | 2083 us (145 launches) | 243 us | 356 us | 130 us |
| 4 x 2048 | 3.31 ms | 2128 us | 618 us | 389 us | 148 us |
| 16 x 512 | 3.69 ms | 2529 us | 636 us | 414 us | 80 us |

Against INT8 weights of 4.15 GB plus the KV cache at 3.0 TB/s, every step sits
at 49-54% of the memory floor. The GEMMs are the pool: 2.53 ms at batch 16
where the bytes need 1.38 ms.

### Three hypotheses, two wrong

`bench/latency_probe.py` times, in the same graph harness as the warmup race,
an empty launch (1.55 us), a kernel that only streams the weight bytes, and the
incumbent. Streaming costs 2.2 us plus bytes at 3.1 TB/s. The incumbent sits
above that by roughly 2.5 us fixed plus 0.06 us/MB at batch 1 and 0.17 us/MB
at batch 16 (o 2.8 us, qkv 2.5, down 4.8, gate/up 5.6 over the floor at
batch 1; 3.7 / 4.2 / 9.0 / 11.3 at batch 16).

1. **Bytes in flight.** A per-warp cp.async ring of 4 or 8 groups in shared
   memory, bit-exact with the register kernel, was 0.90-1.03x on every shape.
   Deeper prefetch is not the limiter.
2. **Compute.** `bench/gemm_bisect.py` compiles the kernel with the MMA, the
   dequantisation, the staging or the epilogue removed. With MMA and dequant
   both gone, gate/up at batch 16 still takes 27.0 us for bytes that stream in
   18.1 us (full kernel 29.6). Compute is not the limiter either. Staging pays
   for itself (removing it costs 4-20 us at batch 16); the split-K planes cost
   up to 4 us per call at batch 16 on the widest shape.
3. **Occupancy and access pattern.** The register kernel needs 120-172
   registers a thread (`cuFuncGetAttribute`), so 2-3 blocks fit per SM, one
   at batch 32. Each warp load instruction also touches 16 rows 2.5 KB apart,
   64 bytes each.

### What shipped: tiled weights and a two-group ring

`cuda_fp8.Prepared` now stores the weight as `[rows/16][K/128][half][j][g][t][16 B]`:
the 16 rows x 128 bytes a warp consumes per group are one contiguous 2 KB
tile, and every warp load instruction reads a contiguous 512 bytes. It is a
load-time permutation of the fragment-permuted bytes (`row_major()` inverts it
for the native INT8 path), bit-exact, `DRYFT_TILED=off` restores row-major.
On its own it is worth 2-11% per projection.

The cp.async ring stays, at depth 2: the same two groups in flight as the
register kernel, but held in shared memory, which brings the kernel to 78
registers and 3-6 blocks per SM. Depth 2 on the tiled layout is where it
pays; depths 4 and 8 spend the occupancy back on shared memory.
`bench/tiled_probe.py`, one H100, twelve weights cycled, best configuration
of each, all bit-exact against row-major at the same warp count and split:

| Projection | Batch | Stream floor | Row-major | Tiled | Tiled + ring 2 | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qkv 16.2 MB | 1 / 16 | 7.0 | 9.9 / 11.6 | 9.7 / 11.2 | 9.5 / 11.5 | 4% / 3% |
| o 10.8 MB | 1 / 16 | 5.7 | 8.3 / 9.2 | 7.9 / 8.9 | 8.1 / 8.9 | 5% / 4% |
| gate/up 51.4 MB | 1 / 16 / 32 | 18.1 | 23.6 / 29.4 / 47.1 | 22.9 / 28.7 / 45.3 | 21.0 / 26.2 / 35.7 | 12% / 12% / 32% |
| down 25.7 MB | 1 / 16 / 32 | 10.2 | 15.0 / 19.1 / 26.8 | 14.3 / 17.9 / 26.5 | 12.4 / 15.5 / 22.6 | 21% / 23% / 19% |
| LM head 401 MB | 1 / 16 / 32 | 139.5 | 152.9 / 180.2 / 282.5 | 150.6 / 180.8 / 276.2 | 150.8 / 168.1 / 221.2 | 1% / 7% / 28% |

(microseconds per call). The warmup race now carries the five ring
configurations that won somewhere alongside six row-major ones; the losing
row-major entries left to keep the race inside the warmup budget. Load pays
one extra permutation per weight, about a second in total.

### Paired, end to end

Fresh processes, five corpus samples per shape, both engines with INT8
weights and the native INT8 MLP enabled, order alternating per workload
(`paired-tiled-20260920.json`, `paired-tiled-ring-20260920.json`):

| Shape | Baseline `07110ef` | Tiled only | Tiled + ring in the race |
| --- | ---: | ---: | ---: |
| public-0, 1 x 512 -> 32 | 339.1 / 330.2 | 341.5 (+0.7%) | 336.2 (+1.8%) |
| public-1, 4 x 2048 -> 32 | 576.1 / 573.1 | 581.1 (+0.9%) | 577.4 (+0.7%) |
| public-2, 16 x 512 -> 128 | 3524.1 / 3508.4 | 3577.6 (+1.5%) | 3509.1 (+0.0%) |
| 1 x 4096 -> 65 | 255.2 / 252.6 | 260.9 (+2.2%) | 259.3 (+2.6%) |

(tok/s; the baseline column lists each run's own baseline.) Every candidate
sample passes with a tie gap equal to or below its baseline's, as bit-exact
kernels should. Load plus warmup stayed at 13-20 s.

The traced step (`trace`, same shapes as above) says why batch 16 is flat.
At batch 1 the step went 2.837 -> 2.718 ms: GEMM kernels -275 us, and the
race's move of gate/up from the fused SwiGLU epilogue to planes plus a
separate SwiGLU cost back 100 us in reduce and SwiGLU launches. At batch 16
the step went 3.691 -> 3.671 ms: the GEMM kernels lost 314 us (2529 -> 2215),
but the winning ring configurations put gate/up on planes + `reduce_partials`
+ `_swiglu_kernel` (+94 +52 us, replacing the one-launch native INT8 kernel
the race now rated a wash), gave down sixteen planes instead of eight so the
add-norm consumer grew 69 us, and the two extra launches per layer added
68 us of gaps. The projection race sees the kernel and its reduce; it does
not see the consumer's plane traffic or the launch it displaces. The next
step here is a consumer-aware race for the planes projections (time
`matmul_partials` plus `add_rms_norm_partials` together) and a SwiGLU that
sums planes itself, which would bank the 314 us at batch 16.

### INT8 prefill through cuBLASLt, measured and stopped

Prefill is 11% / 53% / 18% of the measured time on the three public shapes
and its GEMMs run at 676 TFLOPS in cuBLAS BF16, so INT8 GEMMs through
`torch._int_mm` were the other candidate. Isolated at M=8192 on this runtime
they are 0.95x (qkv), 1.16x (o), 0.77x (gate/up), 1.39x (down) and 0.89x
(LM head) of `F.linear`, 1.03x in aggregate, before paying activation
quantisation and an int32 epilogue; a materialised `[K, N]` operand is 8x
slower still. FP8 `_scaled_mm` is the only 1.5-1.9x GEMM here and E4M3
weights already failed the margin in decode. No engine change; the probe
is `bench/modal_bench.py::int_mm_probe` on branch `prefill-int8`.

## Native integer MMA and fused activation packing, September 20

Starting point: `e9b6f19`, the user-reported best is 1161.6 tok/s. The saved
baseline lives at `/tmp/dryft-baseline-e9b6f19` for paired development runs.
These measurements are local H100 results, not a new leaderboard score.

The existing `cuda_fp8.py` stores INT8 but executes BF16 MMA after expanding
weights. `cuda_int8.py` instead packs weight and activation fragments for
`mma.sync.m16n8k32.s32.s8.s8.s32`. Weights retain the existing group-64
quantization and FP16 scales. Activations use FP32 scales per 64 values. A
four-group weight prefetch and shared-memory activation staging are necessary:
the first version reading activations directly from L1 lost every race.

The selected production path is **gate/up plus SwiGLU at batches 9--16**, only
where ordinary decode already chose INT8 weights and a warmup race measures a
3% win. Other batches retain the existing engine. The residual-add/RMSNorm
before this projection also packs its normalized BF16 output directly into
INT8. It preserves residual, normalization and gain-multiply BF16 rounding;
20 GPU cases check residuals, packed bytes and scales bit-exact against the
separate operations. This removes a standalone activation-quantization launch
and its intermediate write/read. Prefill and batch-one speculation are unchanged.

Controlled graph ablations on one H100, five alternating samples per shape:

| Shape | Existing engine | Native INT8 + fused packing | Change | Worst gap |
| --- | ---: | ---: | ---: | ---: |
| 16 x 512 -> 128 | 3386.2 | 3550.3 | +4.85% | 1.375 |
| 12 x 1024 -> 64 | 1813.1 | 1873.5 | +3.33% | 0.3125 |

Source: `bench/results/native-int8-policy-ablation-20260920.json`.
`DRYFT_INT8_MMA=off` restores the prior arithmetic; `DRYFT_INT8_NORM_FUSION=off`
isolates activation-packing fusion. Both are on by default in this candidate.

### Rejected configurations and the numerical limit

- **BF16 prefill CUDA graphs:** outputs identical on all 15 public samples;
  total-time changes below 1%. Kept opt-in with `DRYFT_PREFILL_GRAPH=on`.
- **Reference W8A8 on every projection:** passed five public-2 and five
  long-context prompts; worst gaps 0.375 and 0.25. This established arithmetic
  feasibility only: the untuned reference kernel is slower than the engine.
- **Two-component activation residual correction:** the full-model reference
  failed at 4.5 logits despite being closer arithmetically. The native corrected
  MLP passed the isolation prompts but cost substantially more than the original.
- **Native W8A8 gate/up and LM head together:** +3.6% public-2 in a fresh-process
  paired benchmark, but a 3.8125-logit failure. Not shipped.
- **Native W8A8 gate/up at batches 24/32:** replacing the BF16 MLP weights and
  quantizing activations failed at 3.125 logits. Not shipped.
- **Native W8A8 LM head alone at wider batches:** about 1.2--1.3% end-to-end
  gain, but a long technical-text sample reached exactly 2.0 logits. It remains
  disabled; that headroom does not justify the small gain.

The 60-trial prose/code/technical study of the narrowed MLP-or-head policy
checked 115,560 emitted positions. Every trial passed, but the exact-limit
LM-head result is why the final default keeps only the MLP path. For the enabled
MLP shapes, corpus worst gaps are 1.75 at batch 16 and 0.625 at batch 12.
This is finite validation, not a universal quantization-error bound.

Reproduction: `bench/modal_next.py` runs graph, quant, micro, native, fused,
isolate and ablation studies; its baseline explicitly disables new features.
Use `--shape-set native` for batches 12/16/24/32 with longer continuations.
`bench/modal_bench.py::compare --baseline-fp8 on --candidate-int8-mma on`
compares against the actual current INT8 baseline, not BF16. The runtime is
pinned, and H200 substitutions are rejected. All 40 CPU tests and local Dryft
archive validation pass.

Final fresh-process paired comparison (five samples, INT8 enabled in both
engines), `paired-native-int8-final-20260920.json`:

| Shape | Baseline tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| public-0 | 339.5 | 339.5 | approximately flat |
| public-1 | 573.9 | 571.2 | -0.5% |
| public-2 | 3350.3 | 3502.8 | +4.6% |
| 1 x 4096 -> 65 | 255.2 | 252.7 | -1.0% |

All baseline and candidate samples pass the local gates. Only public-2 enters
the new path; the other differences are within observed tuning/timing noise.
### Official result

Commit `0bd7995`, submission `9a16e578-c141-49d3-a274-967abd9a715b`, official
run `80693a1f-067f-4cb5-a744-f3f5ed69f9ee`: **1177.2915 tok/s, ranked**, all
workloads passed. This is +15.7 tok/s (+1.35%) over the user's reported best
1161.6. Official public throughputs were 319.2 / 572.5 / 3542.4 tok/s.
The run completed in about 7 minutes 18 seconds. The score change is one
official observation; the paired local studies isolate the MLP's speedup.
`bench/results/official-native-int8-20260920.json` retains the result.

Direct CLI archive upload now returns HTTP 405; the documented repository-push
submission path is required. The CLI also needs `DRYFT_API=https://htn.dryft.ai`
in this installation. No credentials are stored in the repository.

### Follow-up: pre-permuted BF16 activations (prototype only)

`bench/packed_bf16_probe.py` moves activation permutation out of each W8A16
GEMM block. It checks 42 projection/SwiGLU cases bit-exact against the existing
kernel at matching weights and reductions. Direct L1 loads help the batch-one
MLP; a simple shared-memory copy of already-permuted activations is better at
larger batches. The original fragment arithmetic and weight quantization stay.

The promising kernel-only comparisons are gate/up at batch 1 (24.53 -> 22.42
us), QKV at batch 16 (13.36 -> 11.74 us), and down at batch 16 (20.90 -> 17.77
us). These **exclude the cost of packing the activation**. An end-to-end win
requires producing the layout in the preceding norm/SwiGLU/attention kernel
and preserving split-K consumer fusion. Batch-32 comparisons can also change
the incumbent's BF16 weight family to INT8, so they do not isolate layout and
would need independent accuracy validation. No production changes or score
claim are made for this prototype. Results: `prepacked-bf16-20260920.json`.

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

### The 15-minute run budget, and what warmup cost

The first three official runs of this engine did not finish: two exceeded
the platform's 15-minute whole-run limit and the third was cancelled with
the public samples passing at 318 / 578 / 3426 tok/s. Load plus warmup had
grown from about 30 s to 61 to 97 s per workload, and a run has nine of them.

Stage timers in `_ensure` found it: the 8-bit kernel compiled all 48 of its
variants (12 to 23 s) where a process needs 12; the quantiser's scale search
was 1.7 million small kernel launches; nine skinny Triton configurations were
compiled per projection only to lose to a kernel moving half their bytes;
the attention race compiled eight more; the self-check ran a full race at
batch 2; the verification widths raced again at batch 1.

Now: one compile per activation tile count, the scale from the block maximum,
the bfloat16 candidates skipped when 8-bit beats cuBLAS by 10%, the attention
race opt-in again, the self-check racing one 8-bit configuration, and the
verification widths reusing ordinary decode's choice. Load plus warmup is
**29 / 33 / 32 s** at batches 1 / 4 / 16, throughput unchanged
(330 / 584 / 3385 tok/s on one sample each).

### Two wide-shape ideas, measured and rejected, September 20

The hidden score implies wide, long workloads (see `score x metricMs`, below),
so the two levers aimed there went first. Both lost.

**Warps sharing a row block in the 8-bit GEMM**, for batch 32 and up, where the
single-warp layout holds four tiles of B fragments and falls to 0.86x cuBLAS.
Two or four warps per row block, each with one or two tiles, were slower on
every shape at batches 32 and 64 (22.5 vs 17.8 us on qkv at 32): the warps'
duplicate weight loads both miss L2, and staging the wider activation per
block costs more than the register relief buys. Batch 32 stays with cuBLAS;
the machinery stays in `cuda_fp8.py` off the race.

**INT8 keys and values**, one fp32 scale per token and head, dequantised in the
attention kernel, with a bfloat16 copy for prefill's SDPA. Slower on every
public shape (-2.1 to -4.4%) and outside the margin on one 16 x 512 sample:

| shape | bf16 cache | INT8 cache | worst gap bf16 -> INT8 |
| --- | ---: | ---: | --- |
| 1 x 512 -> 32 | 323.8 | 309.7 | 0.125 -> 0.125 |
| 4 x 2048 -> 32 | 571.4 | 557.2 | 0.25 -> 0.50 |
| 16 x 512 -> 128 | 3387.4 | 3287.7 | 1.375 -> **2.75** |
| 1 x 4096 -> 65 | 242.4 | 237.2 | 0.125 -> 0.125 |

The attention kernel was already near bandwidth, so halving its bytes could
save at most 0.25 ms of a 3.5 ms step, and the per-tile dequantisation plus
the copy cost more than that. The gap comes from Qwen's keys having a few
large channels after RoPE, which a per-token scale cannot resolve; per-channel
key scaling would fix the accuracy and not the speed. Off by default
(`DRYFT_KV=int8` to experiment); the GPU checks for it stay.

### Fusing the residual add and RMSNorm into the producing GEMM, rejected

The idea: the o and down GEMMs write no planes; all warps of a block share
sixteen output rows and split K, reduce through shared memory, and the last
block (atomic ticket) adds the residual and normalises the batch. 72 launches
fewer per step. `cuda_fp8.matmul_add_norm` implements it and matches the
two-launch pair on 60 GPU cases.

Bisected on the o projection (`bench/fused_probe.py`):

| batch | planes GEMM | planes + add-norm kernel | fused GEMM only | + ticket | + tail |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.0 us | 11.9 us | 9.0 | 10.0 | 16.7 |
| 4 | 8.4 | 12.3 | 9.1 | 10.3 | 32.3 |
| 16 | 9.3 | 13.4 | 15.8 | 16.7 | 84.8 |

The tail costs about 4 us per batch row: one block of 256 threads doing the
add-norm for 40,960 elements is latency-bound on dependent loads and stores,
where the separate kernel spends 2.5 us launching and then runs on sixteen
SMs. Staging the activation per warp was slower than reading B fragments
from L2, and the direct reads cost 6 us at batch 16. Off by default
(`DRYFT_FUSED_ADD_NORM=on` races it).

What would work is two-phase: the producer writes the residual and a partial
sum of squares per block, and the *consuming* GEMM's staging pass finishes
the norm while it permutes the activation. No serial tail, no planes. The
ceiling is the add-norm kernel's 2.5 to 3.7 us times 72, about 7% at batch
1 and 3% at batch 16, for a day of kernel work on the producer's staging.

### What the platform's aggregate says about the hidden shapes

The run result carries `score` and `metricMs`. Their product is 506.5 in every
run since the first: the score is the geometric mean of tokens per sample over
the geometric mean of median milliseconds, so the six hidden workloads average
about 2^9 = 512 tokens per sample against 203 for the public three. They are
wide or long, which is why the hidden score runs 36% above the public geomean.

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
