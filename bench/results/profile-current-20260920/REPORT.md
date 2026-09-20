# Current engine H100 profile — September 20

Profiled the working engine at `36e856c`, with Triton programmatic launch off.
CUDA GEMMs and small CUDA kernels retain PDL; Triton attention and RoPE do not
join the programmatic chain. The dormant late-attention-trigger setting does
not enable it.
The exact source snapshot, per-file SHA256 hashes, and Git diff are saved beside
this report. No production engine files were changed by this investigation.

Five fresh engine processes ran sequentially on one NVIDIA H100 80GB HBM3,
using Torch 2.5.1 and the existing pinned benchmark image/checkpoint. Each shape
had one warmup and five prose-corpus generation samples. All 25 samples passed
native teacher-forced replay; public-2 reached exactly the 2.0-logit limit.

## Wall-clock measurements

All times are milliseconds, medians. Generation throughput includes prefill.

| Shape (batch / prompt / output) | Standalone prefill | Ordinary decode step | Stream TTFT | Stream TPOT | Stream tok/s | Worst replay gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 512 / 32 | 9.455 | 2.536 | 9.874 | 2.593 | 355.1 | 0.000 |
| 4 / 2048 / 32 | 121.048 | 2.922 | 115.672 | 3.060 | 609.2 | 0.125 |
| 16 / 512 / 128 | 109.475 | 3.277 | 103.709 | 3.382 | 3840.5 | **2.000** |
| 32 / 256 / 32 | 109.535 | 4.549 | 102.141 | 4.745 | 4104.4 | 0.250 |
| 1 / 4096 / 65 | 71.094 | 2.722 | 67.282 | 2.666 | 273.0 | 0.125 |

Stream measurements retain production speculation at batch one. Ordinary decode
is separately captured at the first post-prefill position and includes two
state-restoration copies. It does not measure speculative throughput or growing
context. Standalone prefill and stream TTFT were measured in separate phases;
their different thermal/clock and host conditions mean they should not be
combined into a reconstructed generation total. Profiling was outside all these
wall measurements. Local stream spread was 11.67%, 0.94%, 0.80%, 0.71%, and
5.35%, respectively, using (max-min)/median.

## Prefill attribution

GPU milliseconds per prefill. Projection columns use disjoint tagged `F.linear`
calls; attention and pointwise columns use CUDA kernel events.

| Shape | QKV | Output projection | Gate/up | Down | Flash attention | SwiGLU | Residual/norm | QKV norm/RoPE/cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public-0 | 1.086 | 0.948 | 2.972 | 1.917 | 0.774 | 0.310 | 0.298 | 0.269 |
| public-1 | 12.876 | 9.115 | 41.933 | 20.431 | 20.000 | 5.736 | 3.841 | 3.606 |
| public-2 | 12.976 | 9.055 | 42.281 | 20.595 | 7.601 | 5.751 | 3.844 | 3.577 |
| wide | 12.962 | 9.007 | 42.250 | 20.589 | 5.545 | 5.703 | 3.851 | 3.569 |
| long | 6.401 | 4.917 | 20.053 | 10.667 | 19.494 | 2.840 | 1.843 | 1.821 |

The LM head contributes another 0.25–0.29 ms. Larger prefill GEMMs sustain
approximately 700–710 TFLOP/s, computed from dense transformer projection FLOPs
and GEMM kernel time. This is good utilization, not peak utilization; replacing
cuBLAS must preserve it. The MLP is the largest prefill component. At context
4096, attention is also substantial: 19.49 ms of 71.09 ms standalone prefill.

## Decode and the bandwidth floor

The engine calls its INT8 weight family `fp8`; the label does not mean E4M3.
All four smaller-batch cases chose INT8 weights for every projection. Batch 32
chose BF16 QKV and gate/up, with INT8 output, down, and LM head. Native W8A8 MLP
was not selected in this run, including batch 16. Selected runner configurations
and split-K counts are preserved in `results.json` and `run.log`.

Assuming each selected weight and each live BF16 KV element is read once:

| Shape | Selected weight bytes, GB | Ideal weight+KV time at 3.35 TB/s, ms | Measured ordinary step, ms | Useful-byte rate, TB/s |
| --- | ---: | ---: | ---: | ---: |
| public-0 | 4.148 | 1.261 | 2.536 | 1.666 |
| public-1 | 4.148 | 1.599 | 2.922 | 1.833 |
| public-2 | 4.148 | 1.599 | 3.277 | 1.635 |
| wide | 6.434 | 2.282 | 4.549 | 1.681 |
| long | 4.148 | 1.419 | 2.722 | 1.746 |

These are model-based optimistic floors and useful-byte rates, not measured HBM
traffic or memory-controller utilization. They omit intermediate/activation
traffic, redundant reads, and dependencies. Peak bandwidth is from
[NVIDIA's H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

PDL makes summed decode kernel durations misleading:

| Shape | Summed kernel durations, ms | Union of kernel intervals, ms | Unprofiled step, ms |
| --- | ---: | ---: | ---: |
| public-0 | 5.287 | 2.500 | 2.536 |
| public-1 | 5.393 | 2.889 | 2.922 |
| public-2 | 5.877 | 3.245 | 3.277 |
| wide | 5.608 | 4.524 | 4.549 |
| long | 5.631 | 2.675 | 2.722 |

The union and unprofiled wall are separate observations, so their difference is
not a precise idle-time measurement. The trace nevertheless shows substantial
overlap, and little reason to expect large wins from merely filling launch gaps.
In particular, the 0.7–0.9 ms attributed to RMSNorm includes PDL waiting; removing
norm cannot be assumed to save that duration. No instruction-level stall,
occupancy, or DRAM hardware counters were collected in this profile.

Batch-32 gate/up executes 36 `_skinny_kernel` calls totaling 1.376 ms, plus
0.056 ms SwiGLU. Its QKV cuBLAS calls total 0.577 ms. These are useful targets
for complete-stage experiments, not additive predictions of removable wall time.

## Priorities supported by this run

1. **Prefill MLP:** gate/up plus down costs 62–63 ms on the 8192-row shapes.
   Try BF16 GEMM/SwiGLU fusion without sacrificing matrix throughput; independently
   test final-layer prefill pruning. A 20% reduction of public-1's measured stream
   TTFT would yield approximately 12.4% more total throughput.
2. **Complete decode-stage tuning:** include plane reduction, residual/norm,
   SwiGLU, and PDL interactions when choosing projection configurations. Useful
   bytes are serviced at roughly half nominal peak, despite overlapping launches.
3. **Wider-batch GEMMs:** batch 32 streams 6.43 GB rather than 4.15 GB due to
   BF16 fallbacks. Improving INT8-weight/BF16-activation execution could lower this
   traffic, but changing the selected weight family requires new replay coverage.
4. **Long-context attention:** prefill attention alone costs 19.49 ms at L4096.
   Decode partition counts were 4 / 4 / 1 / 1 / 16 for the five shapes.

The exact-limit public-2 replay result argues for preserving arithmetic while
testing fusion/scheduling first. This profile does not establish that any proposed
change wins, or predict a hidden leaderboard score.

## Artifacts and reproduction

- `results.json`: raw samples, kernel tables, selected families/configurations.
- `metadata.json`, `engine-snapshot/`: immutable source and hashes.
- `*-prefill.json.gz`, `*-decode.json.gz`: ten Chrome/Perfetto CUDA traces;
  decompress to JSON before loading if the viewer does not accept gzip.
- `run.log`: engine tuning decisions and Modal execution log.

```sh
.venv/bin/modal run bench/profile_current.py --output bench/results/profile-next
```

The profiler uses the existing checkpoint volume and creates a separate source
snapshot. It does not submit an official run. Every timing sweep precedes native
replay, avoiding reference forwards warming only one measured variant. These
five samples per shape cover one prose corpus, not broad numerical validation.
