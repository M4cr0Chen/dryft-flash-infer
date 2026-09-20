# Local H100 measurements

These are development benchmarks. The score on Dryft uses six hidden workloads;
the three public shapes here do not predict that score. The published website
now describes official-only submissions, despite the older starter workflow.

The Modal image pins the challenge's Torch, Triton and Transformers versions.
The existing `dryft-qwen3-4b` volume supplies the checkpoint. Timing entrypoints
reject an H200 substitution instead of treating its results as H100 results.

## Compare two engines

Point `DRYFT_BASELINE_DIR` at a separate directory containing the old `engine.py`
and its kernels, then run:

```sh
DRYFT_BASELINE_DIR=/path/to/old/engine .venv/bin/modal run \
  bench/modal_bench.py::compare --samples 5 \
  --output bench/results/paired.json
```

Add `--long-context` to include a 4096-token prompt / 65-token output case.

The baseline uses BF16 with speculation disabled. The candidate defaults to
FP8 weight compression and capped two-token speculation at batch one. Pass
`--candidate-fp8 off --short-draft 0` for a BF16-only comparison. Both engines
run on the same physical H100 and get identical corpus prompts. Old/new order
alternates between workloads. Each engine/workload runs
in a fresh subprocess. The JSON records source hashes, GPU/runtime, each sample's
time and correctness gap, and medians. For differences near the observed noise,
repeat the paired experiment rather than infer a ceiling from one result.

The local clock is in-process, and the reference model shares that process with
the candidate. The official judge uses separate trusted processes and a token
pipe. Local latency/memory checks are diagnostics, not official eligibility.
Every measured sample is replayed through native Qwen on its own emitted prefix.

The comparison checks the actual fused Q/K/V Triton kernel against the
separate kernels, including untouched cache slots and nonzero batch offsets.
It also checks attention boundaries and, when enabled, FP8 projections against
independently dequantized weights across multiple group sizes and row counts.
After timing, a fixed-position graph is replayed 65 times to check that its
position and emitted token remain stable.

## Locate the remaining cost

```sh
.venv/bin/modal run bench/modal_bench.py::trace --batch 1 --context 512
.venv/bin/modal run bench/modal_bench.py::trace_prefill --batch 4 --context 2048
.venv/bin/modal run bench/modal_bench.py::profile --shape public-1
```

`trace`, `micro`, and `utilisation` use `FixedDecodeReplay`. Each replay restores
the input token and position, so it neither grows the context nor overruns the
cache. The two restoration copies are included in microbenchmark timing.
`profile` measures an actual bounded decode sequence without restoration.

The older `trace` and `utilisation` advanced positions beyond their allocated
cache/RoPE tables. Their historical timings cannot establish a performance
limit. Likewise, eager Python-launch races can rank kernels differently from
the CUDA graph that executes them in production.

## MLP, attention, and short verification studies

```sh
.venv/bin/modal run bench/modal_bench.py::study --stage profile \
  --output bench/results/mlp-profile.json
.venv/bin/modal run bench/modal_bench.py::study --stage attention \
  --output bench/results/attention-study.json
.venv/bin/modal run bench/modal_bench.py::study --stage attention_paired \
  --output bench/results/attention-controlled.json
.venv/bin/modal run bench/modal_bench.py::study --stage speculation \
  --output bench/results/short-speculation.json
```

The MLP profile attributes eager prefill's GPU time to each projection and
times the selected decode projections while sweeping all 36 weights in a
graph. The profiler event table includes nested CPU scopes with attributed
GPU time: do not sum that table. The projection totals are disjoint, while
the separate full-stage wall times include the rest of execution.

The attention study compares 30 split/block/warp configurations at both ends
of generation, then compares the best candidate inside the complete decode
graph in alternating order. A one-partition candidate writes the normalized
output directly, avoiding the merge. The separate paired benchmark checks
actual generated tokens teacher-forced; equal microbenchmark outputs alone
do not establish whole-model correctness. `check_attention_dispatch` also
checks 54 SDPA comparisons with causal boundaries, empty partitions, a partial
last block, and NaNs beyond the valid cache.

`attention_paired` holds the engine's projection choices fixed, captures an
ordinary and a tuned graph, and alternates them per prompt. The default engine
keeps the original dispatch because measured total improvements were small;
set `DRYFT_ATTENTION_TUNE=on` to evaluate the optional warmup tuner.

The speculation study imports the shipping `engine/kernels/speculation.py`.
It captures separately tuned
two- and three-row verification graphs, retaining ordinary decode for empty
proposals. Every accepted draft token must equal the target prediction on its
own prefix; rejection emits the target's correction. The study compares four
policies with ordinary decode on five prompts each from prose, Qwen source
code, and the implementation guide. It tests batch one at 512/32, 2048/32, and
512/128 prompt/output lengths, rotating trial order and teacher-forcing every
sample. This is a generalization check, not the judge's hidden corpus. The
attention sweep and speculation study explicitly disable attention autotuning
and automatic short speculation, so each study controls its own variants.
Quantization uses the engine default; the current speculation study therefore
measures its additional benefit over ordinary FP8 decode.

The verifying graphs retain the ordinary graph's weight choices. Where a BF16
GEMM is faster, it reads reconstructed quantized weights rather than switching
back to the original weights. Mixing those choices previously caused a large
teacher-forced failure on a long prose continuation; `speculation_debug`
rechecks the 512/128 corpus cases. Replay failures are retained in the JSON;
inspect `passes` on every sample and the per-policy spread, not just the
process exit status.

`study --stage fp8` compares expanded-weight and postscaled-group FP8 kernels
across all four projections at batches 2, 3, 4, and 16. `study --stage spec_trace`
traces the historical failing prefix and compares graph/eager execution and
individual precision changes. Neither experiment is part of the submitted
engine.

To check longer contexts and batch sizes outside the three public examples:

```sh
.venv/bin/modal run bench/modal_bench.py::main --shape-set coverage \
  --samples 5 --corpus --output bench/results/coverage.json
```

These development cases cover batch/prompt/output shapes 1/4096/65,
3/257/33, 8/1024/64, and 32/256/16. They do not represent the hidden workloads.
