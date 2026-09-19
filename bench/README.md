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

Both engines use BF16, run on the same physical H100, and get identical corpus
prompts. Old/new order alternates between workloads. Each engine/workload runs
in a fresh subprocess. The JSON records source hashes, GPU/runtime, each sample's
time and correctness gap, and medians. For differences near the observed noise,
repeat the paired experiment rather than infer a ceiling from one result.

The local clock is in-process, and the reference model shares that process with
the candidate. The official judge uses separate trusted processes and a token
pipe. Local latency/memory checks are diagnostics, not official eligibility.
Every measured sample is replayed through native Qwen on its own emitted prefix.

The comparison also checks the actual fused Q/K/V Triton kernel against the
separate kernels, including untouched cache slots and nonzero batch offsets.
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
