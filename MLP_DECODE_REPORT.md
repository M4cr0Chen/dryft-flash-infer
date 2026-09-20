# Prefill MLP fusion and decode tuning — September 20

The experiments start from `36e856c`, where Triton attention and RoPE are outside
the programmatic launch chain. This work does not enable Triton PDL. Some test
snapshots included a concurrent experimental wide-batch BF16 gate/up guard;
that guard was subsequently removed by the other work. The batch-32 coverage
row below therefore validates the additions with the tested projection choices,
not all possible choices of the current wide-batch quantization race. The saved
source hashes distinguish snapshots. This work does not change that policy.

## Implemented changes

`engine/kernels/prefill_mlp.py` implements a BF16 gate/up GEMM with a SwiGLU
epilogue. Weights are rearranged once during engine construction to a contiguous
`[hidden, interleaved gate/up]` layout. Projection output, SiLU output, and the
final product retain their separate native BF16 rounding boundaries.

The winning implementation uses a 128 × 256 × 64 tile, eight warps, and three
or four pipeline stages. The original row-major implementation lost to cuBLAS;
transposing the packed weights was necessary. Component comparisons on real
weights at 512 and 8192 rows were bit-exact against BF16 linear + SwiGLU.

Warmup compares both stage counts with cuBLAS plus SwiGLU and requires a 3% win,
then repeats the winning comparison after compilation. Prefills below 1024 rows
keep cuBLAS. Other shapes also keep it if fusion does not win. The 12,288-row
coverage case selected cuBLAS. Every full and remainder batch chunk is prepared
before generation; measured samples do not tune or compile a new shape. The
extra BF16 weight layout is approximately 3.59 GB.

`engine/kernels/decode_tuning.py` compares a bounded set of attention-output
projection schedules inside the **complete ordinary decode graph**. It holds
weight values, attention, and every other projection fixed. The trials restore
their position/token, rotate candidate order, and require a 1% step improvement
with six of seven paired trials favoring the change. The final production graph
is recaptured without the benchmark's restoration copies. The race runs only
during warmup, at batches up to 16, on an existing INT8-weight output projection;
native activation-quantized MLP paths retain their original schedule.

The common change reduces the output projection from 16 split-K planes to eight,
using either the ring kernel or the register kernel depending on batch and the
complete-graph measurement. The quantized weights themselves do not change.

## Controlled results

Five paired samples per shape and corpus, one physical H100 per run. Prose,
Qwen implementation source, and technical documentation provide three corpora.
Both variants use the same original projection choices, except the candidate's
explicit output-projection schedule change. Generation timing precedes native
replay; variant order alternates. Batch-one speculation remains enabled.

| Batch / prompt / output | Throughput change: prose | Code | Technical | Worst candidate replay gap |
| --- | ---: | ---: | ---: | ---: |
| 1 / 512 / 32 | +0.88% | +0.94% | +0.88% | 0.375 |
| 4 / 2048 / 32 | +1.46% | +2.86% | +1.98% | 0.500 |
| 16 / 512 / 128 | +1.95% | +1.72% | +1.67% | 0.875 |
| 1 / 4096 / 65 | +3.46% | +1.68% | +0.59% | 0.125 |
| 3 / 257 / 33 | +1.51% | +1.39% | +1.59% | **2.000** |
| 8 / 1024 / 64 | +1.35% | +1.47% | +1.17% | 0.750 |
| 12 / 1024 / 64 | +0.57% | +0.40% | +0.53% | 0.750 |
| 32 / 256 / 32 | +0.90% | +1.31% | +1.05% | 0.875 |

All 120 candidate generations passed teacher-forced native replay. The exact
2.0-logit sample at batch three also reached 2.0 in the baseline. Maximum
candidate spread within a shape/corpus set was 12.3%, below 25%. This is finite
local coverage, not a guarantee about hidden prompts or the official score.

These figures compare medians within each corpus. Pooling unrelated corpora
can produce a different median: batch one's pooled 15-sample median regresses
2.1%, even though all three within-corpus medians improve. Thirteen of its 15
paired samples improve; one technical sample slows 4.6%. Changed rounding can
change a near-tie and subsequent n-gram acceptance. Retain this qualification
when interpreting the roughly 1% batch-one benefit.

Sources:
- `bench/results/mlp-decode-combined-public-20260920.json`
- `bench/results/mlp-decode-combined-coverage-20260920.json`

Prefill alone passed 45 public-shape samples across the same corpora, with
approximately +1.1% public-1 and +0.5% public-2 aggregate throughput; public-0
retained cuBLAS and was within noise. Raw results are in
`bench/results/prefill-fusion-production-public-20260920.json`.

## Fresh-process confirmation

The final comparison used a fresh engine process per variant/workload on one
H100, five samples each. Both variants used the same source and existing INT8,
CUDA PDL, and speculative settings; the two new feature switches were enabled
only for the candidate. This exercises the actual construction/warmup hooks.

| Workload | Baseline tok/s | Candidate tok/s | Gain |
| --- | ---: | ---: | ---: |
| public-0 | 362.6 | 368.9 | +1.74% |
| public-1 | 614.6 | 625.3 | +1.75% |
| public-2 | 3851.8 | 3910.7 | +1.53% |
| 1 / 4096 / 65 | 281.8 | 283.0 | +0.43% |

All candidate and baseline samples passed the local correctness, latency,
memory, and spread checks. The candidate's largest replay gap was 1.375 and
largest spread 7.3%. These local clocks are not the official process/pipe clock.
No official submission or leaderboard score is claimed.

Source: `bench/results/mlp-decode-fresh-paired-20260920.json`; its adjacent log
preserves selections and runtime checks. `mlp-decode-tested-source-20260920.tar.gz`
preserves the exact tested source before the defaults were enabled. Finalization
also skips output tuning on native activation-quantized MLP paths, which were
not selected in these comparisons.

Validation after finalization: all 43 existing tests passed, and `bin/dryft
validate engine` accepted the source archive (about 69 KB compressed).

## Rejected changes and interpretation

The first fused prefill layout was about 5–10% slower than cuBLAS plus SwiGLU.
The transposed-layout component win did not translate directly to total
generation; full-model gains are smaller and are the numbers above.

An isolated projection-plus-consumer tuner selected a QKV configuration that
looked about 2% faster locally but slowed the complete batch-one step from
2.565 to 2.783 ms, and generation by 6.6%. It was rejected. The accepted tuner
therefore evaluates complete graphs and targets only the output projection.
See `decode-consumer-tuning-20260920.json` and
`decode-graph-tuning-20260920.json` under `bench/results/`.

One earlier batch-32 tuning run failed replay at 4.9375 logits in both baseline
and candidate. Later runs with other warmup-selected projection configurations
passed. That failure remains recorded; it is evidence that the existing
quantization/race policy still needs broad validation. These changes introduce
no new weight or activation quantization.

The measured gain is roughly 1–3%, not enough by itself to close the previously
reported leaderboard gap. Further substantial decode gains require reducing
projection execution cost or weight traffic without increasing numerical error.
Another isolated kernel race or re-enabling Triton PDL is not supported by these
measurements.

## Reproduction and controls

The engine switches are `DRYFT_PREFILL_MLP` and `DRYFT_DECODE_OUT_TUNE`, both
enabled by default after the component, corpus, and fresh-process comparisons.
Setting either to `off` disables that addition independently. The development
driver preserves frozen engine source hashes in its result files:

```sh
.venv/bin/modal run bench/mlp_decode_modal.py --stage combined --shapes public \
  --samples 5 --output bench/results/mlp-decode-repeat.json
```

The complete initial layout/configuration sweeps remain in
`prefill-fusion-micro-20260920.json` and `prefill-fusion-layout-20260920.json`.
Experiments and their source stay outside `engine/`.
