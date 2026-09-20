# Feature-removal experiments — September 20

Two snapshots were tested: the ranked **1177.3 tok/s engine, `0bd7995`**, and
the subsequent tiled-weight/ring implementation, **`43822c0`**. The snapshots
were copied from Git into `/tmp`; the other agent's working files and running
experiments were not modified or paused.

**Most of the existing complexity pays for itself. The strongest removal is
splitting ordinary decode residual/reduction from RMSNorm, combined with one
attention partition at batch 16.** On the ring version, this combination
improved full-generation throughput by **2.8–3.0%** across prose, code, and
technical text, with every replay passing. This is a local result; no new
official score is claimed for these ablations.

## Findings worth acting on

### 1. Residual/RMSNorm fusion can be slower in decode

The alternative keeps the reduction and BF16 residual addition in one small
pointwise Triton kernel, then invokes the existing RMSNorm. Prefill is unchanged.
Native INT8's combined normalization/activation packing is retained wherever
the engine selects it. All reference BF16 cast boundaries are preserved.

On `43822c0`, separating these operations measured:

| Shape | Throughput improvement | Evidence |
| --- | ---: | --- |
| 1 × 512 → 32 | +0.67% | Seven paired samples; all seven favored separation |
| 4 × 2048 → 32 | +1.78% | Five paired samples; TPOT 3.374 → 3.290 ms |
| 16 × 512 → 128 | +1.9–2.3% | Five samples each of prose, code, and technical text |

The likely mechanism is better parallelism for the plane reduction/residual
work, rather than asking the row-wise RMSNorm kernel to do all of it. This is
an interpretation of the measurements, not an instruction-level proof.

Sources: [batch 1](bench/results/ablation-ring-norm-public0-20260920.json),
[batch 4](bench/results/ablation-ring-norm-public1-20260920.json),
[three-corpus batch 16](bench/results/ablation-ring-combined-20260920.json).

**Correction to the first screening pass:** its fully separate implementation
used a Python loop of Torch additions to reduce split-K planes. That exaggerated
the cost of removing fusion. The subsequent implementation uses one pointwise
reduction/residual kernel. Do not use the large residual-normalization losses
in the four initial screening JSONs as evidence for the final conclusion.

### 2. Attention is over-partitioned at batch 16

The original heuristic doubles partitions until `batch × KV heads × partitions`
reaches 132. Batch 16 already supplies 128 programs, but the rule doubles it
to 256 and adds a merge kernel. One partition avoids that merge.

On `0bd7995`, the gain was +1.1% in screening and +1.3–1.6% in the five-sample
confirmation. On `43822c0`, it was approximately +0.8–1.3%.

This is shape-dependent. Removing all partitions slowed batch 1 by 3% and
batch 4 / context 2048 by 8%. Lowering the program target to 128 was essentially
flat at batch 4 and batch 8 in full-generation confirmation. At batch 1 /
context 4096, halving the partitions improved ordinary generation by 1.4–2.4%.
Those long-context attention measurements had speculation disabled on both
sides; they are not comparisons against the shipping speculative stream.

Sources: [batch 16 confirmation](bench/results/ablation-confirm-public2-20260920.json),
[batch 4](bench/results/ablation-confirm-public1-20260920.json),
[batch 8](bench/results/ablation-confirm-medium-20260920.json),
[long context](bench/results/ablation-governor-context-20260920.json).

### 3. The two layout changes combine successfully

Baseline and candidate used the same ring engine, with projection selections
held fixed. Five samples per corpus at 16 × 512 → 128:

| Corpus | Baseline tok/s | Separate decode norm + lower attention split count | Gain | Worst replay gap |
| --- | ---: | ---: | ---: | ---: |
| Prose | 3510.9 | 3609.0 | +2.79% | 1.375 |
| Code | 3504.0 | 3607.6 | +2.96% | 0.500 |
| Technical | 3504.0 | 3604.7 | +2.87% | 1.125 |

Maximum candidate spread across these sets was about 1.2%. All 15 candidate
generations passed native teacher-forced replay. Implementations are available
as the `separate_decode_norm` and `attention_target128` variants in
[ablation_study.py](bench/ablation_study.py).

Source: [combined layout results](bench/results/ablation-ring-layout-combined-20260920.json).

### 4. The speculation governor is conservative, but necessary

Removing the governor improved long code/technical throughput by approximately
35–39% versus ordinary decoding. However, spread reached **44–47%**, failing
the 25% limit. Even the prose set reached 26.1%. Do not remove it wholesale.

Increasing the target from 1.15 to 1.20 or 1.25 is more promising for longer
outputs. On the ring engine, batch 1 / prompt 512 / output 128:

| Corpus | Target 1.20 gain over shipping | Target 1.25 gain | Spread at 1.20 / 1.25 |
| --- | ---: | ---: | ---: |
| Prose | approximately flat | approximately flat | 13.3% / 17.0% |
| Code | +2.28% | +5.22% | 10.2% / 14.0% |
| Technical | +3.75% | +7.99% | 7.9% / 10.7% |

All replays passed, worst gap 0.5. These governor variants share the same
verification graphs, avoiding a separate tuning race as a confound.

For 32-token outputs, increasing the target gave essentially no median gain
and increased spread to about 20%. At prompt 4096 / output 65, target 1.20
gave +2% on prose/code and -0.8% on technical text. A longer-output policy at
1.20 is the conservative candidate; 1.25 spends more timing headroom. The
measurements do not establish a universally safe spread bound.

Sources: [uncapped](bench/results/ablation-speculation-20260920.json),
[short outputs](bench/results/ablation-governor-short-20260920.json),
[long outputs, ranked engine](bench/results/ablation-governor-long-20260920.json),
[long outputs, ring engine](bench/results/ablation-ring-governor-20260920.json),
[long context](bench/results/ablation-governor-context-20260920.json).

## A faster tuning choice that failed correctness

The ring-version projection race can displace the one-launch native INT8 MLP
with a ring projection, plane reduction, and separate SwiGLU. On the first
five public prose samples:

- Removing ring candidates globally: +1.60%.
- Removing them only for gate/up selection: +1.51%.
- Forcing the earlier native INT8 MLP: +1.67%.

However, broader replay rejected forced native INT8: **4.75 logits on code**
and **4.6875 on technical text**. Combining that override with the two layout
changes reached **14.46875 logits**. These configurations must not ship.

This supports investigating a producer/consumer-aware tuner, but does not
support blindly restoring an earlier quantized MLP choice. The passing layout
combination above preserves the ring engine's normal MLP selection. A separate
race allowing at most eight down-projection planes did not beat the incumbent
by its 2% threshold; it retained the sixteen-plane configuration.

Sources: [tuner removals](bench/results/ablation-ring-tuner-20260920.json),
[broader rejection](bench/results/ablation-ring-combined-20260920.json).

## What should remain enabled

Initial screening on `0bd7995`: three paired samples per shape. Numbers are
throughput changes when the feature is removed, relative to ordinary decode
with its other features held fixed. Negative means removal is slower.

| Removed/replaced feature | B1 / 512 / 32 | B4 / 2048 / 32 | B16 / 512 / 128 | B32 / 256 / 32 |
| --- | ---: | ---: | ---: | ---: |
| CUDA decode graph | -72.2% | -52.4% | -63.9% | -42.4% |
| INT8 weight quantization | -23.0% | -14.0% | -14.9% | -0.3%* |
| Custom decode attention → SDPA | -24.6% | -30.5% | -18.6% | -8.8% |
| Combined QKV projection | -13.8% | -7.4% | -12.0% | -6.4% |
| Fused projection + SwiGLU | -2.2% | -1.1% | -1.6% | -0.5% |
| Also separate SiLU and multiply | -4.2% | -7.0% | -5.8% | -6.6% |
| Split-K planes consumed directly | -2.1% | -1.9% | -1.6% | -0.3% |
| Fused Q/K normalization + RoPE/cache | -4.0% | -3.1% | -3.0% | -1.7% |
| Per-projection race → fixed INT8 configs | -1.2% | -1.1% | -3.0% | -0.3% |
| Host stream overlap | -0.3% | -0.3% | -0.7% | -0.4% |

*The batch-32 incumbent selected BF16 for all five projection families, so
removing weight quantization was effectively a no-op there.

On batch 16, removing native INT8 MLP lost 4.47%; removing its fused activation
packing lost 1.59%. These figures describe the ranked engine, not the later
ring version's different projection choices.

Splitting QKV also removes its fused partial-result handoff, and splitting the
MLP epilogue has dependent implementation changes. These rows are complete
operation alternatives, not additive estimates of individual instruction costs.
The static cache is a prerequisite for the fixed-address graph and attention
implementation; it was not independently replaced by a dynamic cache.

Sources: [B1](bench/results/ablation-public0-20260920.json),
[B4](bench/results/ablation-public1-20260920.json),
[B16](bench/results/ablation-public2-20260920.json),
[B32](bench/results/ablation-wide-20260920.json).

## Flat results and inactive experiments

- Replacing the pinned prompt upload with `torch.tensor(..., device='cuda')`
  did not help.
- Removing the unused decode-mask update was within noise (roughly ±1%,
  usually a few tenths of a percent). It is redundant on the custom attention
  path, but not an established material throughput loss.
- At 16 × 4096 → 16, removing prefill batch chunking changed throughput by
  +0.15%. The tested smaller-chunk setting equaled the existing 32768-row
  budget on this shape and was therefore a control, not an independent chunk
  size. Explicit KV-head expansion instead of native GQA lost 3.3%.
- Prefill graphs, INT8 KV cache, last-block GEMM/add/norm fusion, full long
  speculation, and attention autotuning were already off in the ranked
  engine. Their presence as optional code does not explain timed GPU cost.

Source: [large-prefill controls](bench/results/ablation-prefill-20260920.json).

## Method and reproduction

- One physical H100 per comparison; H200 substitutions are rejected.
- Fixed weights/tuning for structural comparisons, separately captured graphs,
  fresh prompt cache for each generation, and rotated variant order.
- Every measured generation is teacher-forced through untouched native Qwen.
- Reference replays run after each timing sweep, so a reference forward does
  not warm weights immediately before just one measured variant.
- Batch-one structural comparisons disable speculation uniformly. Separate
  studies compare the shipping speculative stream, its governor, and ordinary
  decoding. Never interpret an ordinary-decode comparison as a shipping gain
  without accounting for this distinction.
- Corpus slices use fixed per-workload seeds. Repeated variants share prompts;
  the total number of replayed positions is not a count of unique prompts.
- These local timings exclude the official process/pipe boundary. The measured
  tiny host-stream difference may understate its value in the official runner.
- No production engine files were changed by this ablation study.

The harness automatically creates a source snapshot from the requested commit:

```sh
DRYFT_ABLATION_COMMIT=43822c0 .venv/bin/modal run bench/ablation_modal.py \
  --shape public-2 --samples 5 --corpus all \
  --variants separate_decode_norm+attention_target128 \
  --output bench/results/layout-confirmation.json

DRYFT_ABLATION_COMMIT=43822c0 .venv/bin/modal run bench/ablation_modal.py \
  --shape long-output --samples 5 --corpus all \
  --variants shipping,spec_target120,spec_target125 \
  --output bench/results/governor-confirmation.json
```

The report deliberately retains failed variants. They are evidence against
those changes, not candidates to promote based on their throughput.
