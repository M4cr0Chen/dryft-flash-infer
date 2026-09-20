"""BF16 prefill gate/up projection with a SwiGLU epilogue.

Weights are rearranged at load time to [hidden, interleaved gate/up]. The
native BF16 rounding after projection, SiLU, and multiplication is retained.
Warmup chooses between this kernel and the existing cuBLAS + SwiGLU path;
measured generations never compile or tune a new shape.
"""
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .swiglu import swiglu
from .timing import time_calls


@triton.jit
def _gate_up_swiglu(X, W, Y, M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(2 * I, BN)
    first = (pid // (8 * nn)) * 8
    count = tl.minimum(nm - first, 8)
    mi = first + pid % count
    ni = (pid % (8 * nn)) // count
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + kr
        x = tl.load(X + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), 0)
        w = tl.load(W + ks[:, None] * (2 * I) + cols[None, :],
                    (ks[:, None] < K) & (cols[None, :] < 2 * I), 0)
        acc = tl.dot(x, w, acc)
    gate, up = tl.split(acc.to(tl.bfloat16).to(tl.float32).reshape(BM, BN // 2, 2))
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    out_cols = ni * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(Y + rows[:, None] * I + out_cols[None, :], (silu * up).to(tl.bfloat16),
             (rows[:, None] < M) & (out_cols[None, :] < I))


def gate_up_swiglu(x, packed_weight, stages=3):
    """Contiguous BF16 x[M,H], weight[H,2I] -> new contiguous BF16 output[M,I]."""
    rows, hidden = x.shape
    inter = packed_weight.shape[1] // 2
    out = torch.empty((rows, inter), device=x.device, dtype=torch.bfloat16)
    _gate_up_swiglu[(triton.cdiv(rows, 128) * triton.cdiv(2 * inter, 256),)](
        x, packed_weight, out, rows, inter, hidden, 128, 256, 64,
        num_warps=8, num_stages=stages,
    )
    return out


class PrefillMLP:
    def __init__(self, weights):
        self.weights = weights
        self.packed = [w.view(2, w.shape[0] // 2, w.shape[1]).permute(2, 1, 0)
                       .reshape(w.shape[1], w.shape[0]).contiguous() for w in weights]
        self.configs = {}

    @torch.no_grad()
    def tune(self, rows):
        if rows in self.configs:
            return
        self.configs[rows] = None
        # Small prefill was flat in paired full-generation measurements.
        if rows < 1024:
            return
        sample = self.weights[0]
        x = torch.randn(rows, sample.shape[1], device=sample.device, dtype=sample.dtype)
        indices = list(range(0, len(self.weights), 6))
        baseline = lambda i: swiglu(F.linear(x, self.weights[i]))
        initial = time_calls(baseline, indices, reps=1, trials=5)
        reference = baseline(0)
        scale = reference.float().abs().max().item() or 1.0
        best, chosen = initial, None
        for stages in (3, 4):
            try:
                got = gate_up_swiglu(x, self.packed[0], stages)
                if not torch.isfinite(got).all().item():
                    continue
                if (got.float() - reference.float()).abs().max().item() / scale > .01:
                    continue
                elapsed = time_calls(lambda i: gate_up_swiglu(x, self.packed[i], stages),
                                     indices, reps=1, trials=5)
                if elapsed < best and elapsed < initial * .97:
                    best, chosen = elapsed, stages
            except Exception as exc:
                print(f"engine: prefill MLP candidate unavailable: {type(exc).__name__}: {exc}"[:300],
                      file=sys.stderr)
        if chosen is not None:
            # Recheck after compilation; do not promote a cold/hot-clock mismatch.
            current = time_calls(baseline, indices, reps=1, trials=5)
            candidate = time_calls(lambda i: gate_up_swiglu(x, self.packed[i], chosen),
                                   indices, reps=1, trials=5)
            if candidate >= current * .97:
                chosen = None
        self.configs[rows] = chosen
        print(f"engine: prefill MLP rows={rows} fused_stages={chosen} micro={initial / best:.3f}x",
              file=sys.stderr)

    def __call__(self, x, index):
        stages = self.configs.get(x.shape[0])
        if stages is None:
            return swiglu(F.linear(x, self.weights[index]))
        return gate_up_swiglu(x, self.packed[index], stages)
