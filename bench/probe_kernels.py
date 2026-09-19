"""Kernels for diagnostics. Module scope, because triton.jit resolves names
from the defining module's globals and cannot see a function's locals.
"""

import triton
import triton.language as tl


@triton.jit
def drain(W, Y, K, N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Stream a weight and reduce it. No scales, no activation: pure read rate."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < N
    base = W + rows.to(tl.int64)[:, None] * K
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        acc += tl.load(
            base + cols[None, :], mask=live[:, None] & (cols < K)[None, :], other=0.0
        ).to(tl.float32)
    tl.store(Y + rows, tl.sum(acc, axis=1), mask=live)
