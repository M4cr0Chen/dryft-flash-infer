"""Grouped-query flash decoding, split along the KV length.

At batch 1 a decode step has eight KV heads of work, which leaves 132 SMs
mostly idle and makes every attention call cost its launch rather than its
bandwidth. Splitting the cache into chunks that are reduced separately and
merged afterwards is the standard answer, and it is what lets this read only
the live part of the cache instead of the whole capacity.

Both kernels take the sequence length from device memory, so the grid is fixed
and the whole thing captures into a CUDA graph.

Numerics follow the SDPA kernels the reference dispatches to: fp32 softmax and
accumulator, the probability-by-value product in bfloat16.
"""

import torch
import triton
import triton.language as tl

_BLOCK_N = 64
_PAD_M = 16  # tl.dot wants at least 16 rows; a KV group only has four


@triton.jit
def _split_kernel(
    Q, K, V, POS, PACC, PM, PL, scale,
    LMAX: tl.constexpr, SPLIT: tl.constexpr, SPLITS: tl.constexpr,
    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, PAD_M: tl.constexpr,
):
    head = tl.program_id(0)  # batch * n_kv_heads + kv_head
    part = tl.program_id(1)

    live = tl.load(POS) + 1  # keys 0..pos inclusive
    start = part * SPLIT
    stop = tl.minimum(start + SPLIT, live)

    rows = tl.arange(0, PAD_M)
    dims = tl.arange(0, D)
    valid = rows < GROUP

    q = tl.load(
        Q + head.to(tl.int64) * (GROUP * D) + rows[:, None] * D + dims[None, :],
        mask=valid[:, None], other=0.0,
    )

    peak = tl.full((PAD_M,), -float("inf"), tl.float32)
    total = tl.zeros((PAD_M,), tl.float32)
    acc = tl.zeros((PAD_M, D), tl.float32)

    base = head.to(tl.int64) * (LMAX * D)
    for offset in range(start, stop, BLOCK_N):
        keys = offset + tl.arange(0, BLOCK_N)
        keep = keys < stop
        block = base + keys[:, None] * D + dims[None, :]
        k = tl.load(K + block, mask=keep[:, None], other=0.0)
        v = tl.load(V + block, mask=keep[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(keep[None, :], scores, -float("inf"))

        highest = tl.maximum(peak, tl.max(scores, axis=1))
        carry = tl.exp(peak - highest)
        probability = tl.exp(scores - highest[:, None])

        total = total * carry + tl.sum(probability, axis=1)
        acc = acc * carry[:, None] + tl.dot(probability.to(v.dtype), v)
        peak = highest

    empty = start >= live
    peak = tl.where(empty, -float("inf"), peak)
    total = tl.where(empty, 0.0, total)

    slot = head.to(tl.int64) * SPLITS + part
    tl.store(PM + slot * GROUP + rows, peak, mask=valid)
    tl.store(PL + slot * GROUP + rows, total, mask=valid)
    tl.store(
        PACC + slot * (GROUP * D) + rows[:, None] * D + dims[None, :],
        tl.where(empty, 0.0, acc),
        mask=valid[:, None],
    )


@triton.jit
def _merge_kernel(
    PACC, PM, PL, OUT,
    SPLITS: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)

    cols = tl.arange(0, WIDTH)
    parts = tl.arange(0, SPLITS)
    group_of = cols // D

    stat = head * SPLITS * GROUP + parts[:, None] * GROUP + group_of[None, :]
    peak = tl.load(PM + stat)
    total = tl.load(PL + stat)
    acc = tl.load(
        PACC + head * SPLITS * WIDTH + parts[:, None] * WIDTH + cols[None, :]
    )

    highest = tl.max(peak, axis=0)
    weight = tl.exp(peak - highest[None, :])
    out = tl.sum(acc * weight, axis=0) / tl.sum(total * weight, axis=0)
    tl.store(OUT + head * WIDTH + cols, out.to(OUT.dtype.element_ty))


def plan_splits(capacity: int, batch: int, n_kv: int) -> tuple[int, int]:
    """How many KV chunks to run in parallel, and how long each one is.

    Enough programs to cover the SMs, but never so many that a chunk is shorter
    than a block. Both come back as powers of two so the merge can hold all the
    partials in one tile.
    """
    splits = 1
    while (
        splits < 32
        and batch * n_kv * splits < 132
        and capacity // (splits * 2) >= _BLOCK_N * 2
    ):
        splits *= 2
    length = triton.cdiv(triton.cdiv(capacity, splits), _BLOCK_N) * _BLOCK_N
    return splits, length


class DecodeAttention:
    """Persistent partial-result buffers, sized once so the graph can capture."""

    def __init__(self, batch, n_kv, group, head_dim, capacity, device):
        self.batch, self.n_kv, self.group, self.head_dim = batch, n_kv, group, head_dim
        self.capacity = capacity
        self.scale = head_dim**-0.5
        self.splits, self.split_len = plan_splits(capacity, batch, n_kv)

        heads = batch * n_kv
        self.acc = torch.empty(
            heads, self.splits, group * head_dim, dtype=torch.float32, device=device
        )
        self.peak = torch.empty(
            heads, self.splits, group, dtype=torch.float32, device=device
        )
        self.total = torch.empty_like(self.peak)
        self.out = torch.empty(
            heads, group * head_dim, dtype=torch.bfloat16, device=device
        )

    def __call__(self, q, k_cache, v_cache, pos):
        """``q`` is ``[batch, n_kv, group, head_dim]``; caches are ``[batch, n_kv, cap, d]``."""
        heads = self.batch * self.n_kv
        _split_kernel[(heads, self.splits)](
            q, k_cache, v_cache, pos, self.acc, self.peak, self.total, self.scale,
            LMAX=self.capacity, SPLIT=self.split_len, SPLITS=self.splits,
            GROUP=self.group, D=self.head_dim, BLOCK_N=_BLOCK_N, PAD_M=_PAD_M,
            num_warps=4,
        )
        _merge_kernel[(heads,)](
            self.acc, self.peak, self.total, self.out,
            SPLITS=self.splits, GROUP=self.group, D=self.head_dim,
            WIDTH=self.group * self.head_dim, num_warps=4,
        )
        return self.out.view(self.batch, self.n_kv * self.group * self.head_dim)
