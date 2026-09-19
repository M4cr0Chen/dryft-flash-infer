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
    Q, K, V, POS, PACC, PM, PL, scale, NQ,
    LMAX: tl.constexpr, SPLIT: tl.constexpr, SPLITS: tl.constexpr,
    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, PAD_M: tl.constexpr,
    TOKENS: tl.constexpr, NKV: tl.constexpr,
):
    head = tl.program_id(0)  # batch * n_kv_heads + kv_head
    part = tl.program_id(1)

    # A pass may carry several query tokens: the confirmed one and its draft.
    # Token t sits one position later than token t-1, so it sees one more key.
    base = tl.load(POS)
    start = part * SPLIT
    stop = tl.minimum(start + SPLIT, base + TOKENS)

    rows = tl.arange(0, PAD_M)
    dims = tl.arange(0, D)
    valid = rows < TOKENS * GROUP
    token = rows // GROUP
    lane = rows % GROUP
    live = base + token + 1

    sequence = head.to(tl.int64) // NKV
    kv_head = head.to(tl.int64) % NKV
    q = tl.load(
        Q
        + ((sequence * TOKENS + token.to(tl.int64)) * NQ + kv_head * GROUP + lane)[:, None] * D
        + dims[None, :],
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
        # Causal within the draft block as well as against the cache.
        scores = tl.where(keep[None, :] & (keys[None, :] < live[:, None]),
                          scores, -float("inf"))

        highest = tl.maximum(peak, tl.max(scores, axis=1))
        carry = tl.exp(peak - highest)
        probability = tl.exp(scores - highest[:, None])

        total = total * carry + tl.sum(probability, axis=1)
        acc = acc * carry[:, None] + tl.dot(probability.to(v.dtype), v)
        peak = highest

    empty = start >= live
    peak = tl.where(empty, -float("inf"), peak)
    total = tl.where(empty, 0.0, total)

    width = TOKENS * GROUP
    slot = head.to(tl.int64) * SPLITS + part
    tl.store(PM + slot * width + rows, peak, mask=valid)
    tl.store(PL + slot * width + rows, total, mask=valid)
    tl.store(
        PACC + slot * (width * D) + rows[:, None] * D + dims[None, :],
        tl.where(empty[:, None], 0.0, acc),
        mask=valid[:, None],
    )


@triton.jit
def _merge_kernel(
    PACC, PM, PL, OUT, NQ,
    SPLITS: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr,
    TOKENS: tl.constexpr, NKV: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)

    cols = tl.arange(0, WIDTH)  # GROUP * D, one token's worth
    parts = tl.arange(0, SPLITS)
    lane_of = cols // D
    rows_per_head = TOKENS * GROUP

    row = token * GROUP + lane_of
    stat = head * SPLITS * rows_per_head + parts[:, None] * rows_per_head + row[None, :]
    peak = tl.load(PM + stat)
    total = tl.load(PL + stat)
    acc = tl.load(
        PACC
        + head * SPLITS * rows_per_head * D
        + parts[:, None] * (rows_per_head * D)
        + (token * GROUP * D + cols)[None, :]
    )

    highest = tl.max(peak, axis=0)
    weight = tl.exp(peak - highest[None, :])
    out = tl.sum(acc * weight, axis=0) / tl.sum(total * weight, axis=0)

    # Back to the projection's layout: [batch * tokens, n_q * head_dim].
    sequence = head // NKV
    kv_head = head % NKV
    tl.store(
        OUT + (sequence * TOKENS + token) * NQ * D + kv_head * GROUP * D + cols,
        out.to(OUT.dtype.element_ty),
    )


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
    """Persistent partial-result buffers, sized once so the graph can capture.

    ``tokens`` is how many query positions one pass carries: 1 for plain
    decoding, or the confirmed token plus its draft when speculating.
    """

    def __init__(self, batch, n_kv, group, head_dim, capacity, device, tokens=1):
        self.batch, self.n_kv, self.group, self.head_dim = batch, n_kv, group, head_dim
        self.capacity, self.tokens = capacity, tokens
        self.n_q = n_kv * group
        self.scale = head_dim**-0.5
        self.splits, self.split_len = plan_splits(capacity, batch, n_kv)
        self.pad_m = max(_PAD_M, triton.next_power_of_2(tokens * group))

        heads = batch * n_kv
        rows = tokens * group
        self.acc = torch.empty(
            heads, self.splits, rows * head_dim, dtype=torch.float32, device=device
        )
        self.peak = torch.empty(
            heads, self.splits, rows, dtype=torch.float32, device=device
        )
        self.total = torch.empty_like(self.peak)
        self.out = torch.empty(
            batch * tokens, self.n_q * head_dim, dtype=torch.bfloat16, device=device
        )

    def __call__(self, q, k_cache, v_cache, pos):
        """``q`` is ``[batch * tokens, n_q * head_dim]``; caches ``[batch, n_kv, cap, d]``."""
        heads = self.batch * self.n_kv
        _split_kernel[(heads, self.splits)](
            q, k_cache, v_cache, pos, self.acc, self.peak, self.total, self.scale,
            self.n_q,
            LMAX=self.capacity, SPLIT=self.split_len, SPLITS=self.splits,
            GROUP=self.group, D=self.head_dim, BLOCK_N=_BLOCK_N, PAD_M=self.pad_m,
            TOKENS=self.tokens, NKV=self.n_kv, num_warps=4,
        )
        _merge_kernel[(heads, self.tokens)](
            self.acc, self.peak, self.total, self.out, self.n_q,
            SPLITS=self.splits, GROUP=self.group, D=self.head_dim,
            WIDTH=self.group * self.head_dim, TOKENS=self.tokens, NKV=self.n_kv,
            num_warps=4,
        )
        return self.out
