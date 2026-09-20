"""A skinny matrix product for decode, and a chooser that races it against cuBLAS.

Decode multiplies a handful of rows by a very tall weight, and cuBLAS does not
serve every one of those shapes well. Measured on an H100 at batch 1: the
19456-row gate/up projection streams at 2.9 TB/s, but the 2560-row output and
down projections fall to 1.6-2.2 TB/s on a GEMV kernel and an Ampere cutlass
kernel respectively. The difference is occupancy -- 2560 outputs in tiles of 64
is 40 programs for 132 SMs.

So: tile the output rows narrowly enough to fill the GPU, stream the weight
along K where it is contiguous, and accumulate in fp32 as cuBLAS does. Then
measure both and keep whichever wins, per shape, while the clock is off.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .timing import time_calls

_SM_TARGET = 132
_PAD_M = 16


@triton.jit
def _skinny_kernel(
    X, W, Y, K, CHUNK,
    N: tl.constexpr, M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SINGLE: tl.constexpr,
    PAD_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < N
    weight = W + rows.to(tl.int64)[:, None] * K

    # Each K split owns one slice of the reduction and its own output plane.
    begin = tl.program_id(1) * CHUNK
    finish = tl.minimum(begin + CHUNK, K)
    Y = Y + tl.program_id(1).to(tl.int64) * (M * N)

    if SINGLE:
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for start in range(begin, finish, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            inside = cols < finish
            tile = tl.load(
                weight + cols[None, :], mask=live[:, None] & inside[None, :], other=0.0
            ).to(tl.float32)
            vector = tl.load(X + cols, mask=inside, other=0.0).to(tl.float32)
            acc += tl.sum(tile * vector[None, :], axis=1)
        tl.store(Y + rows, acc.to(Y.dtype.element_ty), mask=live)
    else:
        lanes = tl.arange(0, PAD_M)
        present = lanes < M
        acc = tl.zeros((PAD_M, BLOCK_N), tl.float32)
        for start in range(begin, finish, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            inside = cols < finish
            left = tl.load(
                X + lanes[:, None] * K + cols[None, :],
                mask=present[:, None] & inside[None, :], other=0.0,
            )
            tile = tl.load(
                weight + cols[None, :], mask=live[:, None] & inside[None, :], other=0.0
            )
            acc += tl.dot(left, tl.trans(tile))
        tl.store(
            Y + lanes[:, None] * N + rows[None, :],
            acc.to(Y.dtype.element_ty),
            mask=present[:, None] & live[None, :],
        )


@triton.jit
def _reduce_kernel(P, Y, WIDTH, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = cols < WIDTH
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLITS):
        acc += tl.load(P + part * WIDTH + cols, mask=live, other=0.0)
    tl.store(Y + cols, acc.to(Y.dtype.element_ty), mask=live)


def _tiling(rows_out: int) -> tuple[int, int]:
    """Narrow enough to fill the SMs, wide enough that a load is a cache line."""
    block_n = 128
    while block_n > 16 and triton.cdiv(rows_out, block_n) < _SM_TARGET:
        block_n //= 2
    return block_n, 128 if block_n <= 32 else 64


def skinny_matmul(x, weight, config=None):
    """``x @ weight.T`` for a short ``x``; ``weight`` is ``[out, in]`` contiguous."""
    batch, k = x.shape
    n = weight.shape[0]
    if config is None:
        block_n, block_k = _tiling(n)
        config = (block_n, block_k, 4, 1)
    block_n, block_k, warps, splits = config

    out = torch.empty((batch, n), dtype=x.dtype, device=x.device)
    target = out
    if splits > 1:
        target = torch.empty((splits, batch, n), dtype=torch.float32, device=x.device)

    chunk = triton.cdiv(triton.cdiv(k, splits), block_k) * block_k
    _skinny_kernel[(triton.cdiv(n, block_n), splits)](
        x, weight, target, k, chunk,
        N=n, M=batch, BLOCK_N=block_n, BLOCK_K=block_k, SINGLE=(batch == 1),
        PAD_M=max(16, triton.next_power_of_2(batch)), num_warps=warps,
    )
    if splits > 1:
        width = batch * n
        _reduce_kernel[(triton.cdiv(width, 1024),)](
            target, out, width, SPLITS=splits, BLOCK=1024, num_warps=4
        )
    return out


#: Swept at warmup. Wide tiles for tall weights, split K for short ones.
_CONFIGS = [
    (16, 128, 4, 1), (16, 128, 4, 4), (16, 256, 8, 4), (16, 128, 8, 8),
    (32, 128, 4, 1), (32, 128, 8, 4), (64, 128, 8, 1), (64, 128, 8, 2),
    (128, 64, 8, 1),
]


def _mm(x, weight):
    """``x @ weight`` with the weight already stored ``[in, out]``."""
    return torch.mm(x, weight)


def pick_matmul(batch: int, every, reps: int = 3, trials: int = 5, packed=None,
                use_graph=True):
    """Choose how to run this projection shape during decode.

    ``every`` is the weight from every layer. The measurement cycles through
    all of them because timing one in a loop measures the wrong thing: the qkv
    weight is 31 MiB and the output projection 21 MiB, both of which sit inside
    an H100's 50 MiB L2, so a tight loop reports L2 bandwidth while the real
    step streams a different weight per layer from HBM.

    Returns ``(fn, transpose, note)``: call ``fn(x, w)`` with ``w`` transposed
    to ``[in, out]`` if ``transpose`` is set. cuBLAS is the incumbent and has to
    be beaten by a clear margin, because the alternative to trusting a noisy
    measurement is a slower engine for the whole run.

    Runs during the budgeted warmup. Timing uses CUDA graphs when decode does.
    """
    import functools

    weight = every[0]
    device, dtype = weight.device, weight.dtype
    x = torch.randn(batch, weight.shape[1], dtype=dtype, device=device)
    reference = F.linear(x, weight)
    scale = reference.float().abs().max().item() or 1.0
    moved = weight.numel() * weight.element_size()

    def clock(fn, operands):
        return time_calls(lambda w: fn(x, w), operands, reps=reps,
                          trials=trials, use_graph=use_graph)

    def agrees(out, tolerance=0.02):
        return (out.float() - reference.float()).abs().max().item() / scale < tolerance

    incumbent = clock(F.linear, every)
    best = (F.linear, False, incumbent)

    transposed = [w.t().contiguous() for w in every]
    if agrees(_mm(x, transposed[0])):
        elapsed = clock(_mm, transposed)
        if elapsed < best[2]:
            best = (_mm, True, elapsed)
    del transposed
    torch.cuda.empty_cache()

    if packed is not None:
        from .fp8 import fp8_matmul

        from .fp8 import CONFIGS as FP8_CONFIGS

        for cfg in FP8_CONFIGS:
            runner = functools.partial(fp8_matmul, config=cfg)
            try:
                # This projection check catches gross implementation errors.
                # FP8's coarser rounding needs separate end-to-end replay;
                # passing this local bound does not establish token validity.
                if not agrees(runner(x, packed[0]), tolerance=0.08):
                    continue
            except Exception:
                continue
            elapsed = clock(runner, packed)
            if elapsed < best[2]:
                best = (runner, "fp8", elapsed)

    try:
        from . import cuda_gemv

        if cuda_gemv.ready():
            for cfg in cuda_gemv.CONFIGS:
                runner = functools.partial(cuda_gemv.cuda_matmul, config=cfg)
                try:
                    if not agrees(runner(x, every[0])):
                        continue
                except Exception:
                    continue
                elapsed = clock(runner, every)
                if elapsed < best[2]:
                    best = (runner, "cuda", elapsed)
    except Exception:
        pass

    for config in _CONFIGS:
        candidate = functools.partial(skinny_matmul, config=config)
        try:
            if not agrees(candidate(x, weight)):
                continue
        except Exception:
            continue
        elapsed = clock(candidate, every)
        if elapsed < best[2]:
            best = (candidate, False, elapsed)

    #: Only move off cuBLAS for a margin the measurement can actually resolve.
    fn, transpose, elapsed = best
    if fn is not F.linear and elapsed > incumbent * 0.97:
        fn, transpose, elapsed = F.linear, False, incumbent
    if transpose == "cuda":
        return fn, "cuda", f"cuda {moved / (elapsed * 1e-3) / 1e12:.2f} TB/s"
    if transpose == "fp8":
        from .fp8 import bytes_moved

        real = bytes_moved(packed[0])
        return fn, "fp8", (
            f"fp8 {real / (elapsed * 1e-3) / 1e12:.2f} TB/s"
            f" ({moved / (elapsed * 1e-3) / 1e12:.2f} effective)"
        )
    kind = "cublas" if fn is F.linear else ("cublas-t" if transpose else "triton")
    return fn, transpose, f"{kind} {moved / (elapsed * 1e-3) / 1e12:.2f} TB/s"
