"""RMSNorm, and the residual add fused into the norm that follows it.

Both kernels reproduce ``Qwen3RMSNorm.forward``'s cast placement exactly:

    return self.weight * hidden_states.to(input_dtype)

so the normalised value is rounded to bfloat16 *before* the weight multiply.
Keeping the product in fp32 is more accurate, and is a different function: on
some prompt it moves a logit further than the 2.0 tie margin allows. Reorder
arithmetic freely; do not reformulate it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(X, W, Y, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    scale = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * scale).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + off, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _add_rms_norm_kernel(X, D, W, R, Y, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols

    # The residual add is a bfloat16 add in the reference, so it rounds here.
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(D + off, mask=mask, other=0.0).to(tl.float32)
    r = (x + d).to(tl.bfloat16)
    tl.store(R + off, r, mask=mask)

    rf = r.to(tl.float32)
    scale = tl.math.rsqrt(tl.sum(rf * rf, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (rf * scale).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + off, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _add_rms_norm_partials_kernel(
    X, P, W, R, Y, ROWS, N: tl.constexpr, SPLITS: tl.constexpr, eps, BLOCK: tl.constexpr
):
    """The split-K planes summed here instead of by a separate reduce kernel.

    The sum rounds to bfloat16 once, where the projection's output would have,
    and the residual add then rounds as in the reference.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLITS):
        acc += tl.load(P + (part * ROWS + row) * N + cols, mask=mask, other=0.0)
    d = acc.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = (x + d).to(tl.bfloat16)
    tl.store(R + off, r, mask=mask)

    rf = r.to(tl.float32)
    scale = tl.math.rsqrt(tl.sum(rf * rf, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (rf * scale).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + off, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _residual_partials_kernel(X, P, R, WIDTH: tl.constexpr, SPLITS: tl.constexpr):
    """Parallelize the plane reduction across channels before row-wise RMSNorm."""
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    acc = tl.full((256,), 0.0, tl.float32)
    for part in range(SPLITS):
        acc += tl.load(P + part * WIDTH + i, mask=i < WIDTH, other=0.0)
    delta = acc.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + i, mask=i < WIDTH, other=0.0).to(tl.float32)
    tl.store(R + i, (x + delta).to(tl.bfloat16), mask=i < WIDTH)


def _launch_shape(n_cols: int) -> tuple[int, int]:
    block = triton.next_power_of_2(n_cols)
    return block, max(4, min(16, block // 256))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension of a contiguous ``[rows, n]`` tensor."""
    rows, n = x.shape
    out = torch.empty_like(x)
    block, warps = _launch_shape(n)
    _rms_norm_kernel[(rows,)](x, weight, out, n, eps, BLOCK=block, num_warps=warps)
    return out


def add_rms_norm(
    x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual = x + delta`` in bfloat16, then RMSNorm of that residual.

    Returns ``(residual, normed)``. Both are fresh tensors; the caller keeps the
    residual for the next branch and feeds the normed value to the projection.
    """
    rows, n = x.shape
    residual = torch.empty_like(x)
    out = torch.empty_like(x)
    block, warps = _launch_shape(n)
    _add_rms_norm_kernel[(rows,)](
        x, delta, weight, residual, out, n, eps, BLOCK=block, num_warps=warps
    )
    return residual, out


def add_rms_norm_partials(
    x: torch.Tensor, partials: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``add_rms_norm`` where ``delta`` arrives as fp32 planes ``[splits, rows, n]``."""
    rows, n = x.shape
    splits = partials.shape[0]
    residual = torch.empty_like(x)
    out = torch.empty_like(x)
    block, warps = _launch_shape(n)
    _add_rms_norm_partials_kernel[(rows,)](
        x, partials, weight, residual, out, rows, n, splits, eps,
        BLOCK=block, num_warps=warps,
    )
    return residual, out


def add_rms_norm_separate(x, delta, weight, eps):
    """Separate BF16 residual addition and RMSNorm, for short decode matrices."""
    residual = x + delta
    return residual, rms_norm(residual, weight, eps)


def add_rms_norm_partials_separate(x, partials, weight, eps):
    """Sequential FP32 plane sum, BF16 projection/residual casts, then RMSNorm."""
    residual = torch.empty_like(x)
    width = x.numel()
    _residual_partials_kernel[(triton.cdiv(width, 256),)](
        x, partials, residual, width, partials.shape[0], num_warps=4,
    )
    return residual, rms_norm(residual, weight, eps)
