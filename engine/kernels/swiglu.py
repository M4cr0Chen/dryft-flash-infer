"""SwiGLU over a fused ``[gate | up]`` projection output.

The reference is ``act_fn(gate) * up`` with both operands bfloat16, so silu
rounds before the multiply does.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(GU, OUT, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I

    gate = tl.load(GU + row * (2 * I) + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GU + row * (2 * I) + I + cols, mask=mask, other=0.0).to(tl.float32)

    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * I + cols, (silu * up).to(tl.bfloat16), mask=mask)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """``[rows, 2 * inter]`` fused projection output to ``[rows, inter]``."""
    rows, width = gate_up.shape
    inter = width // 2
    out = torch.empty((rows, inter), dtype=gate_up.dtype, device=gate_up.device)
    block = 1024
    _swiglu_kernel[(rows, triton.cdiv(inter, block))](
        gate_up, out, I=inter, BLOCK=block, num_warps=4
    )
    return out


@triton.jit
def _partial_swiglu_kernel(P, OUT, ROWS: tl.constexpr, I: tl.constexpr,
                          SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    gate = tl.full((BLOCK,), 0.0, tl.float32)
    up = tl.full((BLOCK,), 0.0, tl.float32)
    for part in range(SPLITS):
        offset = part * ROWS * (2 * I) + row * (2 * I) + cols
        gate += tl.load(P + offset, mask=mask, other=0.0)
        up += tl.load(P + offset + I, mask=mask, other=0.0)
    # The projection, SiLU and product each retain their BF16 rounding.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * I + cols, (silu * up).to(tl.bfloat16), mask=mask)


def swiglu_partials(partials: torch.Tensor) -> torch.Tensor:
    """Sequentially reduce FP32 [splits, rows, 2*I] planes and apply SwiGLU."""
    splits, rows, width = partials.shape
    inter = width // 2
    out = torch.empty((rows, inter), device=partials.device, dtype=torch.bfloat16)
    _partial_swiglu_kernel[(rows, triton.cdiv(inter, 1024))](
        partials, out, rows, inter, splits, 1024, num_warps=4,
    )
    return out
