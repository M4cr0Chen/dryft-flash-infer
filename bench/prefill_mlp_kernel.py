"""BF16 gate/up GEMM + SwiGLU experiment, for Triton 3.1 on Hopper.

Projection, SiLU, and product each round to BF16 at the native boundaries.
The prepacked weight interleaves gate/up rows; no quantization is introduced.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, W, Y, M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GM: tl.constexpr, PACKED: tl.constexpr, TRANSPOSED: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(2 * I, BN)
    group = pid // (GM * nn)
    first = group * GM
    count = tl.minimum(nm - first, GM)
    mi = first + pid % count
    ni = (pid % (GM * nn)) // count
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    if PACKED:
        wc = cols
    else:
        wc = cols // 2 + (cols % 2) * I
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + kr
        x = tl.load(X + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), 0)
        if TRANSPOSED:
            wp = W + ks[:, None] * (2 * I) + wc[None, :]
        else:
            wp = W + ks[:, None] + wc[None, :] * K
        w = tl.load(wp,
                    (ks[:, None] < K) & (cols[None, :] < 2 * I), 0)
        acc = tl.dot(x, w, acc)
    # Pair columns in the accumulator before the epilogue. Casting is not
    # optional: native F.linear writes BF16 before applying SiLU.
    pair = acc.to(tl.bfloat16).to(tl.float32).reshape(BM, BN // 2, 2)
    gate, up = tl.split(pair)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    out = (silu * up).to(tl.bfloat16)
    output_cols = ni * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(Y + rows[:, None] * I + output_cols[None, :], out,
             (rows[:, None] < M) & (output_cols[None, :] < I))


CONFIGS = [
    (64, 128, 32, 4, 3), (64, 128, 64, 4, 3), (64, 128, 64, 4, 4),
    (128, 128, 32, 4, 3), (128, 128, 64, 4, 3), (128, 128, 64, 8, 3),
    (64, 256, 32, 4, 3), (64, 256, 64, 4, 3), (64, 256, 64, 8, 3),
    (128, 256, 64, 8, 3), (128, 256, 32, 8, 4),
    (128, 128, 128, 8, 3), (64, 128, 128, 4, 3),
]

LARGE_CONFIGS = [
    (128,256,64,8,3), (128,256,64,4,3), (128,256,64,8,4),
    (256,128,64,8,3), (256,128,64,8,4), (256,128,32,8,4),
    (256,256,64,8,3), (128,256,128,8,2), (128,128,64,8,3),
    (128,128,64,4,3), (64,256,64,4,3), (64,128,64,4,3),
]


def pack(weight):
    inter = weight.shape[0] // 2
    return weight.view(2, inter, weight.shape[1]).transpose(0, 1).contiguous().view_as(weight)


def fused(x, weight, config=(128, 128, 64, 4, 3), packed=True, transposed=False):
    rows, k = x.shape
    inter = weight.shape[1 if transposed else 0] // 2
    bm, bn, bk, warps, stages = config
    out = torch.empty((rows, inter), device=x.device, dtype=x.dtype)
    _gate_up[(triton.cdiv(rows, bm) * triton.cdiv(2 * inter, bn),)](
        x, weight, out, rows, inter, k, bm, bn, bk, 8, packed, transposed,
        num_warps=warps, num_stages=stages,
    )
    return out
