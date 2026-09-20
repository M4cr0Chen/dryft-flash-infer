"""A hand-written CUDA GEMV for the decode projections.

cuBLAS reaches 53-79% of peak bandwidth on these shapes depending on how many
output rows they have, and the Triton attempts did worse. The suspicion this
kernel tests is that the gap is memory-level parallelism: at one row of
activation a projection is a pure streaming read, and what matters is how many
loads each thread keeps in flight, not arithmetic.

So: one warp per output row, each lane pulling 16 bytes at a time with several
independent accumulators, and the activation staged once in shared memory.

NVRTC has no toolkit headers, so bfloat16 travels as ``unsigned short``.
``bf16_to_f32`` is an exact bit shift; ``f32_to_bf16`` rounds to nearest even,
which is what cuBLAS does when it writes its bfloat16 result.
"""

import torch

from . import cuda_jit

SOURCE = r"""
typedef unsigned short bf16;

__device__ __forceinline__ float bf16_to_f32(bf16 h) {
    return __int_as_float(((int)(unsigned int)h) << 16);
}

__device__ __forceinline__ bf16 f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int round = ((u >> 16) & 1u) + 0x7fffu;   // nearest, ties to even
    return (bf16)((u + round) >> 16);
}

// out[n] = sum_k W[n, k] * x[k], W is [N, K] row major.
//
// Two ways to keep loads in flight, because which one a shape wants depends on
// its K. Striding one row with four independent accumulators overlaps four
// loads along K, which suits the long reductions (down_proj at K=9728 reaches
// 2.15 TB/s against cuBLAS's 1.79). Short-K shapes run out of iterations to
// overlap, so the R variants give a warp several rows instead. The warmup race
// picks; neither is better everywhere.
extern "C" __global__ void gemv_bf16_r1(
    const bf16* __restrict__ W, const bf16* __restrict__ x,
    bf16* __restrict__ out, int N, int K)
{
    extern __shared__ bf16 shared_x[];
    const int quads = K >> 3;
    for (int i = threadIdx.x; i < quads; i += blockDim.x)
        reinterpret_cast<float4*>(shared_x)[i] =
            reinterpret_cast<const float4*>(x)[i];
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const float4* xs = reinterpret_cast<const float4*>(shared_x);

    for (int row = blockIdx.x * warps + warp; row < N; row += gridDim.x * warps) {
        const float4* w = reinterpret_cast<const float4*>(W + (size_t)row * K);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = lane;
        for (; i + 96 < quads; i += 128) {
            float4 w0 = w[i], w1 = w[i + 32], w2 = w[i + 64], w3 = w[i + 96];
            float4 x0 = xs[i], x1 = xs[i + 32], x2 = xs[i + 64], x3 = xs[i + 96];
            const bf16* wb0 = reinterpret_cast<const bf16*>(&w0);
            const bf16* wb1 = reinterpret_cast<const bf16*>(&w1);
            const bf16* wb2 = reinterpret_cast<const bf16*>(&w2);
            const bf16* wb3 = reinterpret_cast<const bf16*>(&w3);
            const bf16* xb0 = reinterpret_cast<const bf16*>(&x0);
            const bf16* xb1 = reinterpret_cast<const bf16*>(&x1);
            const bf16* xb2 = reinterpret_cast<const bf16*>(&x2);
            const bf16* xb3 = reinterpret_cast<const bf16*>(&x3);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                a0 = fmaf(bf16_to_f32(wb0[j]), bf16_to_f32(xb0[j]), a0);
                a1 = fmaf(bf16_to_f32(wb1[j]), bf16_to_f32(xb1[j]), a1);
                a2 = fmaf(bf16_to_f32(wb2[j]), bf16_to_f32(xb2[j]), a2);
                a3 = fmaf(bf16_to_f32(wb3[j]), bf16_to_f32(xb3[j]), a3);
            }
        }
        for (; i < quads; i += 32) {
            float4 wv = w[i], xv = xs[i];
            const bf16* wb = reinterpret_cast<const bf16*>(&wv);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                a0 = fmaf(bf16_to_f32(wb[j]), bf16_to_f32(xb[j]), a0);
        }
        float acc = (a0 + a1) + (a2 + a3);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[row] = f32_to_bf16(acc);
    }
}

// out[n] = sum_k W[n, k] * x[k], W is [N, K] row major.
//
// One warp owns R consecutive output rows and keeps a load in flight for each,
// so the loads outstanding per lane come from the row count rather than from
// the length of K. That matters because these projections have K of only 2560
// to 9728: striding one row gives a lane just ten 16-byte loads to overlap,
// and the short-K shapes were exactly the ones losing to cuBLAS.
#define GEMV(R)                                                               \
extern "C" __global__ void gemv_bf16_r##R(                                    \
    const bf16* __restrict__ W, const bf16* __restrict__ x,                   \
    bf16* __restrict__ out, int N, int K)                                     \
{                                                                             \
    extern __shared__ bf16 shared_x[];                                        \
    const int quads = K >> 3;                                                 \
    for (int i = threadIdx.x; i < quads; i += blockDim.x)                     \
        reinterpret_cast<float4*>(shared_x)[i] =                              \
            reinterpret_cast<const float4*>(x)[i];                            \
    __syncthreads();                                                          \
                                                                              \
    const int lane  = threadIdx.x & 31;                                       \
    const int warp  = threadIdx.x >> 5;                                       \
    const int warps = blockDim.x >> 5;                                        \
    const float4* xs = reinterpret_cast<const float4*>(shared_x);             \
                                                                              \
    for (int base = (blockIdx.x * warps + warp) * R; base < N;                \
             base += gridDim.x * warps * R) {                                 \
        float acc[R];                                                         \
        _Pragma("unroll") for (int r = 0; r < R; ++r) acc[r] = 0.f;           \
        const int rows = (N - base) < R ? (N - base) : R;                     \
                                                                              \
        for (int i = lane; i < quads; i += 32) {                              \
            float4 xv = xs[i];                                                \
            const bf16* xb = reinterpret_cast<const bf16*>(&xv);              \
            float4 wv[R];                                                     \
            _Pragma("unroll") for (int r = 0; r < R; ++r)                     \
                if (r < rows)                                                 \
                    wv[r] = reinterpret_cast<const float4*>(                  \
                        W + (size_t)(base + r) * K)[i];                       \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                   \
                if (r >= rows) continue;                                      \
                const bf16* wb = reinterpret_cast<const bf16*>(&wv[r]);       \
                _Pragma("unroll") for (int j = 0; j < 8; ++j)                 \
                    acc[r] = fmaf(bf16_to_f32(wb[j]), bf16_to_f32(xb[j]),     \
                                  acc[r]);                                    \
            }                                                                 \
        }                                                                     \
        _Pragma("unroll") for (int r = 0; r < R; ++r) {                       \
            float a = acc[r];                                                 \
            _Pragma("unroll") for (int off = 16; off; off >>= 1)              \
                a += __shfl_down_sync(0xffffffffu, a, off);                   \
            if (lane == 0 && r < rows) out[base + r] = f32_to_bf16(a);        \
        }                                                                     \
    }                                                                         \
}

GEMV(2)
GEMV(4)
GEMV(8)
"""

_module = None
_ready = None


def ready() -> bool:
    """Compile once. False means this runtime cannot JIT CUDA, which is fine."""
    global _module, _ready
    if _ready is None:
        try:
            _module = cuda_jit.Module(SOURCE)
            for rows in (1, 2, 4, 8):
                _module.kernel(f"gemv_bf16_r{rows}")
            _ready = True
        except Exception as exc:
            print(f"cuda_gemv: unavailable ({type(exc).__name__}: {exc})"[:300])
            _ready = False
    return _ready


#: (threads per block, blocks, rows per warp). Swept at warmup.
CONFIGS = [
    (256, 264, 1), (256, 264, 2), (256, 264, 4), (256, 264, 8),
    (128, 528, 2), (128, 528, 4), (512, 132, 4), (256, 528, 2),
]


def cuda_matmul(x: torch.Tensor, weight: torch.Tensor, config=None) -> torch.Tensor:
    """``x @ weight.T`` for a single row of ``x``."""
    batch, k = x.shape
    n = weight.shape[0]
    if batch != 1:
        raise ValueError("the cuda gemv serves batch 1")
    if k % 8:
        raise ValueError(f"K={k} is not a multiple of eight")

    block, grid, rows = config or (256, 264, 4)
    out = torch.empty((batch, n), dtype=torch.bfloat16, device=x.device)
    kernel = _module.kernel(f"gemv_bf16_r{rows}")
    kernel.set_shared(k * 2)
    kernel(grid, block, weight, x, out, n, k)
    return out
