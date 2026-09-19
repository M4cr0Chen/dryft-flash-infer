"""The MLP's first half as one kernel: gate/up projection with SwiGLU folded in.

Three things make this fusion work where others do not. A block that owns
output index ``i`` can also own ``i + inter``, so it holds both halves of the
SwiGLU pair and needs no cross-block exchange. The activation is small enough
to stay in L2, so every block re-reading it costs almost nothing against the
99 MiB of weight each step streams. And the intermediate never reaches memory:
the reference writes 19456 values and reads them back to produce 9728, while
this writes only the 9728.

Rounding follows the reference exactly. ``gate`` and ``up`` are rounded to
bfloat16 as a Linear would before SiLU sees them, SiLU rounds again, and the
product rounds once more -- three roundings, in the same places.
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
    unsigned int round = ((u >> 16) & 1u) + 0x7fffu;
    return (bf16)((u + round) >> 16);
}

// p[b, i] = silu(W[i] . x[b]) * (W[i + I] . x[b])
//
// One warp per output index, both halves of the SwiGLU pair together, and two
// chunks of K in flight at once. Specialised on the batch: a fixed-size
// accumulator array sized for sixteen rows costs thirty-two registers a lane
// even when only one row is live, which was enough to lose the batch-1 case.
#define MLP(NB)                                                               \
extern "C" __global__ void gate_up_swiglu_b##NB(                              \
    const bf16* __restrict__ W, const bf16* __restrict__ x,                   \
    bf16* __restrict__ p, int I, int K, int B)                                \
{                                                                             \
    extern __shared__ bf16 sx[];                                              \
    const int quads = K >> 3;                                                 \
    for (int i = threadIdx.x; i < B * quads; i += blockDim.x)                 \
        reinterpret_cast<float4*>(sx)[i] =                                    \
            reinterpret_cast<const float4*>(x)[i];                            \
    __syncthreads();                                                          \
                                                                              \
    const int lane  = threadIdx.x & 31;                                       \
    const int warp  = threadIdx.x >> 5;                                       \
    const int warps = blockDim.x >> 5;                                        \
    const float4* xs = reinterpret_cast<const float4*>(sx);                   \
                                                                              \
    for (int idx = blockIdx.x * warps + warp; idx < I;                        \
             idx += gridDim.x * warps) {                                      \
        const float4* wg =                                                    \
            reinterpret_cast<const float4*>(W + (size_t)idx * K);             \
        const float4* wu =                                                    \
            reinterpret_cast<const float4*>(W + (size_t)(idx + I) * K);       \
        float g0[NB], u0[NB], g1[NB], u1[NB];                                 \
        _Pragma("unroll") for (int b = 0; b < NB; ++b) {                      \
            g0[b] = 0.f; u0[b] = 0.f; g1[b] = 0.f; u1[b] = 0.f; }             \
                                                                              \
        int i = lane;                                                         \
        for (; i + 32 < quads; i += 64) {                                     \
            float4 a0 = wg[i], a1 = wg[i + 32];                               \
            float4 e0 = wu[i], e1 = wu[i + 32];                               \
            const bf16* ab0 = reinterpret_cast<const bf16*>(&a0);             \
            const bf16* ab1 = reinterpret_cast<const bf16*>(&a1);             \
            const bf16* eb0 = reinterpret_cast<const bf16*>(&e0);             \
            const bf16* eb1 = reinterpret_cast<const bf16*>(&e1);             \
            _Pragma("unroll") for (int b = 0; b < NB; ++b) {                  \
                float4 v0 = xs[(size_t)b * quads + i];                        \
                float4 v1 = xs[(size_t)b * quads + i + 32];                   \
                const bf16* vb0 = reinterpret_cast<const bf16*>(&v0);         \
                const bf16* vb1 = reinterpret_cast<const bf16*>(&v1);         \
                _Pragma("unroll") for (int j = 0; j < 8; ++j) {               \
                    float t0 = bf16_to_f32(vb0[j]);                           \
                    float t1 = bf16_to_f32(vb1[j]);                           \
                    g0[b] = fmaf(bf16_to_f32(ab0[j]), t0, g0[b]);             \
                    g1[b] = fmaf(bf16_to_f32(ab1[j]), t1, g1[b]);             \
                    u0[b] = fmaf(bf16_to_f32(eb0[j]), t0, u0[b]);             \
                    u1[b] = fmaf(bf16_to_f32(eb1[j]), t1, u1[b]);             \
                }                                                             \
            }                                                                 \
        }                                                                     \
        for (; i < quads; i += 32) {                                          \
            float4 a0 = wg[i], e0 = wu[i];                                    \
            const bf16* ab0 = reinterpret_cast<const bf16*>(&a0);             \
            const bf16* eb0 = reinterpret_cast<const bf16*>(&e0);             \
            _Pragma("unroll") for (int b = 0; b < NB; ++b) {                  \
                float4 v0 = xs[(size_t)b * quads + i];                        \
                const bf16* vb0 = reinterpret_cast<const bf16*>(&v0);         \
                _Pragma("unroll") for (int j = 0; j < 8; ++j) {               \
                    float t0 = bf16_to_f32(vb0[j]);                           \
                    g0[b] = fmaf(bf16_to_f32(ab0[j]), t0, g0[b]);             \
                    u0[b] = fmaf(bf16_to_f32(eb0[j]), t0, u0[b]);             \
                }                                                             \
            }                                                                 \
        }                                                                     \
                                                                              \
        _Pragma("unroll") for (int b = 0; b < NB; ++b) {                      \
            float gate = g0[b] + g1[b], up = u0[b] + u1[b];                   \
            _Pragma("unroll") for (int off = 16; off; off >>= 1) {            \
                gate += __shfl_down_sync(0xffffffffu, gate, off);             \
                up   += __shfl_down_sync(0xffffffffu, up, off);               \
            }                                                                 \
            if (lane == 0) {                                                  \
                float gb = bf16_to_f32(f32_to_bf16(gate));                    \
                float ub = bf16_to_f32(f32_to_bf16(up));                      \
                float si = bf16_to_f32(                                       \
                    f32_to_bf16(gb / (1.0f + __expf(-gb))));                  \
                p[(size_t)b * I + idx] = f32_to_bf16(si * ub);                \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

MLP(1)
MLP(2)
MLP(4)
MLP(8)
MLP(16)
"""

_module = None
_ready = None

#: (threads per block, blocks). Swept at warmup.
CONFIGS = [(256, 264), (256, 528), (128, 528), (512, 264), (256, 152)]


def ready() -> bool:
    global _module, _ready
    if _ready is None:
        try:
            _module = cuda_jit.Module(SOURCE)
            for nb in (1, 2, 4, 8, 16):
                _module.kernel(f"gate_up_swiglu_b{nb}")
            _ready = True
        except Exception as exc:
            print(f"cuda_mlp: unavailable ({type(exc).__name__}: {exc})"[:300])
            _ready = False
    return _ready


def gate_up_swiglu(x: torch.Tensor, weight: torch.Tensor, config=None) -> torch.Tensor:
    """``silu(gate) * up`` straight from the activation, skipping the 2*I write."""
    batch, k = x.shape
    inter = weight.shape[0] // 2
    if k % 8:
        raise ValueError(f"K={k} is not a multiple of eight")
    if batch > 16:
        raise ValueError("the fused mlp serves batch 16 and under")
    block, grid = config or (256, 264)
    out = torch.empty((batch, inter), dtype=torch.bfloat16, device=x.device)
    slot = next(nb for nb in (1, 2, 4, 8, 16) if nb >= batch)
    kernel = _module.kernel(f"gate_up_swiglu_b{slot}")
    kernel.set_shared(batch * k * 2)
    kernel(grid, block, weight, x, out, inter, k, batch)
    return out
