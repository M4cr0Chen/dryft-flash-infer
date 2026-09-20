"""The attention input stage as one kernel: norm, QKV, head norms, RoPE, cache.

Four kernels become one. The obstacle is that q_norm and k_norm reduce over a
128-value head, and a head is 128 consecutive rows of the projection output, so
a block can only normalise locally if it owns whole heads -- which is 48 blocks
for 132 SMs and starves the projection.

A grid barrier resolves it. Phase one runs the projection across every resident
block and writes raw QKV to scratch; the barrier; phase two hands one warp per
head, which then has all 128 values and can norm, rotate and store the cache
slot on its own. Cooperative launch is what makes the barrier safe -- every
block is resident, so no block can spin on one that was never scheduled -- and
the launch captures into a CUDA graph, which was checked before this was built.

RoPE's partner lane falls out of the layout: lane ``l`` holds values ``4l..4l+3``
so the value 64 positions away lives in lane ``l ^ 16``, one shuffle away.

Not wired into the engine, because measured it does not pay. It is correct --
Q, K and V all land within one bfloat16 ULP of the four kernels it replaces,
in the right cache slot -- and it is one launch instead of four. But the
projection inside it runs at about 1 TB/s where cuBLAS reaches 2.15 on this
shape, and 6144 output rows is too few to hide that. The kernels it absorbs are
worth about 7 us a layer; the projection gives back more.

The same fusion pays for the MLP (see cuda_mlp.py) because the gate/up weight
has 19456 output rows, where this GEMV does match cuBLAS. That is the rule the
measurements keep returning: fusion is worth it exactly where the projection
inside it is already competitive, which is the large-N shapes.

Kept because the machinery is the point and is reusable: cooperative launch,
a hand-rolled grid barrier, and head-local norm and RoPE, all inside a CUDA
graph. Grid size makes no difference here (46-49 us from 132 to 660 blocks).
"""

import torch

from . import cuda_jit

_HEAD = 128

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

// Safe only under a cooperative launch: every block is resident, so no block
// spins waiting on one the scheduler never ran.
__device__ __forceinline__ void grid_barrier(
    unsigned int* counter, unsigned int* generation, unsigned int blocks)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int gen = atomicAdd(generation, 0u);
        __threadfence();
        if (atomicAdd(counter, 1u) == blocks - 1u) {
            atomicExch(counter, 0u);
            __threadfence();
            atomicAdd(generation, 1u);
        } else {
            while (atomicAdd(generation, 0u) == gen) { __threadfence(); }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float block_sum(float v, float* scratch, int threads) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < (threads >> 5); ++i) total += scratch[i];
    __syncthreads();
    return total;
}

#define QKV(NB, SKIP_NORM, SUFFIX)                                                               \
extern "C" __global__ void qkv_stage_b##NB##SUFFIX(                                   \
    const bf16* __restrict__ x, const bf16* __restrict__ wn,                  \
    const bf16* __restrict__ Wqkv, const bf16* __restrict__ qn,               \
    const bf16* __restrict__ kn, const bf16* __restrict__ cosT,               \
    const bf16* __restrict__ sinT, const int* __restrict__ posp,              \
    bf16* __restrict__ scratch, bf16* __restrict__ qout,                      \
    bf16* __restrict__ kcache, bf16* __restrict__ vcache,                     \
    unsigned int* barrier,                                                    \
    int B, int H, int QW, int KW, int NKV, int cap,                           \
    unsigned int blocks, float eps)                                           \
{                                                                             \
    extern __shared__ bf16 sx[];                                              \
    __shared__ float red[32];                                                 \
    const int quads = H >> 3;                                                 \
    const int rows  = QW + 2 * KW;                                            \
                                                                              \
    /* Every block normalises the activation again rather than moving it      \
       through a kernel boundary. Whether that trade pays is measured, not     \
       assumed: SKIP_NORM compiles the variant that takes it pre-normed. */    \
    for (int b = 0; b < B * (1 - SKIP_NORM); ++b) {                           \
        float ss = 0.f;                                                       \
        for (int k = threadIdx.x; k < H; k += blockDim.x) {                   \
            float v = bf16_to_f32(x[b * H + k]);                              \
            ss += v * v;                                                      \
        }                                                                     \
        float scale = rsqrtf(block_sum(ss, red, blockDim.x) / H + eps);       \
        for (int k = threadIdx.x; k < H; k += blockDim.x) {                   \
            float v = bf16_to_f32(x[b * H + k]) * scale;                      \
            sx[b * H + k] = f32_to_bf16(bf16_to_f32(f32_to_bf16(v))           \
                                        * bf16_to_f32(wn[k]));                \
        }                                                                     \
    }                                                                         \
    if (SKIP_NORM) {                                                          \
        for (int i = threadIdx.x; i < B * H; i += blockDim.x) sx[i] = x[i];   \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    const int lane  = threadIdx.x & 31;                                       \
    const int warp  = threadIdx.x >> 5;                                       \
    const int warps = blockDim.x >> 5;                                        \
    const float4* xs = reinterpret_cast<const float4*>(sx);                   \
                                                                              \
    for (int r = blockIdx.x * warps + warp; r < rows; r += gridDim.x * warps) {\
        const float4* w = reinterpret_cast<const float4*>(Wqkv + (size_t)r * H);\
        /* Two chunks of K in flight. Without this the projection runs at     \
           about 1 TB/s and the fusion loses more on the GEMM than it saves    \
           on the kernels it absorbs. */                                       \
        float acc[NB], acc2[NB];                                              \
        _Pragma("unroll") for (int b = 0; b < NB; ++b) {                      \
            acc[b] = 0.f; acc2[b] = 0.f; }                                    \
        int i = lane;                                                         \
        for (; i + 32 < quads; i += 64) {                                     \
            float4 wv = w[i], wv2 = w[i + 32];                                \
            const bf16* wb = reinterpret_cast<const bf16*>(&wv);              \
            const bf16* wb2 = reinterpret_cast<const bf16*>(&wv2);            \
            _Pragma("unroll") for (int b = 0; b < NB; ++b) {                  \
                float4 xv = xs[(size_t)b * quads + i];                        \
                float4 xv2 = xs[(size_t)b * quads + i + 32];                  \
                const bf16* xb = reinterpret_cast<const bf16*>(&xv);          \
                const bf16* xb2 = reinterpret_cast<const bf16*>(&xv2);        \
                _Pragma("unroll") for (int j = 0; j < 8; ++j) {               \
                    acc[b] = fmaf(bf16_to_f32(wb[j]), bf16_to_f32(xb[j]),     \
                                  acc[b]);                                    \
                    acc2[b] = fmaf(bf16_to_f32(wb2[j]), bf16_to_f32(xb2[j]),  \
                                   acc2[b]);                                  \
                }                                                             \
            }                                                                 \
        }                                                                     \
        for (; i < quads; i += 32) {                                          \
            float4 wv = w[i];                                                 \
            const bf16* wb = reinterpret_cast<const bf16*>(&wv);              \
            _Pragma("unroll") for (int b = 0; b < NB; ++b) {                  \
                float4 xv = xs[(size_t)b * quads + i];                        \
                const bf16* xb = reinterpret_cast<const bf16*>(&xv);          \
                _Pragma("unroll") for (int j = 0; j < 8; ++j)                 \
                    acc[b] = fmaf(bf16_to_f32(wb[j]), bf16_to_f32(xb[j]),     \
                                  acc[b]);                                    \
            }                                                                 \
        }                                                                     \
        _Pragma("unroll") for (int b = 0; b < NB; ++b) {                      \
            float a = acc[b] + acc2[b];                                       \
            _Pragma("unroll") for (int off = 16; off; off >>= 1)              \
                a += __shfl_down_sync(0xffffffffu, a, off);                   \
            if (lane == 0) scratch[(size_t)b * rows + r] = f32_to_bf16(a);    \
        }                                                                     \
    }                                                                         \
                                                                              \
    grid_barrier(barrier, barrier + 1, blocks);                               \
                                                                              \
    /* One warp per head, which now holds all 128 of its values. */           \
    const int heads = (QW + 2 * KW) / 128;                                    \
    const int pos = *posp;                                                    \
    for (int unit = blockIdx.x * warps + warp; unit < heads * B;              \
             unit += gridDim.x * warps) {                                     \
        const int b = unit / heads, h = unit % heads;                         \
        const int qheads = QW / 128, kheads = KW / 128;                       \
        const bf16* src = scratch + (size_t)b * rows + (size_t)h * 128;       \
        const int d0 = lane * 4;                                              \
                                                                              \
        if (h >= qheads + kheads) {                                           \
            bf16* dst = vcache + (((size_t)b * NKV + (h - qheads - kheads))   \
                                  * cap + pos) * 128;                         \
            _Pragma("unroll") for (int j = 0; j < 4; ++j)                     \
                dst[d0 + j] = src[d0 + j];                                    \
            continue;                                                         \
        }                                                                     \
                                                                              \
        const bf16* gain = (h < qheads) ? qn : kn;                            \
        float v[4], ss = 0.f;                                                 \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                       \
            v[j] = bf16_to_f32(src[d0 + j]);                                  \
            ss += v[j] * v[j];                                                \
        }                                                                     \
        _Pragma("unroll") for (int off = 16; off; off >>= 1)                  \
            ss += __shfl_xor_sync(0xffffffffu, ss, off);                      \
        float scale = rsqrtf(ss / 128.0f + eps);                              \
                                                                              \
        float normed[4];                                                      \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
            normed[j] = bf16_to_f32(f32_to_bf16(                              \
                bf16_to_f32(f32_to_bf16(v[j] * scale))                        \
                * bf16_to_f32(gain[d0 + j])));                                \
                                                                              \
        /* rotate_half: value d pairs with d +/- 64, which is lane ^ 16. */   \
        float partner[4];                                                     \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
            partner[j] = __shfl_sync(0xffffffffu, normed[j], lane ^ 16);      \
                                                                              \
        const bf16* co = cosT + (size_t)pos * 128;                            \
        const bf16* si = sinT + (size_t)pos * 128;                            \
        bf16 out[4];                                                          \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                       \
            float rot = (lane < 16) ? -partner[j] : partner[j];               \
            float left = bf16_to_f32(f32_to_bf16(                             \
                normed[j] * bf16_to_f32(co[d0 + j])));                        \
            float right = bf16_to_f32(f32_to_bf16(                            \
                rot * bf16_to_f32(si[d0 + j])));                              \
            out[j] = f32_to_bf16(left + right);                               \
        }                                                                     \
                                                                              \
        bf16* dst = (h < qheads)                                              \
            ? (qout + (size_t)b * QW + (size_t)h * 128)                       \
            : (kcache + (((size_t)b * NKV + (h - qheads)) * cap + pos) * 128);\
        _Pragma("unroll") for (int j = 0; j < 4; ++j) dst[d0 + j] = out[j];   \
    }                                                                         \
}

QKV(1, 0, )
QKV(2, 0, )
QKV(4, 0, )
QKV(8, 0, )
QKV(16, 0, )
QKV(1, 1, _pre)
QKV(4, 1, _pre)
"""

_module = None
_ready = None
THREADS = 256


def ready() -> bool:
    global _module, _ready
    if _ready is None:
        try:
            _module = cuda_jit.Module(SOURCE)
            for nb in (1, 2, 4, 8, 16):
                _module.kernel(f"qkv_stage_b{nb}")
            for nb in (1, 4):
                _module.kernel(f"qkv_stage_b{nb}_pre")
            _ready = True
        except Exception as exc:
            print(f"cuda_qkv: unavailable ({type(exc).__name__}: {exc})"[:400])
            _ready = False
    return _ready


class QkvStage:
    """Holds the scratch and barrier this stage needs, sized once."""

    def __init__(self, batch, hidden, q_width, kv_width, n_kv, capacity, eps, device,
                 prenormed=False, grid=0):
        self.batch, self.hidden = batch, hidden
        self.q_width, self.kv_width, self.n_kv = q_width, kv_width, n_kv
        self.capacity, self.eps = capacity, eps
        self.slot = next(nb for nb in (1, 2, 4, 8, 16) if nb >= batch)
        self.rows = q_width + 2 * kv_width

        self.scratch = torch.empty(
            batch, self.rows, dtype=torch.bfloat16, device=device
        )
        self.q_out = torch.empty(batch, q_width, dtype=torch.bfloat16, device=device)
        self.barrier = torch.zeros(2, dtype=torch.int32, device=device)

        suffix = "_pre" if prenormed else ""
        kernel = _module.kernel(f"qkv_stage_b{self.slot}{suffix}")
        self.shared = batch * hidden * 2
        kernel.set_shared(self.shared)
        # A cooperative grid may not exceed what can be resident at once, but
        # it need not equal it: filling every slot leaves each warp about one
        # output row, too little to amortise its own ramp.
        ceiling = kernel.max_blocks(THREADS, self.shared)
        self.grid = min(grid or ceiling, ceiling)
        self.kernel = kernel

    def __call__(self, x, weights, k_cache, v_cache, pos):
        """``weights`` is (norm, qkv, q_norm, k_norm, cos, sin)."""
        norm, qkv, q_norm, k_norm, cos, sin = weights
        self.kernel.cooperative(
            self.grid, THREADS,
            x, norm, qkv, q_norm, k_norm, cos, sin, pos,
            self.scratch, self.q_out, k_cache, v_cache, self.barrier,
            self.batch, self.hidden, self.q_width, self.kv_width,
            self.n_kv, self.capacity, self.grid, self.eps,
        )
        return self.q_out
