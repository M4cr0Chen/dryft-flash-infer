"""8-bit weights on the tensor cores, for every decode batch size.

The weights are INT8 with one fp16 scale per 64 values, stored offset by 128
as bytes. E4M3 at the same byte count carries three mantissa bits and put a
4.25-logit gap on the judged token in one public sample of five; INT8 within a
64-wide group has about 3.5x less RMS error and the same bandwidth. The module
keeps its name and the "fp8" family label so the rest of the engine did not
have to move.

The earlier Triton FP8 kernels moved half the bytes of cuBLAS and took longer,
because their dequantisation chain and tiny K tiles left the memory system
idle. This kernel is built the other way round, from the memory side:

* one warp owns sixteen output rows and streams them with 16-byte loads, a
  full group of weights in flight while the previous one is consumed;
* each lane converts its own sixteen E4M3 values with the hardware
  ``cvt.rn.f16x2.e4m3x2`` instruction and feeds them straight into
  ``mma.sync.m16n8k16`` as bfloat16 fragments -- the weight columns are
  pre-permuted at load time so that what a lane loads contiguously is exactly
  what its fragments need, and the activation is permuted the same way while
  it is staged into shared memory;
* the group scale is applied once per 128-wide group to the fp32 partial
  sum, so dequantisation is exact and costs a handful of FMAs per group.

The activation stays bfloat16, so no range trick is needed. Products are
bf16 x bf16 into fp32, as cuBLAS does for the reference, and the output is
rounded to bfloat16 once.

Three epilogues: bfloat16 output, fp32 split-K partials (a small reduce kernel
or the fused add-norm sums them), and SwiGLU for the gate/up projection, where
a warp's rows 0-7 are gate rows and 8-15 the matching up rows, so both halves
of every pair land in the same lane.

NVRTC has no headers, so bfloat16 travels as ``unsigned short``.
"""

import torch

from . import cuda_jit

#: Values per scale. One scale covers one 64-wide block of the weight, which
#: is exactly what a lane's two 16-byte loads cover in a K step.
GROUP = 64

_PRELUDE = r"""
typedef unsigned short bf16;
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ bf16 f32_to_bf16(float f) {
    u32 u = __float_as_uint(f);
    u32 round = ((u >> 16) & 1u) + 0x7fffu;
    return (bf16)((u + round) >> 16);
}

__device__ __forceinline__ float bf16_to_f32(bf16 h) {
    return __int_as_float(((int)(u32)h) << 16);
}

// Two E4M3 values in one 16-bit word -> two bfloat16 in one 32-bit word.
// E4M3 has three mantissa bits, so every step here is exact.
__device__ __forceinline__ u32 fp8x2_to_bf16x2(u32 v) {
    u32 h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((u16)v));
    float lo, hi;
    asm("{\n\t.reg .b16 l, u;\n\tmov.b32 {l, u}, %2;\n\t"
        "cvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, u;\n}"
        : "=f"(lo), "=f"(hi) : "r"(h));
    u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// Two bytes of ``w`` (offset-128 INT8) -> two bfloat16 in one word. The magic
// constant places the byte in the mantissa of 2^23, the subtraction recovers
// the signed integer exactly, and the integer is exact in bfloat16.
__device__ __forceinline__ u32 int8x2_to_bf16x2(u32 w, int lo_sel, int hi_sel) {
    const float lo = __uint_as_float(__byte_perm(w, 0x4B000000u, lo_sel)) - 8388736.0f;
    const float hi = __uint_as_float(__byte_perm(w, 0x4B000000u, hi_sel)) - 8388736.0f;
    u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void f16x2_to_f32(u32 v, float& lo, float& hi) {
    asm("{\n\t.reg .b16 l, u;\n\tmov.b32 {l, u}, %2;\n\t"
        "cvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, u;\n}"
        : "=f"(lo), "=f"(hi) : "r"(v));
}

__device__ __forceinline__ void mma_bf16(float* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 word(const uint4& v, int s) {
    return s == 0 ? v.x : (s == 1 ? v.y : (s == 2 ? v.z : v.w));
}

"""

_TEMPLATE = r"""
// W: [rows, K] offset-128 INT8, columns permuted within every 64-wide block.
// S: [rows, K/64] fp16 block scales; G counts 128-wide pairs of blocks.
// X: [B, K] bfloat16, natural column order.
// MODE 0: OUT bf16 [B, N]     MODE 1: OUT fp32 [SPLITK, B, N]
// MODE 2: OUT bf16 [B, N] where N = I and W holds [gate; up] with 2*I rows.
extern "C" __global__ void __launch_bounds__(@THREADS@)
@NAME@(const unsigned char* __restrict__ W, const u32* __restrict__ S,
       const bf16* __restrict__ X, void* __restrict__ OUT,
       int N, int K, int B, int G, int GPS, int SPLITK)
{
    extern __shared__ __align__(16) bf16 xs[];
    const int split = blockIdx.x % SPLITK;
    const int rowblock = blockIdx.x / SPLITK;
    const int g_begin = split * GPS;
    const int g_end = min(g_begin + GPS, G);
    if (g_begin >= g_end) return;
    const int ngroups = g_end - g_begin;
    const int k0 = g_begin * 128;
    const int stride = GPS * 128 + 8;   // +16 bytes keeps the fragment loads conflict-free

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;

    // @NSHARE@ warps share one row block and split the activation tiles
    // between them; the weights they both need come from L2 the second time.
    const int rowgroup = warp / @NSHARE@;
    const int n0 = (warp % @NSHARE@) * (@NT@ * 8);
#if @MODE@ == 2
    const int i0 = (rowblock * (@WARPS@ / @NSHARE@) + rowgroup) * 8;
    const bool live = i0 < N;
    const int row_lo = live ? i0 + g : 0;
    const int row_hi = live ? N + i0 + g : 0;
#else
    const int rb = (rowblock * (@WARPS@ / @NSHARE@) + rowgroup) * 16;
    const bool live = rb < N;
    const int row_lo = live ? rb + g : 0;
    const int row_hi = live ? rb + 8 + g : 0;
#endif

    const unsigned char* w_lo = W + (size_t)row_lo * K + k0 + t * 16;
    const unsigned char* w_hi = W + (size_t)row_hi * K + k0 + t * 16;
    const u32* s_lo = S + (size_t)row_lo * G;
    const u32* s_hi = S + (size_t)row_hi * G;

    // Two groups of weights per lane are in flight at all times: cur is being
    // consumed while nxt streams in. Issue the first pair before staging X so
    // HBM is busy during the prologue.
    uint4 cur[2][2][2], nxt[2][2][2];   // [group in pair][64-block][row half]
    u32 sc[2][2], nsc[2][2];            // two f16 block scales per group, per row half
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const bool ok = p < ngroups;
        const size_t off = (size_t)p * 128;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            cur[p][j][0] = ok ? *reinterpret_cast<const uint4*>(w_lo + off + j * 64) : make_uint4(0,0,0,0);
            cur[p][j][1] = ok ? *reinterpret_cast<const uint4*>(w_hi + off + j * 64) : make_uint4(0,0,0,0);
        }
        sc[p][0] = ok ? s_lo[g_begin + p] : 0u;
        sc[p][1] = ok ? s_hi[g_begin + p] : 0u;
    }

#if @STAGE@
    // Stage this block's slice of X, permuted the way the fragments want it.
    // Eight independent 16-byte loads per thread are in flight before any of
    // them is scattered into shared memory as four 4-byte stores; a
    // load-store-load loop would serialise on L2 latency instead.
    {
        const int nblk = ngroups * 2;                 // 64-wide blocks in the chunk
        const int items = @NT@ * @NSHARE@ * 8 * nblk * 8;   // eight 16-byte pieces per block row
        for (int base = threadIdx.x; base < items; base += @THREADS@ * 8) {
            uint4 v[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int it = base + u * @THREADS@;
                v[u] = make_uint4(0, 0, 0, 0);
                if (it < items) {
                    const int piece = it & 7;
                    const int rest = it >> 3;
                    const int blk = rest % nblk;
                    const int n = rest / nblk;
                    if (n < B)
                        v[u] = *reinterpret_cast<const uint4*>(
                            X + (size_t)n * K + k0 + blk * 64 + piece * 8);
                }
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int it = base + u * @THREADS@;
                if (it < items) {
                    const int piece = it & 7;
                    const int rest = it >> 3;
                    const int blk = rest % nblk;
                    const int n = rest / nblk;
                    bf16* dst = xs + n * stride + blk * 64;
                    const u32 words[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int k = piece * 8 + q * 2;   // natural even index in the block
                        const int s = k >> 4, r = k & 15;
                        const int tt = (r & 7) >> 1, hi = r >> 3;
                        const int pos = tt * 16 + s * 4 + hi * 2;
                        *reinterpret_cast<u32*>(dst + pos) = words[q];
                    }
                }
            }
        }
    }
    __syncthreads();
#endif
    if (!live) return;

    float acc[@NT@][4];
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }

    for (int base = 0; base < ngroups; base += 2) {
        // Prefetch the pair after this one.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int grp = base + 2 + p;
            const bool ok = grp < ngroups;
            const size_t off = (size_t)grp * 128;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                nxt[p][j][0] = ok ? *reinterpret_cast<const uint4*>(w_lo + off + j * 64) : cur[p][j][0];
                nxt[p][j][1] = ok ? *reinterpret_cast<const uint4*>(w_hi + off + j * 64) : cur[p][j][1];
            }
            nsc[p][0] = ok ? s_lo[g_begin + grp] : 0u;
            nsc[p][1] = ok ? s_hi[g_begin + grp] : 0u;
        }

        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int local = base + p;
            if (local < ngroups) {
                float scl_lo[2], scl_hi[2];
                f16x2_to_f32(sc[p][0], scl_lo[0], scl_lo[1]);
                f16x2_to_f32(sc[p][1], scl_hi[0], scl_hi[1]);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float tmp[@NT@][4];
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) { tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f; }
                    uint4 xb[@NT@][2];
#if @STAGE@
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        const uint4* src = reinterpret_cast<const uint4*>(
                            xs + (n0 + nt * 8 + g) * stride + local * 128 + j * 64 + t * 16);
                        xb[nt][0] = src[0];
                        xb[nt][1] = src[1];
                    }
#else
                    // Straight from the activation in natural order: eight
                    // 4-byte L1-resident loads per tile, no staging pass.
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        const int n = n0 + nt * 8 + g;
                        const bf16* xrow = X + (size_t)n * K + k0 + local * 128 + j * 64 + 2 * t;
                        u32 v[8];
                        #pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            v[2 * s] = (n < B) ? *reinterpret_cast<const u32*>(xrow + 16 * s) : 0u;
                            v[2 * s + 1] = (n < B) ? *reinterpret_cast<const u32*>(xrow + 16 * s + 8) : 0u;
                        }
                        xb[nt][0] = make_uint4(v[0], v[1], v[2], v[3]);
                        xb[nt][1] = make_uint4(v[4], v[5], v[6], v[7]);
                    }
#endif
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const u32 wl = word(cur[p][j][0], s);
                        const u32 wh = word(cur[p][j][1], s);
                        u32 a[4];
                        a[0] = int8x2_to_bf16x2(wl, 0x7650, 0x7651);
                        a[1] = int8x2_to_bf16x2(wh, 0x7650, 0x7651);
                        a[2] = int8x2_to_bf16x2(wl, 0x7652, 0x7653);
                        a[3] = int8x2_to_bf16x2(wh, 0x7652, 0x7653);
                        #pragma unroll
                        for (int nt = 0; nt < @NT@; ++nt) {
                            const u32 b0 = word(xb[nt][s >> 1], (s & 1) * 2);
                            const u32 b1 = word(xb[nt][s >> 1], (s & 1) * 2 + 1);
                            mma_bf16(tmp[nt], a, b0, b1);
                        }
                    }
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        acc[nt][0] = fmaf(tmp[nt][0], scl_lo[j], acc[nt][0]);
                        acc[nt][1] = fmaf(tmp[nt][1], scl_lo[j], acc[nt][1]);
                        acc[nt][2] = fmaf(tmp[nt][2], scl_hi[j], acc[nt][2]);
                        acc[nt][3] = fmaf(tmp[nt][3], scl_hi[j], acc[nt][3]);
                    }
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) { cur[p][j][0] = nxt[p][j][0]; cur[p][j][1] = nxt[p][j][1]; }
            sc[p][0] = nsc[p][0]; sc[p][1] = nsc[p][1];
        }
    }

#if @MODE@ == 0
    bf16* out = reinterpret_cast<bf16*>(OUT);
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = n0 + nt * 8 + 2 * t + c;
            if (n < B) {
                out[(size_t)n * N + row_lo] = f32_to_bf16(acc[nt][c]);
                out[(size_t)n * N + row_hi] = f32_to_bf16(acc[nt][2 + c]);
            }
        }
    }
#elif @MODE@ == 1
    float* out = reinterpret_cast<float*>(OUT) + (size_t)split * B * N;
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = n0 + nt * 8 + 2 * t + c;
            if (n < B) {
                out[(size_t)n * N + row_lo] = acc[nt][c];
                out[(size_t)n * N + row_hi] = acc[nt][2 + c];
            }
        }
    }
#else
    // Reference rounding: gate and up round to bf16 as a Linear output would,
    // silu rounds, and the product rounds once more.
    bf16* out = reinterpret_cast<bf16*>(OUT);
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = n0 + nt * 8 + 2 * t + c;
            if (n < B) {
                const float gb = bf16_to_f32(f32_to_bf16(acc[nt][c]));
                const float ub = bf16_to_f32(f32_to_bf16(acc[nt][2 + c]));
                const float si = bf16_to_f32(f32_to_bf16(gb / (1.0f + __expf(-gb))));
                out[(size_t)n * N + i0 + g] = f32_to_bf16(si * ub);
            }
        }
    }
#endif
}
"""

_FUSED = r"""
// The o and down projections with the residual add and the RMSNorm that
// follow them. All @WARPS@ warps of a block own the same sixteen output rows
// and split K between them; the block reduces through shared memory and
// writes a bfloat16 delta. The last block to finish (an atomic ticket) adds
// the residual, normalises every row, and writes both, so no planes reach
// memory and no second kernel launches.
//
// X: [B, K] bf16 natural order.  RES, DELTA, OUT_RES, OUT_NORM: [B, N] bf16.
// NW: [N] bf16 norm weight.  COUNTER: one int, zero between launches.
extern "C" __global__ void __launch_bounds__(@THREADS@)
@NAME@(const unsigned char* __restrict__ W, const u32* __restrict__ S,
       const bf16* __restrict__ X, const bf16* __restrict__ RES,
       const bf16* __restrict__ NW, bf16* __restrict__ DELTA,
       bf16* __restrict__ OUT_RES, bf16* __restrict__ OUT_NORM,
       int* __restrict__ COUNTER, int N, int K, int B, int G, int GPW, float eps)
{
    extern __shared__ __align__(16) bf16 xs[];   // [WARPS][NT*8][GPW*128 + 8], permuted
    __shared__ float red[@WARPS@][@NT@ * 8][16];
    __shared__ float partial[@WARPS@];
    __shared__ int am_last;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int rb = blockIdx.x * 16;
    const int row_lo = rb + g;
    const int row_hi = rb + 8 + g;

    const int g_begin = warp * GPW;
    const int g_end = min(g_begin + GPW, G);
    const int ngroups = max(0, g_end - g_begin);
    const int k0 = g_begin * 128;
    const int stride = GPW * 128 + 8;
    bf16* xw = xs + (size_t)warp * (@NT@ * 8) * stride;

    const unsigned char* w_lo = W + (size_t)row_lo * K + k0 + t * 16;
    const unsigned char* w_hi = W + (size_t)row_hi * K + k0 + t * 16;
    const u32* s_lo = S + (size_t)row_lo * G;
    const u32* s_hi = S + (size_t)row_hi * G;

    uint4 cur[2][2][2], nxt[2][2][2];
    u32 sc[2][2], nsc[2][2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const bool ok = p < ngroups;
        const size_t off = (size_t)p * 128;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            cur[p][j][0] = ok ? *reinterpret_cast<const uint4*>(w_lo + off + j * 64) : make_uint4(0,0,0,0);
            cur[p][j][1] = ok ? *reinterpret_cast<const uint4*>(w_hi + off + j * 64) : make_uint4(0,0,0,0);
        }
        sc[p][0] = ok ? s_lo[g_begin + p] : 0u;
        sc[p][1] = ok ? s_hi[g_begin + p] : 0u;
    }

#if @STAGE@
    // Each warp stages its own K chunk of the activation, permuted into
    // fragment order, with eight 16-byte loads in flight per lane.
    {
        const int nblk = ngroups * 2;
        const int items = @NT@ * 8 * nblk * 8;
        for (int base = lane; base < items; base += 32 * 8) {
            uint4 v[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int it = base + u * 32;
                v[u] = make_uint4(0, 0, 0, 0);
                if (it < items) {
                    const int piece = it & 7;
                    const int rest = it >> 3;
                    const int blk = rest % nblk;
                    const int n = rest / nblk;
                    if (n < B)
                        v[u] = *reinterpret_cast<const uint4*>(X + (size_t)n * K + k0 + blk * 64 + piece * 8);
                }
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int it = base + u * 32;
                if (it < items) {
                    const int piece = it & 7;
                    const int rest = it >> 3;
                    const int blk = rest % nblk;
                    const int n = rest / nblk;
                    bf16* dst = xw + n * stride + blk * 64;
                    const u32 words[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int k = piece * 8 + q * 2;
                        const int s = k >> 4, r = k & 15;
                        const int tt = (r & 7) >> 1, hi = r >> 3;
                        *reinterpret_cast<u32*>(dst + tt * 16 + s * 4 + hi * 2) = words[q];
                    }
                }
            }
        }
        __syncwarp();
    }
#endif

    float acc[@NT@][4];
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }

    for (int base = 0; base < ngroups; base += 2) {
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int grp = base + 2 + p;
            const bool ok = grp < ngroups;
            const size_t off = (size_t)grp * 128;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                nxt[p][j][0] = ok ? *reinterpret_cast<const uint4*>(w_lo + off + j * 64) : cur[p][j][0];
                nxt[p][j][1] = ok ? *reinterpret_cast<const uint4*>(w_hi + off + j * 64) : cur[p][j][1];
            }
            nsc[p][0] = ok ? s_lo[g_begin + grp] : 0u;
            nsc[p][1] = ok ? s_hi[g_begin + grp] : 0u;
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int local = base + p;
            if (local < ngroups) {
                float scl_lo[2], scl_hi[2];
                f16x2_to_f32(sc[p][0], scl_lo[0], scl_lo[1]);
                f16x2_to_f32(sc[p][1], scl_hi[0], scl_hi[1]);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float tmp[@NT@][4];
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) { tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f; }
                    uint4 xb[@NT@][2];
#if @STAGE@
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        const uint4* src = reinterpret_cast<const uint4*>(
                            xw + (nt * 8 + g) * stride + local * 128 + j * 64 + t * 16);
                        xb[nt][0] = src[0];
                        xb[nt][1] = src[1];
                    }
#else
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        const int n = nt * 8 + g;
                        const bf16* xrow = X + (size_t)n * K + k0 + local * 128 + j * 64 + 2 * t;
                        u32 v[8];
                        #pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            v[2 * s] = (n < B) ? *reinterpret_cast<const u32*>(xrow + 16 * s) : 0u;
                            v[2 * s + 1] = (n < B) ? *reinterpret_cast<const u32*>(xrow + 16 * s + 8) : 0u;
                        }
                        xb[nt][0] = make_uint4(v[0], v[1], v[2], v[3]);
                        xb[nt][1] = make_uint4(v[4], v[5], v[6], v[7]);
                    }
#endif
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const u32 wl = word(cur[p][j][0], s);
                        const u32 wh = word(cur[p][j][1], s);
                        u32 a[4];
                        a[0] = int8x2_to_bf16x2(wl, 0x7650, 0x7651);
                        a[1] = int8x2_to_bf16x2(wh, 0x7650, 0x7651);
                        a[2] = int8x2_to_bf16x2(wl, 0x7652, 0x7653);
                        a[3] = int8x2_to_bf16x2(wh, 0x7652, 0x7653);
                        #pragma unroll
                        for (int nt = 0; nt < @NT@; ++nt) {
                            const u32 b0 = word(xb[nt][s >> 1], (s & 1) * 2);
                            const u32 b1 = word(xb[nt][s >> 1], (s & 1) * 2 + 1);
                            mma_bf16(tmp[nt], a, b0, b1);
                        }
                    }
                    #pragma unroll
                    for (int nt = 0; nt < @NT@; ++nt) {
                        acc[nt][0] = fmaf(tmp[nt][0], scl_lo[j], acc[nt][0]);
                        acc[nt][1] = fmaf(tmp[nt][1], scl_lo[j], acc[nt][1]);
                        acc[nt][2] = fmaf(tmp[nt][2], scl_hi[j], acc[nt][2]);
                        acc[nt][3] = fmaf(tmp[nt][3], scl_hi[j], acc[nt][3]);
                    }
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) { cur[p][j][0] = nxt[p][j][0]; cur[p][j][1] = nxt[p][j][1]; }
            sc[p][0] = nsc[p][0]; sc[p][1] = nsc[p][1];
        }
    }

    // Each warp's K-slice of the sixteen rows, then one fixed-order sum.
    #pragma unroll
    for (int nt = 0; nt < @NT@; ++nt) {
        red[warp][nt * 8 + 2 * t][g] = acc[nt][0];
        red[warp][nt * 8 + 2 * t + 1][g] = acc[nt][1];
        red[warp][nt * 8 + 2 * t][8 + g] = acc[nt][2];
        red[warp][nt * 8 + 2 * t + 1][8 + g] = acc[nt][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < B * 16; i += @THREADS@) {
        const int n = i >> 4, r = i & 15;
        float d = 0.f;
        #pragma unroll
        for (int w = 0; w < @WARPS@; ++w) d += red[w][n][r];
        DELTA[(size_t)n * N + rb + r] = f32_to_bf16(d);
    }
#if @TAIL@ == 0
    return;
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int ticket = atomicAdd(COUNTER, 1);
        am_last = (ticket == (int)gridDim.x - 1);
    }
    __syncthreads();
    if (!am_last) return;
#if @TAIL@ == 2
    if (threadIdx.x == 0) *COUNTER = 0;
    return;
#endif
    __threadfence();

    // Residual add, then RMSNorm, every row of the batch at once: each row is
    // shared by WPR warps, each warp owns N / WPR columns. Rounding as the
    // reference: the add rounds to bf16, the normalised value rounds to bf16
    // before the weight, and the product rounds once more.
    const int wpr = @WARPS@ / min(@WARPS@, B) ;           // warps per row
    const int rows_per_pass = @WARPS@ / wpr;
    const int seg_len = N / wpr;
    const int passes = (B + rows_per_pass - 1) / rows_per_pass;
    for (int pass = 0; pass < passes; ++pass) {
        const int n = pass * rows_per_pass + warp / wpr;
        const int seg = warp % wpr;
        const bool mine = n < B;
        float sumsq = 0.f;
        if (mine) {
            const size_t rowoff = (size_t)n * N + seg * seg_len;
            #pragma unroll 8
            for (int c = lane; c < seg_len; c += 32) {
                const float x = bf16_to_f32(RES[rowoff + c]);
                const float d = bf16_to_f32(__ldcg(reinterpret_cast<const u16*>(DELTA) + rowoff + c));
                const bf16 r16 = f32_to_bf16(x + d);
                OUT_RES[rowoff + c] = r16;
                const float r = bf16_to_f32(r16);
                sumsq = fmaf(r, r, sumsq);
            }
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
        if (lane == 0) partial[warp] = sumsq;
        __syncthreads();
        float total = 0.f;
        for (int w = 0; w < wpr; ++w) total += partial[(warp / wpr) * wpr + w];
        __syncthreads();
        if (mine) {
            const float scale = rsqrtf(total / (float)N + eps);
            const size_t rowoff = (size_t)n * N + seg * seg_len;
            #pragma unroll 8
            for (int c = lane; c < seg_len; c += 32) {
                const float r = bf16_to_f32(OUT_RES[rowoff + c]);
                const float y = bf16_to_f32(f32_to_bf16(r * scale)) * bf16_to_f32(NW[seg * seg_len + c]);
                OUT_NORM[rowoff + c] = f32_to_bf16(y);
            }
        }
    }
    if (threadIdx.x == 0) *COUNTER = 0;
}
"""

_REDUCE = r"""
extern "C" __global__ void reduce_partials(const float* __restrict__ P,
                                           unsigned short* __restrict__ OUT,
                                           int width, int splits)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += P[(size_t)s * width + i];
    unsigned int u = __float_as_uint(acc);
    unsigned int round = ((u >> 16) & 1u) + 0x7fffu;
    OUT[i] = (unsigned short)((u + round) >> 16);
}
"""

_WARPS = (1, 2, 4, 8)
_NTS = (1, 2, 4, 8)
_MODES = (0, 1, 2)


_STAGES = (1,)


def _name(warps: int, nt: int, mode: int, stage: int = 1, nshare: int = 1) -> str:
    return f"fp8_mma_w{warps}_n{nt}_m{mode}_s{stage}_r{nshare}"


_NSHARES = (1, 2, 4)


def _source(nt: int, nshare: int) -> str:
    """Every warp count and epilogue for one tile count and row-block sharing.

    A process serves one batch size (plus its verification widths, which
    share the tile count up to eight rows), so compiling the other tile counts
    would spend load budget on kernels that never launch.
    """
    parts = [_PRELUDE]
    if (nt, nshare) == (_NTS[0], 1):
        parts.append(_REDUCE)
    for warps in _WARPS:
        if warps % nshare:
            continue
        for mode in _MODES:
            for stage in _STAGES:
                parts.append(
                    _TEMPLATE.replace("@NAME@", _name(warps, nt, mode, stage, nshare))
                    .replace("@WARPS@", str(warps))
                    .replace("@THREADS@", str(warps * 32))
                    .replace("@NT@", str(nt))
                    .replace("@NSHARE@", str(nshare))
                    .replace("@MODE@", str(mode))
                    .replace("@STAGE@", str(stage))
                )
    return "\n".join(parts)


_modules = {}
_ready = None
_FUSED_ROWS = 2560   # the hidden width; the fused epilogue normalises rows of this length


#: Bisection switches for the fused kernel: stage None picks by shared size;
#: tail 1 is the real kernel, 0 stops after the delta, 2 stops after the ticket.
FUSED_DEBUG = {"stage": None, "tail": 1}


def _fused_name(warps: int, nt: int, stage: int, tail: int = 1) -> str:
    return f"fp8_fused_w{warps}_n{nt}_s{stage}_t{tail}"


def _fused_module(warps: int, nt: int, stage: int, tail: int = 1):
    key = ("fused", warps, nt, stage, tail)
    if key not in _modules:
        threads = warps * 32
        source = _PRELUDE + _FUSED.replace("@NAME@", _fused_name(warps, nt, stage, tail)) \
            .replace("@WARPS@", str(warps)).replace("@THREADS@", str(threads)) \
            .replace("@NT@", str(nt)).replace("@STAGE@", str(stage)).replace("@TAIL@", str(tail))
        _modules[key] = cuda_jit.Module(source)
    return _modules[key]


def _module_for(nt: int, nshare: int = 1):
    if (nt, nshare) not in _modules:
        _modules[(nt, nshare)] = cuda_jit.Module(_source(nt, nshare))
    return _modules[(nt, nshare)]


def ready() -> bool:
    """Compile the smallest variant once. False means this runtime cannot JIT CUDA."""
    global _ready
    if _ready is None:
        try:
            _module_for(_NTS[0], 1).kernel("reduce_partials")
            _ready = True
        except Exception as exc:
            print(f"cuda_fp8: unavailable ({type(exc).__name__}: {exc})"[:400])
            _ready = False
    return _ready


def _permutation(device) -> torch.Tensor:
    """Natural column index held at each permuted position of a 64-wide block."""
    order = []
    for t in range(4):
        for s in range(4):
            for u in (0, 1, 8, 9):
                order.append(16 * s + 2 * t + u)
    return torch.tensor(order, dtype=torch.int64, device=device)


def quantize(weight: torch.Tensor, group: int = GROUP):
    """``[out, in]`` bfloat16 -> offset-128 INT8 bytes plus ``[out, in/group]`` fp16 scales.

    Symmetric, round to nearest, one scale per 64 values from the block maximum.
    """
    out, inner = weight.shape
    if inner % group:
        raise ValueError(f"{inner} is not a multiple of {group}")
    tiles = weight.float().view(out, inner // group, group)
    # The scale the kernel multiplies by is the fp16 one, so quantise against
    # that. Nudging it up by one fp16 ulp before rounding keeps the block
    # maximum from clipping when the conversion rounds down. A least-squares
    # grid search over the scale bought about 1% error and cost ten seconds
    # of load budget in kernel launches; it is not worth it.
    scale = (tiles.abs().amax(dim=2, keepdim=True) / 127.0).clamp(min=1e-12)
    scale = (scale * (1.0 + 2.0 ** -10)).to(torch.float16)
    q = torch.round(tiles / scale.float()).clamp(-127.0, 127.0)
    packed = (q + 128.0).to(torch.uint8).view(out, inner).contiguous()
    return packed, scale.squeeze(2).contiguous()


def dequantize(packed) -> torch.Tensor:
    """The bfloat16 weight the kernel effectively multiplies by."""
    weight, scale = packed
    rows, k = weight.shape
    blocks = scale.shape[1]
    q = weight.float() - 128.0
    return (q.view(rows, blocks, k // blocks) * scale.float()[:, :, None]).view(rows, k).to(torch.bfloat16)


class Prepared:
    """An INT8 weight in fragment order plus fp16 block scales."""

    __slots__ = ("weight", "scale", "rows", "k", "groups")

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor):
        rows, k = packed.shape
        if k % 128:
            raise ValueError(f"K={k} is not a multiple of 128")
        if rows % 16:
            raise ValueError(f"rows={rows} is not a multiple of 16")
        if tuple(scale.shape) != (rows, k // GROUP):
            raise ValueError("scales do not match a 64-wide blocking")
        perm = _permutation(packed.device)
        cols = (torch.arange(k // 64, device=packed.device)[:, None] * 64 + perm[None, :]).flatten()
        self.weight = packed.view(torch.uint8)[:, cols].contiguous()
        # Two adjacent f16 block scales form the u32 the kernel loads per 128 group.
        self.scale = scale.to(torch.float16).contiguous()
        self.rows, self.k = rows, k
        self.groups = k // 128

    def bytes_moved(self) -> int:
        return self.weight.numel() + self.scale.numel() * 2


def prepare(packed) -> Prepared:
    """``packed`` is ``(uint8 weight, fp16 scales)`` from ``quantize``."""
    weight, scale = packed
    return Prepared(weight, scale)


def _ntiles(batch: int, nshare: int = 1) -> int:
    """Tiles per warp when ``nshare`` warps split the batch's tiles."""
    tiles = -(-batch // 8)
    per_warp = -(-tiles // nshare)
    for nt in _NTS:
        if per_warp <= nt:
            return nt
    raise ValueError(f"batch {batch} exceeds {_NTS[-1] * 8 * nshare} rows")


def _shared_bytes(nt: int, gps: int, nshare: int = 1) -> int:
    return nt * nshare * 8 * (gps * 128 + 8) * 2


#: Hopper allows up to 227 KiB of dynamic shared memory per block.
_SHARED_LIMIT = 200 * 1024


def _plan(prepared: Prepared, batch: int, config):
    """Resolve ``(warps, splitk[, stage[, nshare]])`` into a launch, growing split-K to fit shared."""
    warps, splitk = config[0], config[1]
    stage = config[2] if len(config) > 2 else 1
    nshare = config[3] if len(config) > 3 else 1
    if warps % nshare:
        raise ValueError("warps sharing a row block must divide the block's warps")
    nt = _ntiles(batch, nshare)
    groups = prepared.groups
    if splitk == "pairs":
        splitk = -(-groups // 2)
    splitk = max(1, min(splitk, groups))
    gps = -(-groups // splitk)
    if stage:
        while _shared_bytes(nt, gps, nshare) > _SHARED_LIMIT and gps > 1:
            splitk += 1
            gps = -(-groups // splitk)
    splitk = -(-groups // gps)
    return warps, nt, splitk, gps, stage, nshare


def _launch(mode, prepared, x, out, n_eff, config):
    batch, k = x.shape
    if k != prepared.k:
        raise ValueError("activation width does not match the weight")
    warps, nt, splitk, gps, stage, nshare = _plan(prepared, batch, config)
    if mode == 2 and splitk != 1:
        raise ValueError("the SwiGLU epilogue needs the whole reduction in one block")
    rows_per_block = (warps // nshare) * (8 if mode == 2 else 16)
    rowblocks = -(-n_eff // rows_per_block)
    kernel = _module_for(nt, nshare).kernel(_name(warps, nt, mode, stage, nshare))
    kernel.set_shared(_shared_bytes(nt, gps, nshare) if stage else 0)
    kernel(rowblocks * splitk, warps * 32,
           prepared.weight, prepared.scale, x, out,
           n_eff, k, batch, prepared.groups, gps, splitk)
    return splitk


def splits(prepared: Prepared, batch: int, config=(4, 1)) -> int:
    """How many fp32 planes ``matmul`` would reduce for this shape and config."""
    return _plan(prepared, batch, config)[2]


def matmul_partials(x: torch.Tensor, prepared: Prepared, config=(4, 1)) -> torch.Tensor:
    """``x @ W.T`` as fp32 split-K planes ``[splits, B, N]``; sum them to finish."""
    batch = x.shape[0]
    warps, nt, splitk, gps, stage, nshare = _plan(prepared, batch, config)
    out = torch.empty((splitk, batch, prepared.rows), dtype=torch.float32, device=x.device)
    _launch(1, prepared, x, out, prepared.rows, (warps, splitk, stage, nshare))
    return out


def matmul(x: torch.Tensor, prepared: Prepared, config=(4, 1)) -> torch.Tensor:
    """``x @ W.T`` in bfloat16, ``x`` of shape ``[B, K]`` with ``B <= 64``."""
    batch = x.shape[0]
    warps, nt, splitk, gps, stage, nshare = _plan(prepared, batch, config)
    out = torch.empty((batch, prepared.rows), dtype=torch.bfloat16, device=x.device)
    if splitk == 1:
        _launch(0, prepared, x, out, prepared.rows, (warps, 1, stage, nshare))
        return out
    partial = torch.empty((splitk, batch, prepared.rows), dtype=torch.float32, device=x.device)
    _launch(1, prepared, x, partial, prepared.rows, (warps, splitk, stage, nshare))
    width = batch * prepared.rows
    reduce = _module_for(_NTS[0], 1).kernel("reduce_partials")
    reduce(-(-width // 256), 256, partial, out, width, splitk)
    return out


def add_norm_scratch(batch: int, rows: int, device):
    """The delta buffer and ticket counter ``matmul_add_norm`` needs, per call site."""
    return (torch.empty((batch, rows), dtype=torch.bfloat16, device=device),
            torch.zeros(1, dtype=torch.int32, device=device))


def matmul_add_norm(x: torch.Tensor, prepared: Prepared, residual: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, scratch, warps: int = 8):
    """``residual + x @ W.T`` in bfloat16 and its RMSNorm, in one launch.

    ``x`` is ``[B, K]``, ``residual`` ``[B, N]`` with ``N`` the weight's rows;
    returns ``(new_residual, normed)``. ``scratch`` comes from
    ``add_norm_scratch`` for this batch and is reused across layers.
    """
    batch, k = x.shape
    n = prepared.rows
    if k != prepared.k or n % 16 or n != _FUSED_ROWS:
        raise ValueError("the fused add-norm serves the hidden-width projections")
    if tuple(residual.shape) != (batch, n) or residual.dtype != torch.bfloat16:
        raise ValueError("residual must be bfloat16 [batch, rows]")
    nt = _ntiles(batch)
    if nt > 2:
        raise ValueError("the fused add-norm serves up to sixteen rows")
    delta, counter = scratch
    if tuple(delta.shape) != (batch, n):
        raise ValueError("scratch does not match this batch")
    out_res = torch.empty_like(residual)
    out_norm = torch.empty_like(residual)
    gpw = -(-prepared.groups // warps)
    shared = warps * nt * 8 * (gpw * 128 + 8) * 2
    stage = 1 if shared <= _SHARED_LIMIT else 0
    if FUSED_DEBUG["stage"] is not None and shared <= 227 * 1024:
        stage = FUSED_DEBUG["stage"]
    tail = FUSED_DEBUG["tail"]
    kernel = _fused_module(warps, nt, stage, tail).kernel(_fused_name(warps, nt, stage, tail))
    kernel.set_shared(shared if stage else 0)
    kernel(n // 16, warps * 32,
           prepared.weight, prepared.scale, x, residual, norm_weight, delta,
           out_res, out_norm, counter, n, k, batch, prepared.groups, gpw, float(eps))
    return out_res, out_norm


#: Warps per block for the fused add-norm GEMM; each takes a slice of K.
ADD_NORM_CONFIGS = (4, 8)


def gate_up_swiglu(x: torch.Tensor, prepared: Prepared, config=(4, 1)) -> torch.Tensor:
    """``silu(gate) * up`` from a fused ``[gate; up]`` FP8 weight, never writing 2*I."""
    batch = x.shape[0]
    inter = prepared.rows // 2
    out = torch.empty((batch, inter), dtype=torch.bfloat16, device=x.device)
    stage = config[2] if len(config) > 2 else 1
    nshare = config[3] if len(config) > 3 else 1
    _launch(2, prepared, x, out, inter, (config[0], 1, stage, nshare))
    return out


#: (warps per block, K splits). Swept at warmup like every other projection.
#: (warps per block, K splits, staged activation). Staged reads the block's
#: activation slice into shared memory once; unstaged reads fragments through
#: L1 from the activation in place and never synchronises.
#: The configurations that won a projection somewhere in the sweeps over
#: batches 1 to 32 and the five shapes, and nothing else: every entry costs a
#: graph capture per projection at warmup, inside a run budget of 15 minutes.
_BASE = [
    (8, 1), (8, 2), (8, 4),
    (4, 2), (4, 4), (4, 8), (4, 16),
    (2, 8), (1, 8),
]
CONFIGS = [(w, k, 1, 1) for w, k in _BASE]
# Two or four warps sharing a row block (``nshare`` > 1) were measured at
# batches 32 and 64 and lost to the single-warp layout on every shape: the
# warps' duplicate weight loads both miss L2, and staging the wider activation
# per block costs more than the register relief buys. The machinery stays for
# experiments; the race does not spend warmup on it.
SWIGLU_CONFIGS = [(w, 1, 1, 1) for w in (4, 8)]


def applicable(config, batch: int, swiglu: bool = False) -> bool:
    """Whether a config is worth racing for this batch: sharing needs tiles to split."""
    nshare = config[3] if len(config) > 3 else 1
    if swiglu and config[1] != 1:
        return False
    return nshare == 1 or -(-batch // 8) >= nshare
