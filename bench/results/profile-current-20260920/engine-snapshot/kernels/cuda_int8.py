"""Group-64 W8A8 tensor-core projections, compiled through the NVRTC loader.

Weights retain the existing INT8 values and FP16 scales, repacked into native
m16n8k32 fragments. Activations are packed INT8 with FP32 scales per 64 values.
The normal engine enables only gate/up at batches 9--16 after a warmup race.
Its preceding residual-add/RMSNorm produces the packed activation directly,
preserving every BF16 cast before quantization. Split-K and activation residual
correction remain available to the development probes, not selected by default.
"""
import functools
import torch
import triton
import triton.language as tl
from kernels import cuda_jit


@triton.jit
def _add_norm_quant(X, D, W, RES, Q, S, ROWS: tl.constexpr, N: tl.constexpr,
                    SPLITS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    off = row * N + c
    delta = tl.full((BLOCK,), 0., tl.float32)
    for part in range(SPLITS):
        delta += tl.load(D + part * ROWS * N + off, mask=c < N, other=0.).to(tl.float32)
    delta = delta.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + off, mask=c < N, other=0.).to(tl.float32)
    r = (x + delta).to(tl.bfloat16)
    tl.store(RES + off, r, mask=c < N)
    rf = r.to(tl.float32)
    inv = tl.rsqrt(tl.sum(rf * rf, axis=0) / N + EPS)
    w = tl.load(W + c, mask=c < N, other=0.).to(tl.float32)
    y = ((rf * inv).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(y, (BLOCK // 64, 64))
    scale = tl.maximum(tl.max(tl.abs(grouped), axis=1) / 127., 1.e-20)
    qi = tl.extra.cuda.libdevice.nearbyint(grouped / scale[:, None]).to(tl.int8)
    qi = tl.reshape(qi, (BLOCK,))
    within = c % 64
    physical = (within % 16 // 4) * 16 + (within // 16) * 4 + within % 4
    tl.store(Q + row * N + (c // 64) * 64 + physical, qi, mask=c < N)
    g = tl.arange(0, BLOCK // 64)
    tl.store(S + row * (N // 64) + g, scale, mask=g < N // 64)


def add_norm_quant(residual, delta, weight, eps):
    """Preserve BF16 residual/RMSNorm casts, then emit packed INT8 activations."""
    m, n = residual.shape
    splits = delta.shape[0] if delta.ndim == 3 else 1
    out = torch.empty_like(residual)
    q = torch.empty_like(residual, dtype=torch.int8)
    scale = torch.empty((m, n // 64), device=residual.device, dtype=torch.float32)
    block = triton.next_power_of_2(n)
    _add_norm_quant[(m,)](residual, delta, weight, out, q, scale, m, n, splits,
                         eps, block, num_warps=16, enable_fp_fusion=False)
    return out, (q, scale, q, scale)


@triton.jit
def _quant(X, Q, S, R, RS, K: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    groups = tl.program_id(1) * 4 + tl.arange(0, 4)
    p = tl.arange(0, 64)
    # Contiguous groups of four bytes form native INT8 MMA fragments.
    natural = (p // 16) * 4 + ((p % 16) // 4) * 16 + p % 4
    x = tl.load(X + row * K + groups[:, None] * 64 + natural[None, :],
                mask=groups[:, None] < K // 64, other=0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=1) / 127., 1.e-20)
    q = tl.extra.cuda.libdevice.nearbyint(x / s[:, None]).to(tl.int8)
    off = row * K + groups[:, None] * 64 + p[None, :]
    tl.store(Q + off, q, mask=groups[:, None] < K // 64)
    tl.store(S + row * (K // 64) + groups, s, mask=groups < K // 64)
    if C == 2:
        residual = x - q.to(tl.float32) * s[:, None]
        sr = tl.maximum(tl.max(tl.abs(residual), axis=1) / 127., 1.e-20)
        qr = tl.extra.cuda.libdevice.nearbyint(residual / sr[:, None]).to(tl.int8)
        tl.store(R + off, qr, mask=groups[:, None] < K // 64)
        tl.store(RS + row * (K // 64) + groups, sr, mask=groups < K // 64)


def quantize_activation(x, components=1):
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((m, k // 64), device=x.device, dtype=torch.float32)
    r = torch.empty_like(q) if components == 2 else q
    rs = torch.empty_like(s) if components == 2 else s
    _quant[(m, triton.cdiv(k // 64, 4))](x, q, s, r, rs, k, components, 64,
                                      num_warps=4, enable_fp_fusion=False)
    return q, s, r, rs


@triton.jit
def _reduce(P, Y, WIDTH: tl.constexpr, SPLITS: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    acc = tl.full((256,), 0, tl.float32)
    for part in range(SPLITS):
        acc += tl.load(P + part * WIDTH + i, mask=i < WIDTH, other=0.)
    tl.store(Y + i, acc, mask=i < WIDTH)


class Prepared:
    def __init__(self, source):
        self.rows, self.k = source.rows, source.k
        p = torch.arange(64, device=source.weight.device)
        natural = (p // 16) * 4 + ((p % 16) // 4) * 16 + p % 4
        old = ((natural % 8) // 2) * 16 + (natural // 16) * 4 + ((natural % 16) // 8) * 2 + natural % 2
        cols = (torch.arange(self.k // 64, device=p.device)[:, None] * 64 + old[None, :]).flatten()
        self.weight = (source.row_major()[:, cols] ^ 128).contiguous()
        self.scale = source.scale


SOURCE = r'''
typedef unsigned int u32;
typedef unsigned short bf16;
__device__ __forceinline__ bf16 bf(float f) {
    u32 u=__float_as_uint(f); return (u + 0x7fff + ((u >> 16)&1)) >> 16;
}
__device__ __forceinline__ float fp(bf16 b) { return __uint_as_float((u32)b << 16); }
__device__ __forceinline__ float half_float(unsigned short s) {
    float f; asm("cvt.f32.f16 %0,%1;" : "=f"(f): "h"(s)); return f;
}
__device__ __forceinline__ void mma(int *c, uint4 lo, uint4 hi, uint4 b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(lo.x),"r"(hi.x),"r"(lo.y),"r"(hi.y),"r"(b.x),"r"(b.y));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(lo.z),"r"(hi.z),"r"(lo.w),"r"(hi.w),"r"(b.z),"r"(b.w));
}
extern "C" __global__ void __launch_bounds__(@THREADS@)
run(const unsigned char* __restrict__ W, const unsigned short* __restrict__ WS,
    const unsigned char* __restrict__ X, const float* __restrict__ XS,
    const unsigned char* __restrict__ R, const float* __restrict__ RS,
    void* __restrict__ OUT, int B, int N, int K, int SPLITS, int GPS) {
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5, g=lane>>2, t=lane&3;
    const int part=blockIdx.x % SPLITS, rb=blockIdx.x / SPLITS;
#if @MODE@ == 2
    const int row=(rb*@WARPS@+warp)*8;
    const int lo=row+g, hi=row+N+g;
#else
    const int row=(rb*@WARPS@+warp)*16;
    const int lo=row+g, hi=row+8+g;
#endif
    const int groups=K/64, begin=part*GPS, end=min(groups,begin+GPS);
    extern __shared__ __align__(16) unsigned char xs[];
    const int stride=GPS*64+16;
    float *ss=(float*)(xs+@NT@*8*stride);
    // Stage packed activations and their group scales once per block.
    for(int it=threadIdx.x; it<@NT@*8*GPS*4; it+=@THREADS@) {
        const int n=it/(GPS*4), piece=it%(GPS*4);
        uint4 value=make_uint4(0,0,0,0);
        if(n<B && begin*64+piece*16<K)
            value=*reinterpret_cast<const uint4*>(X+(size_t)n*K+begin*64+piece*16);
        *reinterpret_cast<uint4*>(xs+n*stride+piece*16)=value;
    }
    for(int it=threadIdx.x;it<@NT@*8*GPS;it+=@THREADS@) {
        const int n=it/GPS,j=it%GPS;
        ss[it]=(n<B && begin+j<groups) ? XS[n*groups+begin+j] : 0.f;
    }
    __syncthreads();
    if (row>=N) return;
    float acc[@NT@][4]={};
    uint4 wl[4],wh[4];
    #pragma unroll
    for(int p=0;p<4;++p) {
        wl[p]=wh[p]=make_uint4(0,0,0,0);
        if (begin+p<end) {
            wl[p]=*reinterpret_cast<const uint4*>(W+(size_t)lo*K+(begin+p)*64+t*16);
            wh[p]=*reinterpret_cast<const uint4*>(W+(size_t)hi*K+(begin+p)*64+t*16);
        }
    }
    for(int base=begin;base<end;base+=4) {
        uint4 nl[4],nh[4];
        #pragma unroll
        for(int p=0;p<4;++p) {
            nl[p]=wl[p];nh[p]=wh[p];
            if(base+4+p<end) {
                nl[p]=*reinterpret_cast<const uint4*>(W+(size_t)lo*K+(base+4+p)*64+t*16);
                nh[p]=*reinterpret_cast<const uint4*>(W+(size_t)hi*K+(base+4+p)*64+t*16);
            }
        }
        #pragma unroll
        for(int p=0;p<4;++p) {
        const int j=base+p;
        if(j<end) {
        const float sl=half_float(WS[(size_t)lo*groups+j]);
        const float sh=half_float(WS[(size_t)hi*groups+j]);
        #pragma unroll
        for(int nt=0;nt<@NT@;++nt) {
            const int bn=nt*8+g;
            uint4 x=*reinterpret_cast<const uint4*>(xs+bn*stride+(j-begin)*64+t*16);
            int tmp[4]={0,0,0,0}; mma(tmp,wl[p],wh[p],x);
            #pragma unroll
            for(int c=0;c<2;++c) {
                const int n=nt*8+t*2+c;
                const float sx=ss[n*GPS+j-begin];
                acc[nt][c]=fmaf((float)tmp[c]*sx,sl,acc[nt][c]);
                acc[nt][c+2]=fmaf((float)tmp[c+2]*sx,sh,acc[nt][c+2]);
            }
#if @COMP@ == 2
            uint4 r=make_uint4(0,0,0,0);
            if(bn<B) r=__ldg(reinterpret_cast<const uint4*>(R+(size_t)bn*K+j*64+t*16));
            int tr[4]={0,0,0,0}; mma(tr,wl[p],wh[p],r);
            #pragma unroll
            for(int c=0;c<2;++c) {
                const int n=nt*8+t*2+c;
                const float sx=n<B ? __ldg(RS+n*groups+j) : 0.f;
                acc[nt][c]=fmaf((float)tr[c]*sx,sl,acc[nt][c]);
                acc[nt][c+2]=fmaf((float)tr[c+2]*sx,sh,acc[nt][c+2]);
            }
#endif
        }
        }
        }
        #pragma unroll
        for(int p=0;p<4;++p) { wl[p]=nl[p];wh[p]=nh[p]; }
    }
    #pragma unroll
    for(int nt=0;nt<@NT@;++nt) {
        #pragma unroll
        for(int c=0;c<2;++c) {
            const int n=nt*8+t*2+c;
            if(n<B) {
#if @MODE@ == 1
                float* out=(float*)OUT+(size_t)part*B*N;
                out[(size_t)n*N+lo]=acc[nt][c]; out[(size_t)n*N+hi]=acc[nt][c+2];
#elif @MODE@ == 2
                const float gate=fp(bf(acc[nt][c])), up=fp(bf(acc[nt][c+2]));
                ((bf16*)OUT)[(size_t)n*N+lo]=bf(fp(bf(gate/(1.f+__expf(-gate))))*up);
#else
                ((bf16*)OUT)[(size_t)n*N+lo]=bf(acc[nt][c]);
                ((bf16*)OUT)[(size_t)n*N+hi]=bf(acc[nt][c+2]);
#endif
            }
        }
    }
}
'''


@functools.lru_cache(None)
def module(nt, warps, components, mode):
    src = SOURCE
    for key, value in {'NT': nt, 'WARPS': warps, 'THREADS': warps*32,
                       'COMP': components, 'MODE': mode}.items():
        src = src.replace('@'+key+'@', str(value))
    return cuda_jit.Module(src)


def matmul_quantized(activation, prepared, config=(4, 1), components=1, mode=0):
    x, sx, r, sr = activation
    m, k = x.shape
    n = prepared.rows // 2 if mode == 2 else prepared.rows
    warps, splits = config
    if mode == 2 and splits != 1:
        raise ValueError('SwiGLU requires no split K')
    nt = triton.cdiv(m, 8)
    gps = triton.cdiv(k // 64, splits)
    while nt*8*(gps*68+16) > 190*1024:
        splits += 1
        gps = triton.cdiv(k // 64, splits)
    splits = triton.cdiv(k // 64, gps)
    actual_mode = 1 if splits > 1 else mode
    out = torch.empty((splits, m, n) if splits > 1 else (m, n), device=x.device,
                      dtype=torch.float32 if splits > 1 else torch.bfloat16)
    rows = warps * (8 if mode == 2 else 16)
    kernel = module(nt, warps, components, actual_mode).kernel('run')
    kernel.set_shared(nt*8*(gps*68+16))
    kernel(triton.cdiv(n, rows)*splits, warps*32, prepared.weight, prepared.scale,
           x, sx, r, sr, out, m, n, k, splits, gps)
    if splits > 1:
        reduced = torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
        _reduce[(triton.cdiv(m*n,256),)](out,reduced,m*n,splits,num_warps=4)
        return reduced
    return out


def matmul(x, prepared, config=(4, 1), components=1, mode=0):
    activation = x if isinstance(x, tuple) else quantize_activation(x, components)
    return matmul_quantized(activation, prepared, config, components, mode)
