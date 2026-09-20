"""The small decode kernels between projections, as CUDA driver launches.

RMSNorm, the split-K residual sum and the split-K SwiGLU each move a few
megabytes and cost 2 to 4 us a launch, almost all of it launch latency and the
dependency chain of one wave. As driver launches they can join the
programmatic-dependent-launch chain with the GEMMs: each triggers its
dependents at entry and waits for its producer before its first load, so the
next kernel's launch and prologue overlap this one's body.

The arithmetic follows the Triton kernels these replace step for step: fp32
plane sums in plane order, the same bfloat16 roundings in the same places.
The residual sum and SwiGLU are bit-exact with them; RMSNorm reduces the sum
of squares in a different order, which moves the scale by an ulp at most.
"""

import os

import torch

from . import cuda_jit

PDL = {"off": 0, "late": 1, "on": 2, "early": 2}[os.environ.get("DRYFT_PDL", "early")]

_SOURCE = r"""
typedef unsigned short bf16;
typedef unsigned int u32;

__device__ __forceinline__ float bf2f(bf16 h) { return __uint_as_float(((u32)h) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float round_bf(float f) { return bf2f(f2bf(f)); }

#if @PDL@
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT()
#define PDL_TRIGGER()
#endif

// residual[i] = bf16( x[i] + bf16( sum_s planes[s][i] ) ), one thread per element.
extern "C" __global__ void __launch_bounds__(256)
residual_partials(const bf16* __restrict__ X, const float* __restrict__ P,
                  bf16* __restrict__ R, int width, int splits)
{
    PDL_TRIGGER();
    const int i = blockIdx.x * 256 + threadIdx.x;
    PDL_WAIT();
    if (i >= width) return;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += P[(size_t)s * width + i];
    R[i] = f2bf(bf2f(X[i]) + round_bf(acc));
}

// y = bf16( bf16(x * rsqrt(mean(x^2) + eps)) * w ), one block of @THREADS@ per row.
extern "C" __global__ void __launch_bounds__(@THREADS@)
rms_norm(const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ Y,
         int n, float eps)
{
    PDL_TRIGGER();
    __shared__ float part[@THREADS@ / 32];
    const size_t row = (size_t)blockIdx.x * n;
    PDL_WAIT();
    float x[@PER@];
    float ss = 0.f;
    #pragma unroll
    for (int u = 0; u < @PER@; ++u) {
        const int c = threadIdx.x + u * @THREADS@;
        x[u] = c < n ? bf2f(X[row + c]) : 0.f;
        ss += x[u] * x[u];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < @THREADS@ / 32; ++w) total += part[w];
    const float scale = rsqrtf(total / (float)n + eps);
    #pragma unroll
    for (int u = 0; u < @PER@; ++u) {
        const int c = threadIdx.x + u * @THREADS@;
        if (c < n) Y[row + c] = f2bf(round_bf(x[u] * scale) * bf2f(W[c]));
    }
}

// out[r][c] = bf16( bf16(silu(bf16(gate))) * bf16(up) ) from fp32 planes [splits][rows][2I].
extern "C" __global__ void __launch_bounds__(256)
partial_swiglu(const float* __restrict__ P, bf16* __restrict__ OUT,
               int rows, int inter, int splits)
{
    PDL_TRIGGER();
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    PDL_WAIT();
    if (c >= inter) return;
    float gate = 0.f, up = 0.f;
    const size_t base = (size_t)row * 2 * inter + c;
    const size_t plane = (size_t)rows * 2 * inter;
    for (int s = 0; s < splits; ++s) {
        gate += P[s * plane + base];
        up += P[s * plane + base + inter];
    }
    gate = round_bf(gate);
    up = round_bf(up);
    const float silu = round_bf(gate / (1.f + expf(-gate)));
    OUT[(size_t)row * inter + c] = f2bf(silu * up);
}
"""

_THREADS = 320   # 2560 = 320 x 8; every thread owns eight contiguous-stride elements
_module = None
_ready = None


def _get():
    global _module
    if _module is None:
        src = (_SOURCE.replace("@PDL@", str(PDL)).replace("@THREADS@", str(_THREADS))
               .replace("@PER@", "8"))
        _module = cuda_jit.Module(src)
    return _module


def ready() -> bool:
    global _ready
    if _ready is None:
        try:
            _get().kernel("rms_norm")
            _ready = True
        except Exception as exc:
            print(f"cuda_small: unavailable ({type(exc).__name__}: {exc})"[:300])
            _ready = False
    return _ready


def _kernel(name):
    k = _get().kernel(name)
    k.pdl = bool(PDL)
    return k


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rows, n = x.shape
    if n > _THREADS * 8:
        raise ValueError(f"rms_norm serves rows up to {_THREADS * 8} wide")
    out = torch.empty_like(x)
    _kernel("rms_norm")(rows, _THREADS, x, weight, out, n, float(eps))
    return out


def residual_partials(x: torch.Tensor, partials: torch.Tensor) -> torch.Tensor:
    """``bf16(x + bf16(sum of planes))`` for planes ``[splits, rows, n]``."""
    residual = torch.empty_like(x)
    width = x.numel()
    _kernel("residual_partials")(-(-width // 256), 256, x, partials, residual, width, partials.shape[0])
    return residual


def add_rms_norm_partials_separate(x, partials, weight, eps):
    residual = residual_partials(x, partials)
    return residual, rms_norm(residual, weight, eps)


def swiglu_partials(partials: torch.Tensor) -> torch.Tensor:
    splits, rows, width = partials.shape
    inter = width // 2
    out = torch.empty((rows, inter), device=partials.device, dtype=torch.bfloat16)
    kernel = _kernel("partial_swiglu")
    kernel.launch_2d(-(-inter // 256), rows, 256, partials, out, rows, inter, splits)
    return out
