"""Can a Triton kernel join the programmatic-launch chain without a rewrite?

1. Launch a Triton kernel normally, keep the CompiledKernel it returns, and
   relaunch its CUfunction through cuda_jit.Kernel.launch_ex with the same
   arguments: outputs must be bit-identical.
2. Compile a Triton kernel that issues griddepcontrol.wait / launch_dependents
   through inline PTX, and check it runs (plain and programmatic).
3. Time a chain: CUDA spin primary (early trigger) -> Triton secondary with
   inline wait, launched plainly versus programmatically through the handle.
"""

import statistics
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/engine")
from kernels import cuda_jit  # noqa: E402
from kernels.norm import _rms_norm_kernel  # noqa: E402

print("triton", triton.__version__, flush=True)


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise("griddepcontrol.wait; mov.u32 $0, $1;", "=r,r", [tl.zeros((1,), tl.int32)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _gdc_trigger():
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents; mov.u32 $0, $1;", "=r,r",
                              [tl.zeros((1,), tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pdl_rms_norm(X, W, Y, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    _gdc_trigger()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols
    _gdc_wait()
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    scale = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * scale).to(tl.bfloat16).to(tl.float32) * w
    tl.store(Y + off, y.to(tl.bfloat16), mask=mask)


SPIN = r"""
extern "C" __global__ void primary(unsigned short* x, long long cycles) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long start = clock64();
    while (clock64() - start < cycles) { }
    if (threadIdx.x == 0 && blockIdx.x == 0) x[0] = x[0];
}
"""


class Handle:
    """A Triton CompiledKernel relaunched through the driver with our launcher."""

    def __init__(self, compiled, num_warps):
        cuda_jit._init()
        self.kernel = cuda_jit.Kernel(__import__("ctypes").c_void_p(compiled.function), compiled.name
                                      if hasattr(compiled, "name") else "triton")
        self.kernel.set_shared(compiled.metadata.shared)
        self.block = 32 * num_warps

    def __call__(self, grid, *args, pdl=False):
        self.kernel.launch_ex(grid, self.block, *args, pdl=pdl)


def clock(fn, reps=50):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / reps * 1e3)
    return statistics.median(ts)


@torch.inference_mode()
def run():
    torch.manual_seed(1)
    rows, n = 16, 2560
    x = torch.randn(rows, n, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y0 = torch.empty_like(x)
    y1 = torch.empty_like(x)
    compiled = _rms_norm_kernel[(rows,)](x, w, y0, n, 1e-6, BLOCK=4096, num_warps=8)
    print("compiled kernel type:", type(compiled).__name__, "function:", hex(compiled.function),
          "shared:", compiled.metadata.shared, flush=True)
    # Non-constexpr args in signature order: X, W, Y, eps. N and BLOCK are constexpr.
    handle = Handle(compiled, 8)
    handle(rows, x, w, y1, 1e-6)
    torch.cuda.synchronize()
    print("handle relaunch bit-exact:", torch.equal(y0, y1), flush=True)

    y2 = torch.empty_like(x)
    pdl_compiled = _pdl_rms_norm[(rows,)](x, w, y2, n, 1e-6, BLOCK=4096, num_warps=8)
    torch.cuda.synchronize()
    print("inline griddepcontrol compiles and runs plainly; bit-exact:", torch.equal(y0, y2), flush=True)
    pdl_handle = Handle(pdl_compiled, 8)
    y3 = torch.empty_like(x)
    pdl_handle(rows, x, w, y3, 1e-6, pdl=True)
    torch.cuda.synchronize()
    print("programmatic relaunch bit-exact:", torch.equal(y0, y3), flush=True)

    spin = cuda_jit.Module(SPIN).kernel("primary")
    cycles = 1980 * 20
    def chain(pdl):
        spin(132, 128, x, cycles)
        if pdl:
            pdl_handle(rows, x, w, y3, 1e-6, pdl=True)
        else:
            handle(rows, x, w, y1, 1e-6)
    print(f"spin(20us) -> rms_norm: plain {clock(lambda: chain(False)):.1f} us   "
          f"programmatic {clock(lambda: chain(True)):.1f} us", flush=True)
    g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        chain(True)
    st.synchronize()
    with torch.cuda.graph(g, stream=st):
        chain(True)
    print(f"graphed programmatic {clock(g.replay):.1f} us; bit-exact after replay: {torch.equal(y0, y3)}", flush=True)


if __name__ == "__main__":
    run()
