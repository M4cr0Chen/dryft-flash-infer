"""Does programmatic dependent launch engage on this driver, eagerly and under graph capture?

Kernel A spins ~T us and triggers at its start. Kernel B spins ~T us of
independent work, waits, then does a little dependent work. Serial: 2T.
With PDL engaged: about T plus the dependent tail.
"""

import ctypes
import statistics
import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_jit  # noqa: E402

SRC = r"""
__device__ __forceinline__ void spin(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) { }
}
extern "C" __global__ void primary(int* flag, long long cycles, int trigger_early) {
    if (trigger_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    spin(cycles);
    if (threadIdx.x == 0) flag[blockIdx.x] = 1;
    if (!trigger_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
extern "C" __global__ void secondary(const int* flag, int* out, long long cycles, int use_wait) {
    spin(cycles);                       // independent prologue
    if (use_wait) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (threadIdx.x == 0) out[blockIdx.x] = flag[blockIdx.x] + 1;
}
"""


def clock(fn, reps=20):
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


def run():
    v = ctypes.c_int()
    cuda_jit._init()
    cuda_jit._driver.cuDriverGetVersion(ctypes.byref(v))
    print(f"driver {v.value}", flush=True)
    m = cuda_jit.Module(SRC)
    a, b = m.kernel("primary"), m.kernel("secondary")
    flag = torch.zeros(132, dtype=torch.int32, device="cuda")
    out = torch.zeros(132, dtype=torch.int32, device="cuda")
    cycles = 1980 * 50   # ~50 us at 1.98 GHz

    def pair(pdl, early, wait=True):
        a(132, 128, flag, cycles, int(early))
        if pdl:
            b.launch_ex(132, 128, flag, out, cycles, int(wait), pdl=True)
        else:
            b(132, 128, flag, out, cycles, int(wait))

    for label, pdl, early in (("serial", False, False), ("PDL, late trigger", True, False),
                              ("PDL, early trigger", True, True)):
        eager = clock(lambda: pair(pdl, early))
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            pair(pdl, early)
        st.synchronize()
        with torch.cuda.graph(g, stream=st):
            pair(pdl, early)
        graphed = clock(g.replay)
        torch.cuda.synchronize()
        ok = bool((out == 2).all().item())
        print(f"{label:20s} eager {eager:7.1f} us   graph {graphed:7.1f} us   correct {ok}", flush=True)


if __name__ == "__main__":
    run()
