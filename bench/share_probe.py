"""Warps sharing one weight tile through the ring, at the wide batches.

Every race configuration is checked bit-exact against the row-major kernel at
the same warp count and split, then the best single-warp configuration is
timed against the best shared one. Registers and resident blocks per SM come
from the driver.
"""

import ctypes
import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8, cuda_jit  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560),
          ("down", 2560, 9728), ("lm_head", 151936, 2560)]


def registers(kernel):
    n = ctypes.c_int()
    cuda_jit._driver.cuFuncGetAttribute(ctypes.byref(n), 4, kernel._handle)
    return n.value


def occupancy(prepared, batch, cfg, mode):
    warps, nt, splitk, gps, stage, nshare, ring = cuda_fp8._plan(prepared, batch, cfg)
    if ring:
        kernel = cuda_fp8._ring_module(nt, ring, prepared.tiled, prepared.bits, nshare).kernel(
            cuda_fp8._name(warps, nt, mode, 1, nshare, ring, prepared.tiled, prepared.bits))
    else:
        kernel = cuda_fp8._module_for(nt, nshare, prepared.tiled, prepared.bits).kernel(
            cuda_fp8._name(warps, nt, mode, stage, nshare, 0, prepared.tiled, prepared.bits))
    shared = cuda_fp8._shared_bytes(nt, gps, nshare, warps, ring, prepared.bits)
    return registers(kernel), kernel.max_blocks(warps * 32, shared) // 132, splitk, nt


def runner(x, prepared, cfg):
    splitk = cuda_fp8.splits(prepared, x.shape[0], cfg)
    if splitk > 1:
        return lambda p, c=cfg: cuda_fp8.matmul_partials(x, p, config=c)
    return lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c)


@torch.inference_mode()
def run(batches):
    torch.manual_seed(13)
    assert cuda_fp8.ready()
    for name, n, k in SHAPES:
        count = 1 if name == "lm_head" else 12
        packed = [cuda_fp8.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02)
                  for _ in range(count)]
        rowmajor = cuda_fp8.prepare(packed[0], tiled=False)
        tiled = [cuda_fp8.prepare(p, tiled=True) for p in packed]
        del packed
        reps = 1 if name == "lm_head" else 2
        for batch in batches:
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            times = {}
            for cfg in cuda_fp8.CONFIGS:
                if not cuda_fp8.applicable(cfg, batch):
                    continue
                plan = cuda_fp8._plan(tiled[0], batch, cfg)
                want = runner(x, rowmajor, (plan[0], plan[2], 1, 1, 0))(rowmajor)
                got = runner(x, tiled[0], cfg)(tiled[0])
                if not torch.equal(want, got):
                    raise AssertionError(f"{name} b{batch} {cfg} differs from row-major: "
                                         f"{(want.float() - got.float()).abs().max().item()}")
                times[cfg] = time_calls(runner(x, tiled[0], cfg), tiled, reps=reps, trials=3) * 1e3
            single = min((c for c in times if c[3] == 1), key=times.get)
            shared = min((c for c in times if c[3] > 1), key=times.get)
            mode = 1 if cuda_fp8.splits(tiled[0], batch, shared) > 1 else 0
            print(f"{name:8s} b{batch:<3d} single {times[single]:6.1f}us {single} | shared {times[shared]:6.1f}us {shared} "
                  f"x{times[single]/times[shared]:.2f} regs/blocks/split/nt {occupancy(tiled[0], batch, shared, mode)}",
                  flush=True)
            if name == "gate_up":
                sw = {}
                for cfg in cuda_fp8.SWIGLU_CONFIGS:
                    if not cuda_fp8.applicable(cfg, batch, swiglu=True):
                        continue
                    want = cuda_fp8.gate_up_swiglu(x, rowmajor, (cfg[0], 1, 1, 1, 0))
                    got = cuda_fp8.gate_up_swiglu(x, tiled[0], cfg)
                    if not torch.equal(want, got):
                        raise AssertionError(f"swiglu b{batch} {cfg} differs")
                    sw[cfg] = time_calls(lambda p, c=cfg: cuda_fp8.gate_up_swiglu(x, p, config=c), tiled, reps=reps, trials=3) * 1e3
                s1 = min((c for c in sw if c[3] == 1), key=sw.get)
                s2 = min((c for c in sw if c[3] > 1), key=sw.get)
                print(f"    swiglu single {sw[s1]:6.1f}us {s1} | shared {sw[s2]:6.1f}us {s2} x{sw[s1]/sw[s2]:.2f}", flush=True)
        del rowmajor, tiled
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run([int(b) for b in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["16", "32", "64"])])
