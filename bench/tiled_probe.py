"""Weight layouts for the decode GEMM: row-major, tiled, and tiled with a cp.async ring.

Fresh process on the H100. Every variant is checked bit-exact against the
row-major kernel at the same warp count and split, then the best configuration
of each is timed on the real shapes, against a kernel that only streams the
bytes. Registers and resident blocks per SM are reported for the winners.
"""

import ctypes
import json
import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8, cuda_jit  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

STREAM = r"""
extern "C" __global__ void __launch_bounds__(256)
stream(const uint4* __restrict__ p, long chunks, int* flag) {
    unsigned int acc = 0;
    long i = (long)blockIdx.x * 256 * 8 + threadIdx.x;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        const long idx = i + (long)u * 256;
        if (idx < chunks) { const uint4 v = p[idx]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x9e3779b9u) flag[1] = (int)acc;
}
"""

SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560),
          ("down", 2560, 9728), ("lm_head", 151936, 2560)]


def registers(kernel):
    n = ctypes.c_int()
    cuda_jit._driver.cuFuncGetAttribute(ctypes.byref(n), 4, kernel._handle)
    return n.value


def runner(x, prepared, cfg):
    """The call the engine would make: planes when K splits, bf16 otherwise."""
    splitk = cuda_fp8.splits(prepared, x.shape[0], cfg)
    if splitk > 1:
        return lambda p, c=cfg: cuda_fp8.matmul_partials(x, p, config=c)
    return lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c)


def occupancy(prepared, batch, cfg, mode):
    warps, nt, splitk, gps, stage, nshare, ring = cuda_fp8._plan(prepared, batch, cfg)
    if ring:
        kernel = cuda_fp8._ring_module(nt, ring, prepared.tiled).kernel(
            cuda_fp8._name(warps, nt, mode, 1, 1, ring, prepared.tiled))
    else:
        kernel = cuda_fp8._module_for(nt, nshare, prepared.tiled).kernel(
            cuda_fp8._name(warps, nt, mode, stage, nshare, 0, prepared.tiled))
    shared = cuda_fp8._shared_bytes(nt, gps, nshare, warps, ring)
    return registers(kernel), kernel.max_blocks(warps * 32, shared) // 132, splitk


def sweep(x, rowmajor, candidates, configs, reps, trials, swiglu=False):
    """Best config over ``candidates`` (a list of Prepared), each checked against row-major."""
    best = (float("inf"), None)
    for cfg in configs:
        if not cuda_fp8.applicable(cfg, x.shape[0], swiglu):
            continue
        try:
            plan = cuda_fp8._plan(candidates[0], x.shape[0], cfg)
            if swiglu:
                if plan[2] != 1:
                    continue
                want = cuda_fp8.gate_up_swiglu(x, rowmajor, (plan[0], 1, 1, 1))
                got = cuda_fp8.gate_up_swiglu(x, candidates[0], cfg)
                fn = lambda p, c=cfg: cuda_fp8.gate_up_swiglu(x, p, config=c)
            else:
                want = runner(x, rowmajor, (plan[0], plan[2], 1, 1))(rowmajor)
                got = runner(x, candidates[0], cfg)(candidates[0])
                fn = runner(x, candidates[0], cfg)
        except ValueError:
            continue
        if not torch.equal(want, got):
            diff = (want.float() - got.float()).abs().max().item()
            raise AssertionError(f"{cfg} tiled={candidates[0].tiled} differs from row-major: max {diff}")
        us = time_calls(fn, candidates, reps=reps, trials=trials) * 1e3
        if us < best[0]:
            best = (us, cfg)
    return best


@torch.inference_mode()
def run(batches, shapes):
    torch.manual_seed(7)
    assert cuda_fp8.ready()
    stream = cuda_jit.Module(STREAM).kernel("stream")
    flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    ring_cfgs = [(w, k, 1, 1, r) for r in cuda_fp8.RING_DEPTHS for w in cuda_fp8._RING_WARPS
                 for k in (1, 2, 4, 8, 16)]
    ring_sw = [(w, 1, 1, 1, r) for r in cuda_fp8.RING_DEPTHS for w in cuda_fp8._RING_WARPS]
    rows = []
    for name, n, k in SHAPES:
        if shapes and name not in shapes:
            continue
        count = 1 if name == "lm_head" else 12
        packed = [cuda_fp8.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02)
                  for _ in range(count)]
        rowmajor = [cuda_fp8.prepare(p, tiled=False) for p in packed]
        tiled = [cuda_fp8.prepare(p, tiled=True) for p in packed]
        del packed
        assert torch.equal(tiled[0].row_major(), rowmajor[0].weight), "row_major() round trip"
        nbytes = tiled[0].bytes_moved()
        chunks = tiled[0].weight.numel() // 16
        grid = -(-chunks // (256 * 8))
        reps = 1 if name == "lm_head" else 2
        stream_us = time_calls(lambda p: stream(grid, 256, p.weight, chunks, flag), tiled, reps=reps, trials=5) * 1e3
        for batch in batches:
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            old = sweep(x, rowmajor[0], rowmajor, cuda_fp8.CONFIGS, reps, 3)
            til = sweep(x, rowmajor[0], tiled, cuda_fp8.CONFIGS, reps, 3)
            rng = sweep(x, rowmajor[0], tiled, ring_cfgs, reps, 3)
            best = min(til, rng)
            mode = 1 if cuda_fp8.splits(tiled[0], batch, best[1]) > 1 else 0
            row = {"projection": name, "batch": batch, "mb": nbytes / 1e6, "stream_us": stream_us,
                   "rowmajor_us": old[0], "rowmajor_cfg": old[1],
                   "tiled_us": til[0], "tiled_cfg": til[1],
                   "ring_us": rng[0], "ring_cfg": rng[1],
                   "best_tbps": nbytes / best[0] / 1e6, "speedup": old[0] / best[0],
                   "regs_blocks_split": occupancy(tiled[0], batch, best[1], mode)}
            line = (f"{name:8s} b{batch:<3d} {nbytes/1e6:5.1f}MB stream {stream_us:6.1f} | row {old[0]:6.1f} {old[1][:2]} "
                    f"| tiled {til[0]:6.1f} {til[1][:2]} | ring {rng[0]:6.1f} {rng[1][:2]}g{rng[1][4]} "
                    f"| best {row['best_tbps']:.2f}TB/s x{row['speedup']:.2f} r/b/s {row['regs_blocks_split']}")
            if name == "gate_up":
                sw_old = sweep(x, rowmajor[0], rowmajor, cuda_fp8.SWIGLU_CONFIGS, reps, 3, swiglu=True)
                sw_til = sweep(x, rowmajor[0], tiled, cuda_fp8.SWIGLU_CONFIGS, reps, 3, swiglu=True)
                sw_rng = sweep(x, rowmajor[0], tiled, ring_sw, reps, 3, swiglu=True)
                row.update({"swiglu_rowmajor_us": sw_old[0], "swiglu_tiled_us": sw_til[0],
                            "swiglu_ring_us": sw_rng[0], "swiglu_ring_cfg": sw_rng[1]})
                line += f" | swiglu row {sw_old[0]:.1f} tiled {sw_til[0]:.1f} ring {sw_rng[0]:.1f} {sw_rng[1]}"
            rows.append(row)
            print(line, flush=True)
        del rowmajor, tiled
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    batches = [int(b) for b in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "4", "16", "32"])]
    shapes = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
    print("RESULT_JSON=" + json.dumps(run(batches, shapes)), flush=True)
