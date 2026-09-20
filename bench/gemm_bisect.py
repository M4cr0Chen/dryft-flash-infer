"""Attribute the decode GEMM's time above the streaming floor to its parts.

Compiles the incumbent kernel with one component removed at a time (MMA,
dequantisation, activation staging, epilogue) and times each variant on the
real shapes at batch 1 and 16, against a kernel that only streams the bytes.
Outputs are wrong by construction for every variant but "full".
"""

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

MMA_LINE = "                            mma_bf16(tmp[nt], a, b0, b1);\n"
NOMMA = ("                            tmp[nt][0] += __uint_as_float((a[0] ^ b0) & 0x3fffffffu);\n"
         "                            tmp[nt][1] += __uint_as_float((a[1] ^ b1) & 0x3fffffffu);\n"
         "                            tmp[nt][2] += __uint_as_float((a[2] ^ b0) & 0x3fffffffu);\n"
         "                            tmp[nt][3] += __uint_as_float((a[3] ^ b1) & 0x3fffffffu);\n")
DEQ = ("                        a[0] = DEQUANT(wl, 0x7650, 0x7651);\n"
       "                        a[1] = DEQUANT(wh, 0x7650, 0x7651);\n"
       "                        a[2] = DEQUANT(wl, 0x7652, 0x7653);\n"
       "                        a[3] = DEQUANT(wh, 0x7652, 0x7653);\n")
NODEQ = ("                        a[0] = wl; a[1] = wh; a[2] = wl ^ 0x00010001u; a[3] = wh ^ 0x00010001u;\n")

XB_LOADS = ("                        xb[nt][0] = src[0];\n"
            "                        xb[nt][1] = src[1];\n")
NOXB = ("                        xb[nt][0] = make_uint4(0x3f803f80u ^ (u32)nt, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);\n"
        "                        xb[nt][1] = make_uint4(0x3f803f80u, 0x3f803f80u ^ (u32)local, 0x3f803f80u, 0x3f803f80u);\n")

VARIANTS = {
    "full": {},
    "nomma": {"mma": True},
    "nodeq": {"deq": True},
    "noxb": {"xb": True},          # activation fragments from registers, not shared memory
    "loads_only": {"mma": True, "deq": True},
    "nostage": {"stage": 0},
    "noepi": {"epi": True},
}


def build(name, opts, warps, nt, mode):
    body = cuda_fp8._BODY
    if opts.get("mma"):
        assert MMA_LINE in body
        body = body.replace(MMA_LINE, NOMMA)
    if opts.get("deq"):
        assert DEQ in body
        body = body.replace(DEQ, NODEQ)
    if opts.get("xb"):
        assert XB_LOADS in body
        body = body.replace(XB_LOADS, NOXB)
    epi = cuda_fp8._EPILOGUE
    if opts.get("epi"):
        # Keep one dependent store so the compiler cannot drop the loop.
        epi = ("\n    if (acc[0][0] == 1234.5f) ((bf16*)OUT)[0] = 1;\n}\n\"\"\"".replace('"""', ''))
    stage = opts.get("stage", 1)
    src = cuda_fp8._HEAD + cuda_fp8._STAGING + body + epi
    kname = f"bisect_{name}_w{warps}_n{nt}_m{mode}_s{stage}"
    src = cuda_fp8._substitute(cuda_fp8._PRELUDE + src, NAME=kname, WARPS=warps, THREADS=warps * 32,
                               NT=nt, NSHARE=1, MODE=mode, STAGE=stage, TILED=1, BITS=8, PDL=0)
    return cuda_jit.Module(src).kernel(kname), stage


def launch(kernel, stage, prepared, x, config, mode):
    batch, k = x.shape
    warps, nt, splitk, gps, _, nshare, _ = cuda_fp8._plan(prepared, batch, config)
    n = prepared.rows
    rowblocks = -(-n // (warps * 16))
    out = torch.empty((splitk, batch, n), dtype=torch.float32, device=x.device) if mode == 1 \
        else torch.empty((batch, n), dtype=torch.bfloat16, device=x.device)
    kernel.set_shared(cuda_fp8._shared_bytes(nt, gps) if stage else 0)
    kernel(rowblocks * splitk, warps * 32, prepared.weight, prepared.scale, x, out,
           n, k, batch, prepared.groups, gps, splitk)
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(5)
    assert cuda_fp8.ready()
    stream = cuda_jit.Module(STREAM).kernel("stream")
    flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    shapes = [("gate_up", 19456, 2560, (8, 4, 1, 1, 0)), ("down", 2560, 9728, (8, 4, 1, 1, 0)),
              ("lm_head", 151936, 2560, (8, 2, 1, 1, 0))]
    rows = []
    for name, n, k, config in shapes:
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
                   for _ in range(1 if name == "lm_head" else 12)]
        prepared = [cuda_fp8.prepare(cuda_fp8.quantize(w), tiled=True) for w in weights]
        del weights
        nbytes = prepared[0].bytes_moved()
        chunks = prepared[0].weight.numel() // 16
        grid = -(-chunks // (256 * 8))
        stream_us = time_calls(lambda p: stream(grid, 256, p.weight, chunks, flag), prepared, reps=2, trials=5) * 1e3
        for batch in (16, 32, 64):
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            warps, nt, splitk, gps, _, _, _ = cuda_fp8._plan(prepared[0], batch, config)
            mode = 1 if splitk > 1 else 0
            timings = {}
            for vname, opts in VARIANTS.items():
                kernel, stage = build(vname, opts, warps, nt, mode)
                if vname == "full":
                    ref = cuda_fp8.matmul_partials(x, prepared[0], config) if mode == 1 else cuda_fp8.matmul(x, prepared[0], config)
                    got = launch(kernel, stage, prepared[0], x, config, mode)
                    assert torch.equal(ref, got), "rebuilt full kernel differs"
                timings[vname] = time_calls(lambda p, kk=kernel, st=stage: launch(kk, st, p, x, config, mode),
                                            prepared, reps=2, trials=5) * 1e3
            row = {"projection": name, "batch": batch, "mb": nbytes / 1e6, "stream_us": stream_us,
                   "config": config, "splitk": splitk, **{f"{v}_us": t for v, t in timings.items()}}
            rows.append(row)
            print(f"{name:8s} b{batch:<3d} {nbytes/1e6:5.1f}MB split{splitk} | stream {stream_us:5.1f} | "
                  + " | ".join(f"{v} {t:5.1f}" for v, t in timings.items()), flush=True)
        del prepared
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    print("RESULT_JSON=" + json.dumps(run()), flush=True)
