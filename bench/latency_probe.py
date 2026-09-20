"""Fixed cost versus streaming cost of the decode GEMM, in the decode timing harness.

For one K and a sweep of row counts: the incumbent kernel (planes when it
splits K, as the engine consumes them), the cp.async ring kernel, a kernel
that only streams the same bytes, and an empty launch. Everything is timed
as graph nodes cycling twelve weights, like the warmup race and the step.
"""

import json
import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8, cuda_jit  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

_SRC = r"""
extern "C" __global__ void noop(int* flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && flag[0] == 12345) flag[1] = 1;
}
// Each thread reads @UNROLL@ x 16 bytes, strided by the grid, and keeps a
// data dependency alive so nothing is optimised away.
extern "C" __global__ void __launch_bounds__(256)
stream(const uint4* __restrict__ p, long chunks, int* flag) {
    unsigned int acc = 0;
    long i = (long)blockIdx.x * 256 * @UNROLL@ + threadIdx.x;
    #pragma unroll
    for (int u = 0; u < @UNROLL@; ++u) {
        const long idx = i + (long)u * 256;
        if (idx < chunks) { const uint4 v = p[idx]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x9e3779b9u) flag[1] = (int)acc;
}
"""


def timed(fn, operands, reps=2, trials=5):
    return time_calls(fn, operands, reps=reps, trials=trials) * 1e3


def best_gemm(x, prepared, configs):
    best = (float("inf"), None)
    for cfg in configs:
        if not cuda_fp8.applicable(cfg, x.shape[0]):
            continue
        try:
            splitk = cuda_fp8.splits(prepared[0], x.shape[0], cfg)
            fn = (lambda p, c=cfg: cuda_fp8.matmul_partials(x, p, config=c)) if splitk > 1 \
                else (lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c))
            fn(prepared[0])
        except Exception:
            continue
        us = timed(fn, prepared)
        if us < best[0]:
            best = (us, cfg)
    return best


@torch.inference_mode()
def run():
    torch.manual_seed(3)
    assert cuda_fp8.ready()
    module = cuda_jit.Module(_SRC.replace("@UNROLL@", "8"))
    noop, stream = module.kernel("noop"), module.kernel("stream")
    flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    noop_us = timed(lambda _: noop(1, 32, flag), [None] * 12)
    print(f"empty launch as a graph node: {noop_us:.2f} us", flush=True)

    rows = []
    for k, ns in ((2560, (256, 1024, 2560, 6144, 19456)), (9728, (2560,)), (4096, (2560,))):
        for n in ns:
            weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(12)]
            prepared = [cuda_fp8.prepare(cuda_fp8.quantize(w)) for w in weights]
            del weights
            nbytes = prepared[0].bytes_moved()
            chunks = prepared[0].weight.numel() // 16
            grid = -(-chunks // (256 * 8))
            stream_us = timed(lambda p: stream(grid, 256, p.weight, chunks, flag), prepared)
            for batch in (1, 16):
                x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
                old_us, old_cfg = best_gemm(x, prepared, cuda_fp8.CONFIGS)
                ring_us, ring_cfg = best_gemm(x, prepared, [(w, k, 1, 1, r) for r in cuda_fp8.RING_DEPTHS for w in cuda_fp8._RING_WARPS for k in (1, 2, 3, 4, 6, 8)])
                row = {"k": k, "n": n, "batch": batch, "mb": nbytes / 1e6, "noop_us": noop_us,
                       "stream_us": stream_us, "stream_tbps": nbytes / stream_us / 1e6,
                       "old_us": old_us, "old_cfg": old_cfg, "old_tbps": nbytes / old_us / 1e6,
                       "ring_us": ring_us, "ring_cfg": ring_cfg, "ring_tbps": nbytes / ring_us / 1e6}
                rows.append(row)
                print(f"K{k:<5d} N{n:<6d} b{batch:<3d} {row['mb']:6.1f}MB | stream {stream_us:6.1f}us "
                      f"{row['stream_tbps']:.2f}TB/s | old {old_us:6.1f}us {row['old_tbps']:.2f}TB/s {old_cfg} "
                      f"| ring {ring_us:6.1f}us {row['ring_tbps']:.2f}TB/s {ring_cfg}", flush=True)
            del prepared
            torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    print("RESULT_JSON=" + json.dumps(run()), flush=True)
