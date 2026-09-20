"""4-bit versus 8-bit decode weights, per projection and batch, on one H100.

Fresh process. Runs the GPU correctness check first, then times the best race
configuration of each width, twelve weights cycled, in the decode graph harness.
"""

import json
import sys

import torch

sys.path.insert(0, "/root")
sys.path.insert(0, "/root/engine")
from gpu_checks import check_int4  # noqa: E402
from kernels import cuda_fp8  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560),
          ("down", 2560, 9728), ("lm_head", 151936, 2560)]


def best(x, prepared, reps):
    top = (float("inf"), None)
    for cfg in cuda_fp8.CONFIGS:
        if not cuda_fp8.applicable(cfg, x.shape[0]):
            continue
        splitk = cuda_fp8.splits(prepared[0], x.shape[0], cfg)
        fn = (lambda p, c=cfg: cuda_fp8.matmul_partials(x, p, config=c)) if splitk > 1 \
            else (lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c))
        us = time_calls(fn, prepared, reps=reps, trials=3) * 1e3
        if us < top[0]:
            top = (us, cfg)
    return top


@torch.inference_mode()
def run(batches):
    check_int4()
    torch.manual_seed(9)
    rows = []
    for name, n, k in SHAPES:
        count = 1 if name == "lm_head" else 12
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(count)]
        eight = [cuda_fp8.prepare(cuda_fp8.quantize(w)) for w in weights]
        four = [cuda_fp8.prepare(cuda_fp8.quantize(w, bits=4)) for w in weights]
        del weights
        reps = 1 if name == "lm_head" else 2
        for batch in batches:
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            b8 = best(x, eight, reps)
            b4 = best(x, four, reps)
            row = {"projection": name, "batch": batch,
                   "int8_us": b8[0], "int8_cfg": b8[1], "int8_tbps": eight[0].bytes_moved() / b8[0] / 1e6,
                   "int4_us": b4[0], "int4_cfg": b4[1], "int4_tbps": four[0].bytes_moved() / b4[0] / 1e6,
                   "speedup": b8[0] / b4[0]}
            rows.append(row)
            print(f"{name:8s} b{batch:<3d} int8 {b8[0]:6.1f}us {row['int8_tbps']:.2f}TB/s {b8[1]} | "
                  f"int4 {b4[0]:6.1f}us {row['int4_tbps']:.2f}TB/s {b4[1]} | x{row['speedup']:.2f}", flush=True)
        del eight, four
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    batches = [int(b) for b in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "4", "16", "32"])]
    print("RESULT_JSON=" + json.dumps(run(batches)), flush=True)
