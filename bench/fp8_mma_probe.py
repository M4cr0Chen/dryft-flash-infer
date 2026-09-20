"""Correctness and bandwidth of the CUDA tensor-core FP8 GEMM, per shape and batch.

Runs in a fresh process on the H100. Prints one line per (projection, batch)
with cuBLAS BF16, the best existing runner, and the new kernel, and emits a
RESULT_JSON line for the notebook.
"""

import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8, fp8  # noqa: E402
from kernels.timing import time_calls  # noqa: E402

SHAPES = [("qkv", 6144, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560),
          ("down", 2560, 9728), ("lm_head", 151936, 2560)]


def dequant(packed):
    weight, scale = packed
    rows, k = weight.shape
    groups = scale.shape[1]
    return (weight.float().view(rows, groups, k // groups)
            * scale.float()[:, :, None]).view(rows, k).to(torch.bfloat16)


@torch.inference_mode()
def run(batches):
    torch.manual_seed(11)
    start = time.perf_counter()
    if not cuda_fp8.ready():
        raise SystemExit("cuda_fp8 failed to compile")
    print(f"compiled in {time.perf_counter() - start:.1f}s", flush=True)

    results = []
    for name, n, k in SHAPES:
        count = 1 if name == "lm_head" else 12
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
                   for _ in range(count)]
        packed = [cuda_fp8.quantize(w) for w in weights]
        prepared = [cuda_fp8.prepare(p) for p in packed]
        exact = cuda_fp8.dequantize(packed[0])
        for batch in batches:
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            reference = F.linear(x, exact).float()   # same dequantised weight, cuBLAS order
            scale = reference.abs().max().item() or 1.0
            # Correctness for every config, including the SwiGLU epilogue.
            worst = 0.0
            for cfg in cuda_fp8.CONFIGS:
                if not cuda_fp8.applicable(cfg, batch):
                    continue
                out = cuda_fp8.matmul(x, prepared[0], config=cfg).float()
                err = (out - reference).abs().max().item() / scale
                worst = max(worst, err)
                if not torch.isfinite(out).all() or err > 1e-2:
                    raise AssertionError(f"{name} b{batch} cfg{cfg} rel err {err}")
            if name == "gate_up":
                inter = n // 2
                want = F.silu(reference[:, :inter].to(torch.bfloat16)).to(torch.bfloat16).float() \
                    * reference[:, inter:].to(torch.bfloat16).float()
                for cfg in cuda_fp8.SWIGLU_CONFIGS:
                    if not cuda_fp8.applicable(cfg, batch, swiglu=True):
                        continue
                    got = cuda_fp8.gate_up_swiglu(x, prepared[0], config=cfg).float()
                    err = (got - want).abs().max().item() / (want.abs().max().item() or 1.0)
                    if not torch.isfinite(got).all() or err > 2e-2:
                        raise AssertionError(f"swiglu b{batch} cfg{cfg} rel err {err}")

            reps = 1 if name == "lm_head" else 2
            trials = 3
            cublas = time_calls(lambda w: F.linear(x, w), weights, reps=reps, trials=trials)
            timings = {}
            for cfg in cuda_fp8.CONFIGS:
                if not cuda_fp8.applicable(cfg, batch):
                    continue
                timings[cfg] = time_calls(
                    lambda p, c=cfg: cuda_fp8.matmul(x, p, config=c), prepared,
                    reps=reps, trials=trials)
            best_cfg = min(timings, key=timings.get)
            best = timings[best_cfg]
            if name in ("qkv", "gate_up") and batch >= 32:
                for cfg, ms in sorted(timings.items(), key=lambda kv: kv[1])[:12]:
                    print(f"    {name} b{batch} {str(cfg):16s} {ms*1e3:7.1f}us", flush=True)
            old = float("inf")
            legacy = [fp8.quantize(w) for w in weights]
            for cfg in fp8.CONFIGS:
                try:
                    fp8.fp8_matmul(x, legacy[0], config=cfg)
                    old = min(old, time_calls(
                        lambda p, c=cfg: fp8.fp8_matmul(x, p, config=c), legacy,
                        reps=reps, trials=trials))
                except Exception:
                    continue
            del legacy
            swiglu_ms = None
            if name == "gate_up":
                sw = {cfg: time_calls(lambda p, c=cfg: cuda_fp8.gate_up_swiglu(x, p, config=c),
                                      prepared, reps=reps, trials=trials)
                      for cfg in cuda_fp8.SWIGLU_CONFIGS if cuda_fp8.applicable(cfg, batch, swiglu=True)}
                swiglu_ms = min(sw.values())
            bf16_bytes = n * k * 2
            fp8_bytes = prepared[0].bytes_moved()
            row = {"projection": name, "batch": batch, "cublas_ms": cublas,
                   "triton_fp8_ms": old, "cuda_fp8_ms": best, "config": best_cfg,
                   "swiglu_ms": swiglu_ms, "worst_rel_err": worst,
                   "cublas_tbps": bf16_bytes / cublas / 1e9,
                   "cuda_fp8_tbps": fp8_bytes / best / 1e9,
                   "speedup_vs_cublas": cublas / best}
            results.append(row)
            print(f"{name:8s} b{batch:<3d} cublas {cublas*1e3:7.1f}us {row['cublas_tbps']:.2f}TB/s | "
                  f"triton-fp8 {old*1e3:7.1f}us | cuda-fp8 {best*1e3:7.1f}us "
                  f"{row['cuda_fp8_tbps']:.2f}TB/s  x{cublas/best:.2f} cfg{best_cfg}"
                  + (f" | swiglu {swiglu_ms*1e3:7.1f}us" if swiglu_ms else "")
                  + f" | err {worst:.1e}", flush=True)
        del weights, packed, prepared
        torch.cuda.empty_cache()
    return results


@torch.inference_mode()
def norm_variants(batches):
    """How the fused add-norm over planes behaves by warp count and plane count."""
    from kernels import norm
    kernel = norm._add_rms_norm_partials_kernel
    plain = norm._add_rms_norm_kernel
    n = 2560
    weight = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    for batch in batches:
        x = torch.randn(batch, n, dtype=torch.bfloat16, device="cuda")
        d = torch.randn(batch, n, dtype=torch.bfloat16, device="cuda")
        r, y = torch.empty_like(x), torch.empty_like(x)
        base = {}
        for warps in (4, 8, 16):
            base[warps] = time_calls(lambda _: plain[(batch,)](x, d, weight, r, y, n, 1e-6, BLOCK=4096, num_warps=warps),
                                     [None] * 8, reps=4, trials=5)
        print(f"add_rms_norm b{batch}: " + "  ".join(f"w{w} {v*1e3:5.2f}us" for w, v in base.items()), flush=True)
        for splits in (4, 8, 16):
            planes = torch.randn(splits, batch, n, dtype=torch.float32, device="cuda")
            row = {}
            for warps in (4, 8, 16):
                row[warps] = time_calls(
                    lambda _: kernel[(batch,)](x, planes, weight, r, y, batch, n, splits, 1e-6,
                                               BLOCK=4096, num_warps=warps),
                    [None] * 8, reps=4, trials=5)
            print(f"  planes S={splits:2d}: " + "  ".join(f"w{w} {v*1e3:5.2f}us" for w, v in row.items()), flush=True)


if __name__ == "__main__":
    batches = [int(b) for b in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "4", "16"])]
    print("RESULT_JSON=" + json.dumps(run(batches)), flush=True)
