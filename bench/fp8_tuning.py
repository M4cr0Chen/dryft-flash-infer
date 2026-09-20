"""Compare two weight dequantization placements before changing the engine."""

import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, "/root/engine")
from kernels import fp8
from kernels.timing import time_calls


@triton.jit
def postscaled_gemm(
    X, W, S, Y, K,
    N: tl.constexpr, M: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PAD_M: tl.constexpr,
    SPLITS: tl.constexpr, PER_SPLIT: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, PAD_M)
    begin = tl.program_id(1) * PER_SPLIT
    finish = tl.minimum(begin + PER_SPLIT, GROUPS)
    acc = tl.zeros((PAD_M, BLOCK_N), tl.float32)
    for group in range(begin, finish):
        inner = group * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X + rows[:, None] * K + inner[None, :],
                    mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        packed = tl.load(W + cols.to(tl.int64)[:, None] * K + inner[None, :],
                         mask=(cols[:, None] < N) & (inner[None, :] < K), other=0.0)
        scale = tl.load(S + cols.to(tl.int64) * GROUPS + group,
                        mask=cols < N, other=0.0).to(tl.float32)
        # E4M3 values are exactly representable in BF16. Apply each group's
        # scale to its dot product rather than every expanded weight value.
        acc += tl.dot(x, tl.trans(packed.to(tl.bfloat16))) * scale[None, :]
    tl.store(Y + tl.program_id(1).to(tl.int64) * (M * N)
             + rows[:, None] * N + cols[None, :],
             acc.to(Y.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@torch.inference_mode()
def run():
    torch.manual_seed(37)
    original = fp8._fp8_gemm
    results = []
    for name, n, k in (("qkv", 6144, 2560), ("o", 2560, 4096),
                       ("gate_up", 19456, 2560), ("down", 2560, 9728)):
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                   for _ in range(36)]
        packed = [fp8.quantize(w) for w in weights]
        for batch in (2, 3, 4, 16):
            x = torch.randn(batch, k, device="cuda", dtype=torch.bfloat16)
            fp8._fp8_gemm = original
            reference = fp8.fp8_matmul(x, packed[0])
            scale = reference.float().abs().max().item()
            rows = []
            for label, kernel in (("expanded", original), ("postscaled", postscaled_gemm)):
                fp8._fp8_gemm = kernel
                for config in fp8.CONFIGS:
                    fn = lambda w, c=config: fp8.fp8_matmul(x, w, config=c)
                    out = fn(packed[0])
                    error = (out.float() - reference.float()).abs().max().item() / scale
                    assert torch.isfinite(out).all() and error < 0.02, (label, config, error)
                    ms = time_calls(fn, packed, reps=2, trials=3) * len(packed)
                    rows.append({"variant": label, "config": config, "ms": ms,
                                 "relative_error": error})
            bf16 = time_calls(lambda w: F.linear(x, w), weights, reps=2, trials=3) * len(weights)
            best = min(rows, key=lambda row: row["ms"])
            results.append({"projection": name, "batch": batch, "cublas_ms": bf16,
                            "candidates": rows, "best": best})
            print(f"{name} b{batch}: {best['variant']} {best['config']} "
                  f"{best['ms']:.3f} ms vs cuBLAS {bf16:.3f} ms", flush=True)
        del packed, weights
        torch.cuda.empty_cache()
    fp8._fp8_gemm = original
    return results


if __name__ == "__main__":
    print("RESULT_JSON=" + json.dumps(run()), flush=True)
