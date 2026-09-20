"""The fused GEMM+add-norm kernel against the planes-and-consumer pair, per shape."""

import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8  # noqa: E402
from kernels.norm import add_rms_norm_partials  # noqa: E402
from kernels.timing import time_calls  # noqa: E402


@torch.inference_mode()
def run():
    torch.manual_seed(5)
    assert cuda_fp8.ready()
    n = 2560
    for name, k, cfg in (("o", 4096, (4, 8, 1, 1)), ("down", 9728, (4, 8, 1, 1))):
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(12)]
        prepared = [cuda_fp8.prepare(cuda_fp8.quantize(w), tiled=False) for w in weights]  # row-major kernel
        norms = [torch.randn(n, dtype=torch.bfloat16, device="cuda") for _ in range(12)]
        pairs = list(zip(prepared, norms))
        for batch in (1, 4, 16):
            x = torch.randn(batch, k, dtype=torch.bfloat16, device="cuda")
            res = torch.randn(batch, n, dtype=torch.bfloat16, device="cuda")

            def planes_only(pair):
                return cuda_fp8.matmul_partials(x, pair[0], cfg)

            def split(pair):
                return add_rms_norm_partials(res, cuda_fp8.matmul_partials(x, pair[0], cfg), pair[1], 1e-6)

            def gemm_only(pair):
                return cuda_fp8.matmul(x, pair[0], cfg)

            scratch = cuda_fp8.add_norm_scratch(batch, n, "cuda")
            row = {"planes": time_calls(planes_only, pairs, reps=2, trials=5),
                   "planes+norm": time_calls(split, pairs, reps=2, trials=5),
                   "gemm bf16 out": time_calls(gemm_only, pairs, reps=2, trials=5)}
            for stage in (0, 1):
                for tail in (0, 2, 1):
                    cuda_fp8.FUSED_DEBUG.update(stage=stage, tail=tail)
                    row[f"w8 s{stage} t{tail}"] = time_calls(
                        lambda pair: cuda_fp8.matmul_add_norm(x, pair[0], res, pair[1], 1e-6, scratch, warps=8),
                        pairs, reps=2, trials=5)
            cuda_fp8.FUSED_DEBUG.update(stage=None, tail=1)
            print(f"{name:5s} b{batch:<3d} " + "  ".join(f"{k_}: {v*1e3:6.1f}us" for k_, v in row.items()), flush=True)


if __name__ == "__main__":
    run()
