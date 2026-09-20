"""Programmatic dependent launch on a chain of our GEMMs, inside a CUDA graph.

72 dependent GEMMs (each reads the previous one's output), twelve weights
cycled, captured once and replayed. Variants: plain launches; kernels built
with griddepcontrol hooks but launched plainly (the hooks' own cost); hooks
plus the programmatic launch attribute. Then the same with a Triton RMSNorm
between GEMMs, launched normally, to see what PDL on the GEMMs alone buys.
"""

import statistics
import sys

import torch

sys.path.insert(0, "/root/engine")
from kernels import cuda_fp8  # noqa: E402
from kernels.norm import rms_norm  # noqa: E402


def graph_time(fn, reps=20):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(7):
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / reps)
    return statistics.median(times) * 1e3


@torch.inference_mode()
def run():
    torch.manual_seed(3)
    results = {}
    for batch in (1, 16):
        weights = [torch.randn(2560, 2560, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(12)]
        gain = torch.ones(2560, dtype=torch.bfloat16, device="cuda")
        x0 = torch.randn(batch, 2560, dtype=torch.bfloat16, device="cuda")
        for label, level in (("plain", 0), ("PDL, late trigger", 1), ("PDL, early trigger", 2)):
            cuda_fp8.PDL = level
            cuda_fp8._modules.clear()
            assert cuda_fp8.ready()
            prepared = [cuda_fp8.prepare(cuda_fp8.quantize(w)) for w in weights]
            for cfg in ((4, 4, 1, 1, 0), (8, 4, 1, 1, 2)):
                def chain(norm=False):
                    x = x0
                    for i in range(72):
                        out = cuda_fp8.matmul(x, prepared[i % 12], config=cfg)   # 4 planes + reduce
                        x = rms_norm(out, gain, 1e-6) if norm else out
                    return x
                ref = chain()
                us = graph_time(chain)
                us_norm = graph_time(lambda: chain(True))
                key = (batch, label, cfg)
                results[key] = (us, us_norm)
                print(f"b{batch:<3d} {label:22s} {cfg}  GEMM+reduce chain {us/72:6.2f} us/GEMM   "
                      f"+RMSNorm {us_norm/72:6.2f} us/GEMM   checksum {ref.float().abs().sum().item():.3e}", flush=True)
            del prepared
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    run()
