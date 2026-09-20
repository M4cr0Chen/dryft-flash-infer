"""Warmup-only output-projection selection in the complete decode graph.

An isolated projection/reduction race misses PDL interactions and the cost of
the residual/RMSNorm consumer. Keep the existing INT8 weights and compare a
small set of split-K schedules with every other operation held fixed.
"""
import functools
import statistics
import sys

import torch

from . import cuda_fp8


@torch.no_grad()
def tune_output_projection(engine):
    if (engine.graph is None or engine.draft or engine.batch > 16 or engine.integer_mlp
            or engine.families.get("o") != "fp8"
            or "o" not in engine.partial_config):
        return False
    incumbent = tuple(engine.partial_config["o"])
    candidates = list(dict.fromkeys([
        incumbent, (8, 8, 1, 1, 2), (4, 8, 1, 1, 0), (4, 8, 1, 1, 2),
    ]))
    position, token = engine.pos.clone(), engine.token.clone()
    graphs = {}
    try:
        for config in candidates:
            engine.partial_config["o"] = config
            def step():
                engine._decode_step()
                engine.pos.copy_(position)
                engine.token.copy_(token)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                step()
            graphs[config] = graph
        torch.cuda.synchronize()
        samples = {c: [] for c in candidates}
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for trial in range(7):
            order = candidates[trial % len(candidates):] + candidates[:trial % len(candidates)]
            if trial % 2:
                order = order[::-1]
            for config in order:
                graph = graphs[config]
                graph.replay()
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[config].append(start.elapsed_time(end) / 20)
        medians = {c: statistics.median(v) for c, v in samples.items()}
        best = min(candidates, key=medians.get)
        # Require a full-step win and consistent paired trial direction.
        wins = sum(a < b for a, b in zip(samples[best], samples[incumbent]))
        if medians[best] >= medians[incumbent] * .99 or wins < 6:
            best = incumbent
        engine.partial_config["o"] = best
        if best != incumbent:
            engine.matmul["o"] = functools.partial(cuda_fp8.matmul, config=best)
            engine._ordinary_matmul = dict(engine.matmul)
        print(f"engine: output graph race b{engine.batch} {incumbent} -> {best}; "
              f"{medians[incumbent]:.4f} -> {medians[best]:.4f} ms", file=sys.stderr)
        return best != incumbent
    except Exception:
        engine.partial_config["o"] = incumbent
        raise
    finally:
        engine.pos.copy_(position)
        engine.token.copy_(token)
        torch.cuda.synchronize()
