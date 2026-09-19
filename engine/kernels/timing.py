"""Warmup-only timing in the execution mode used by decode."""

import statistics

import torch


@torch.no_grad()
def time_calls(fn, operands, *, reps=3, trials=5, use_graph=True):
    """Median milliseconds per call, cycling through distinct layer weights.

    Capture an entire sweep so Python/ctypes launch overhead cannot decide a
    race between kernels that will run inside the decode graph. Cycling all
    layers also avoids measuring an unrealistically hot weight in L2.
    """
    if not operands or reps < 1 or trials < 1:
        raise ValueError("timing needs operands and positive repeat counts")

    def sweep():
        for _ in range(reps):
            for operand in operands:
                fn(operand)

    graph = None
    if use_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for operand in operands[:4]:
                fn(operand)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            sweep()
        for _ in range(2):
            graph.replay()
        run = graph.replay
    else:
        for operand in operands[:4]:
            fn(operand)
        run = sweep

    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(trials):
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / (reps * len(operands)))
    return statistics.median(times)
