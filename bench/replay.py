"""A repeatable decode microbenchmark with bounded cache accesses.

This is a fixed-prefix experiment, not generation throughput. State restoration
is captured and included in its timings; it adds two device copies per replay.
"""

import torch


class FixedDecodeReplay:
    @torch.no_grad()
    def __init__(self, engine):
        if engine.graph is None or engine.draft:
            raise ValueError("fixed decode replay requires a non-speculative CUDA graph")
        self.position = int(engine.pos.item())
        if not 0 <= self.position < engine.capacity:
            raise ValueError("decode position is outside the allocated cache")
        self._position = engine.pos.clone()
        self._token = engine.token.clone()

        def step():
            engine._decode_step()
            # The next replay consumes the same token at the same position.
            # Slots after that position are masked by the decode attention.
            engine.pos.copy_(self._position)
            engine.token.copy_(self._token)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            step()
        torch.cuda.synchronize()

    def replay(self):
        self.graph.replay()


def check_fixed_replay(engine, repeats=64):
    """Exercise more replays than the spare cache slots without advancing it."""
    fixed = FixedDecodeReplay(engine)
    fixed.replay()
    expected = engine.emitted.clone()
    for _ in range(repeats):
        fixed.replay()
    torch.cuda.synchronize()
    if int(engine.pos.item()) != fixed.position or not torch.equal(engine.emitted, expected):
        raise AssertionError("fixed decode replay changed its prefix or output")
