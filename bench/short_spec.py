"""Short exact verification experiment with an ordinary-decode fallback.

This stays outside the engine until paired, diverse-prompt measurements justify
enabling it. Each width owns separately tuned operands, attention buffers, and
a CUDA graph. An empty proposal runs the original one-token graph.
"""

import torch

from kernels.attention import DecodeAttention
from kernels.ngram import NgramDrafter


class ShortVerifier:
    @torch.inference_mode()
    def __init__(self, engine, max_draft=2):
        if engine.batch != 1 or engine.graph is None or engine.draft:
            raise ValueError("short verification needs a warmed ordinary batch-one engine")
        self.engine = engine
        self.configs = {}
        original = {name: getattr(engine, name) for name in
                    ("matmul", "operand", "fused_mlp", "decode_attention", "draft")}
        original_position = engine.pos.clone()
        for draft in range(1, max_draft + 1):
            width = draft + 1
            engine.draft = draft
            engine._choose_matmuls(width)
            engine.decode_attention = DecodeAttention(
                1, engine.n_kv, engine.n_q // engine.n_kv, engine.head_dim,
                engine.capacity, engine.device, tokens=width,
            )
            engine.spec_ids = torch.zeros(1, width, device="cuda", dtype=torch.int64)
            engine.spec_pred = torch.zeros(width, device="cuda", dtype=torch.int64)
            engine.offsets = torch.arange(width, device="cuda", dtype=torch.int32)
            engine.spec_positions = torch.zeros(width, device="cuda", dtype=torch.int32)
            engine.spec_mask = torch.zeros(width, engine.capacity, device="cuda", dtype=torch.bool)
            engine.pos.fill_(engine.seq_len)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                engine._spec_step()
                engine._spec_step()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                engine._spec_step()
            self.configs[width] = {
                name: getattr(engine, name) for name in
                (*original, "spec_ids", "spec_pred", "offsets", "spec_positions", "spec_mask")
            }
            self.configs[width].update(
                graph=graph,
                ids_host=torch.empty((1, width), dtype=torch.int64, pin_memory=True),
                pred_host=torch.empty(width, dtype=torch.int64, pin_memory=True),
            )
        for name, value in original.items():
            setattr(engine, name, value)
        engine.pos.copy_(original_position)
        self.host = torch.empty(1, dtype=torch.int64, pin_memory=True)
        self.ready = torch.cuda.Event()

    def generate(self, prompts, steps, draft=2, order=2, target=1.15):
        engine = self.engine
        if steps <= 0:
            return
        if len(prompts) != 1 or len(prompts[0]) != engine.seq_len:
            raise ValueError("verification study shape changed")
        self.stats = {"ordinary_passes": 0, "verify_passes": 0, "accepted": 0,
                      "proposed": 0, "outputs": steps}
        drafter = NgramDrafter(order=order, draft=draft, target=target)
        drafter.reset(prompts[0])
        with torch.inference_mode():
            engine._prefill(engine._upload(prompts, 1, engine.seq_len))
            self.host.copy_(engine.emitted, non_blocking=True)
            self.ready.record()
        self.ready.synchronize()
        first = self.host.tolist()
        drafter.commit(first)
        yield first
        remaining = steps - 1
        while remaining:
            proposal = drafter.propose()[:min(draft, remaining - 1)]
            with torch.inference_mode():
                if proposal:
                    config = self.configs[len(proposal) + 1]
                    config["ids_host"][0, 0] = drafter.context[-1]
                    for i, token in enumerate(proposal, 1):
                        config["ids_host"][0, i] = token
                    config["spec_ids"].copy_(config["ids_host"], non_blocking=True)
                    config["graph"].replay()
                    config["pred_host"].copy_(config["spec_pred"], non_blocking=True)
                    self.ready.record()
                else:
                    engine.graph.replay()
                    self.host.copy_(engine.emitted, non_blocking=True)
                    self.ready.record()
            self.ready.synchronize()
            if proposal:
                predicted = config["pred_host"].tolist()
                accepted = 0
                for guess, actual in zip(proposal, predicted):
                    if guess != actual:
                        break
                    accepted += 1
                produced = predicted[:accepted + 1]
                with torch.inference_mode():
                    engine.pos.add_(len(produced))
                    engine.token.copy_(config["spec_pred"][accepted:accepted + 1].view(1, 1))
                    engine.emitted.copy_(config["spec_pred"][accepted:accepted + 1])
                self.stats["verify_passes"] += 1
                self.stats["proposed"] += len(proposal)
                self.stats["accepted"] += accepted
            else:
                produced = self.host.tolist()
                self.stats["ordinary_passes"] += 1
            drafter.commit(produced)
            remaining -= len(produced)
            for token in produced:
                yield [token]
