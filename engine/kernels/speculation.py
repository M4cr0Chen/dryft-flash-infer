"""Capped short verification with an ordinary-decode fallback.

Each width owns its tuned operands, buffers and graph. Only target-confirmed
drafts are accepted. Quantized target outputs remain subject to the judge's
native teacher-forced margin; speculation does not repair quantization error.

Every width races only the weight family ordinary decode chose for each
projection, so the verifying model is the model that filled the cache.
"""

import torch

from . import DecodeAttention
from .ngram import NgramDrafter


class ShortVerifier:
    @torch.inference_mode()
    def __init__(self, engine, max_draft=2):
        if engine.batch != 1 or engine.draft or max_draft not in (1, 2):
            raise ValueError("short verification needs a warmed ordinary batch-one engine")
        self.engine = engine
        self.max_draft = max_draft
        self.configs = {}
        original = {name: getattr(engine, name) for name in
                    ("matmul", "operand", "fused_mlp", "fused_operand", "partial_config",
                     "decode_attention", "draft")}
        original_position = engine.pos.clone()
        try:
            for draft in range(1, max_draft + 1):
                width = draft + 1
                engine.draft = draft
                engine._choose_matmuls(width, families=engine.families)
                engine.decode_attention = (DecodeAttention(
                    1, engine.n_kv, engine.n_q // engine.n_kv, engine.head_dim,
                    engine.capacity, engine.device, tokens=width,
                ) if original["decode_attention"] is not None else None)
                engine.spec_ids = torch.zeros(1, width, device=engine.device, dtype=torch.int64)
                engine.spec_pred = torch.zeros(width, device=engine.device, dtype=torch.int64)
                engine.offsets = torch.arange(width, device=engine.device, dtype=torch.int32)
                engine.spec_positions = torch.zeros(width, device=engine.device, dtype=torch.int32)
                engine.spec_mask = torch.zeros(width, engine.capacity, device=engine.device, dtype=torch.bool)
                engine.pos.fill_(engine.seq_len)
                graph = None
                if engine.graph is not None:
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
                    ids_host=torch.empty((1, width), dtype=torch.int64, pin_memory=engine.cuda),
                    pred_host=torch.empty(width, dtype=torch.int64, pin_memory=engine.cuda),
                )
        finally:
            for name, value in original.items():
                setattr(engine, name, value)
            engine.pos.copy_(original_position)
        self.host = torch.empty(1, dtype=torch.int64, pin_memory=engine.cuda)
        self.ready = torch.cuda.Event() if engine.cuda else None

    def _record(self):
        if self.ready is not None:
            self.ready.record()

    def _wait(self):
        if self.ready is not None:
            self.ready.synchronize()

    def _verify_step(self, config):
        if config["graph"] is not None:
            config["graph"].replay()
            return
        # Eager/CPU path lets tests exercise cache rollback and graph-width
        # dispatch against a real small Qwen model without a GPU.
        names = ("matmul", "operand", "fused_mlp", "fused_operand", "partial_config",
                 "decode_attention", "draft", "spec_ids", "spec_pred", "offsets",
                 "spec_positions", "spec_mask")
        original = {name: getattr(self.engine, name) for name in names}
        try:
            for name in names:
                setattr(self.engine, name, config[name])
            self.engine._spec_step()
        finally:
            for name, value in original.items():
                setattr(self.engine, name, value)

    def generate(self, prompts, steps, draft=2, order=2, target=1.15):
        """Standalone benchmark wrapper; Engine.generate calls stream directly."""
        if steps <= 0:
            return
        with torch.inference_mode():
            self.engine._prefill(self.engine._upload(prompts, 1, self.engine.seq_len))
        yield from self.stream(prompts[0], steps, draft=draft, order=order, target=target)

    def stream(self, prompt, steps, draft=None, order=2, target=1.15):
        engine = self.engine
        if steps <= 0:
            return
        draft = self.max_draft if draft is None else draft
        if len(prompt) != engine.seq_len or not 1 <= draft <= self.max_draft:
            raise ValueError("verification study shape changed")
        self.stats = {"ordinary_passes": 0, "verify_passes": 0, "accepted": 0,
                      "proposed": 0, "outputs": steps}
        drafter = NgramDrafter(order=order, draft=draft, target=target)
        drafter.reset(prompt)
        with torch.inference_mode():
            self.host.copy_(engine.emitted, non_blocking=engine.cuda)
            self._record()
        self._wait()
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
                    config["spec_ids"].copy_(config["ids_host"], non_blocking=engine.cuda)
                    self._verify_step(config)
                    config["pred_host"].copy_(config["spec_pred"], non_blocking=engine.cuda)
                    self._record()
                else:
                    if engine.graph is not None:
                        engine.graph.replay()
                    else:
                        engine._decode_step()
                    self.host.copy_(engine.emitted, non_blocking=engine.cuda)
                    self._record()
            self._wait()
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
