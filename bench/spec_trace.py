"""Trace the first failing long prose case without changing shipping kernels."""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/engine")
import engine as module
from harness import _prompts, _replay, load_corpus
from kernels.speculation import ShortVerifier
from transformers import AutoTokenizer


@torch.inference_mode()
def run():
    module.SHORT_DRAFT = 0
    torch.manual_seed(42)
    load_corpus("/root/corpus.txt", "/weights/qwen3-4b")
    engine = module.Engine("/weights/qwen3-4b")
    list(engine.generate(_prompts(1, 512, engine.embed.shape[0], 9001), 128))
    verifier = ShortVerifier(engine)
    inputs = _prompts(1, 512, engine.embed.shape[0], 4100)
    tokenizer = AutoTokenizer.from_pretrained("/weights/qwen3-4b", local_files_only=True)
    graph, verify = engine.graph, verifier._verify_step
    traces = []
    compare_eager = False

    class TracedGraph:
        def replay(self):
            position = engine.pos.item()
            token = engine.token.clone()
            graph.replay()
            prediction = engine.emitted.clone()
            record = {"kind": "ordinary", "position": position,
                      "ids": token.flatten().tolist(), "pred": prediction.tolist()}
            if compare_eager and 615 <= position <= 628:
                engine.pos.fill_(position)
                engine.token.copy_(token)
                engine._decode_step()
                record["eager"] = engine.emitted.tolist()
                engine.token.copy_(prediction.view(1, 1))
                engine.emitted.copy_(prediction)
            traces.append(record)

    def traced_verify(config):
        position = engine.pos.item()
        ids = config["spec_ids"].flatten().tolist()
        verify(config)
        prediction = config["spec_pred"].clone()
        record = {"kind": "verify", "position": position, "ids": ids,
                  "pred": prediction.tolist()}
        if compare_eager and 615 <= position <= 628:
            saved_graph = config["graph"]
            config["graph"] = None
            try:
                verify(config)
                record["eager"] = config["spec_pred"].tolist()
            finally:
                config["graph"] = saved_graph
                config["spec_pred"].copy_(prediction)
        traces.append(record)

    results = []
    for label in ("normal", "trace", "eager_compare"):
        traces.clear()
        if label != "normal":
            engine.graph = TracedGraph()
            verifier._verify_step = traced_verify
        compare_eager = label == "eager_compare"
        emitted = list(verifier.generate(inputs, 128))
        gap, exact = _replay(engine._native, engine.device, inputs, emitted, 128)
        row = {"label": label, "gap": gap, "tokens": emitted, "stats": verifier.stats,
               "trace": list(traces),
               "text": tokenizer.decode([r[0] for r in emitted])}
        results.append(row)
        print(label, gap, row["text"], flush=True)
        for record in traces:
            if 615 <= record["position"] <= 628:
                print(record, flush=True)
    # Replay the failing prefix with ordinary decode, then remove each
    # compressed projection separately. This distinguishes numeric drift from
    # a graph/cache control-flow error without changing the prompt or prefix.
    verifier._verify_step = verify
    ordinary_matmul, ordinary_operands = engine.matmul, engine.operand
    failing_tokens = torch.tensor(results[0]["tokens"], device=engine.device)
    for label, bf16 in (("all_fp8", ()), ("bf16_qkv", ("qkv",)),
                        ("bf16_gate_up", ("gate_up",)),
                        ("all_bf16", ("qkv", "gate_up"))):
        engine.matmul = dict(ordinary_matmul)
        engine.operand = dict(ordinary_operands)
        for name in bf16:
            engine.matmul[name] = F.linear
            engine.operand[name] = [getattr(layer, name) for layer in engine.layers]
        engine._capture()
        emitted = list(verifier.generate(inputs, 128))
        gap, _ = _replay(engine._native, engine.device, inputs, emitted, 128)
        engine._prefill(engine._upload(inputs, 1, 512))
        for position in range(1, 112):
            engine.token.copy_(failing_tokens[position - 1].view(1, 1))
            engine.graph.replay()
        forced = engine.emitted.item()
        row = {"label": label, "gap": gap, "tokens": emitted,
               "forced_prefix_prediction_111": forced,
               "forced_prefix_text_111": tokenizer.decode([forced])}
        results.append(row)
        print(label, "gap", gap, "forced-prefix prediction", forced,
              repr(row["forced_prefix_text_111"]), flush=True)
    return results


if __name__ == "__main__":
    print("RESULT_JSON=" + json.dumps(run()), flush=True)
