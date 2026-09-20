"""End-to-end equivalence: the engine's own forward against native Qwen.

A real Qwen3ForCausalLM with random weights and the pinned model's proportions,
small enough to run on a CPU. Everything structural is under test here -- cache
slots, absolute positions, the decode mask, the grouped-query reshape, the
residual chain, batch-split prefill -- against the same Transformers path the
judge replays through. Only the Triton codegen and the CUDA graph are not.

    python -m pytest tests/test_engine.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from engine import Engine  # noqa: E402

VOCAB = 512


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """A tiny checkpoint with the pinned model's shape, saved to disk."""
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        use_sliding_window=False,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    # Random init leaves the norms at 1.0, which hides any q_norm/k_norm bug.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    path = tmp_path_factory.mktemp("checkpoint")
    model.save_pretrained(path)
    return str(path)


def _native_tokens(model_path, prompts, steps):
    """The baseline engine's loop, which is what the judge's reference does."""
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )
        .eval()
    )
    current = torch.tensor(prompts, dtype=torch.int64)
    cache = None
    out = []
    with torch.inference_mode():
        for _ in range(steps):
            result = model(
                input_ids=current, past_key_values=cache,
                use_cache=True, logits_to_keep=1, return_dict=True,
            )
            current = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache = result.past_key_values
            out.append(current[:, 0].tolist())
    return out


def _prompts(batch, length, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=generator).tolist()


@pytest.mark.parametrize(
    "batch,length,steps",
    [(1, 24, 8), (4, 16, 6), (3, 7, 5), (2, 33, 12)],
)
def test_matches_native(model_path, batch, length, steps):
    prompts = _prompts(batch, length, seed=batch * 100 + length)
    expected = _native_tokens(model_path, prompts, steps)

    engine = Engine(model_path)
    assert engine._fast, "custom path should be live"
    got = list(engine.generate(prompts, steps))

    assert len(got) == steps
    assert all(len(row) == batch for row in got)
    assert got == expected


def test_repeated_calls_do_not_share_state(model_path):
    """Warmup then two different prompts: no cached content may leak between them."""
    engine = Engine(model_path)
    first, second = _prompts(2, 20, 1), _prompts(2, 20, 2)

    list(engine.generate(first, 4))  # warmup, as the platform does
    got_first = list(engine.generate(first, 6))
    got_second = list(engine.generate(second, 6))

    assert got_first == _native_tokens(model_path, first, 6)
    assert got_second == _native_tokens(model_path, second, 6)
    assert got_first != got_second


def test_batch_split_prefill_matches(model_path, monkeypatch):
    """Force the prefill to split on batch; the cache offsets must still line up."""
    import engine as engine_module

    monkeypatch.setattr(engine_module, "PREFILL_ROW_BUDGET", 16)
    prompts = _prompts(6, 8, seed=99)
    engine = Engine(model_path)
    assert list(engine.generate(prompts, 5)) == _native_tokens(model_path, prompts, 5)


def test_prefill_logits_track_native(model_path):
    """The whole layer stack, judged on logits rather than on the argmax.

    Token equality is blind to drift that has not yet flipped a choice; this is
    what notices a cast moved or a residual scaled.
    """
    prompts = _prompts(2, 20, seed=11)
    native = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    with torch.inference_mode():
        reference = native(
            input_ids=torch.tensor(prompts), logits_to_keep=1, return_dict=True
        ).logits[:, -1, :].float()

    engine = Engine(model_path)
    # Native 4.51.3 explicitly repeats KV heads before SDPA. enable_gqa=True
    # can select a different CPU attention kernel, so use native's layout for
    # this bit-exact arithmetic test. Other generation tests cover GQA.
    engine._enable_gqa = False
    engine._ensure(2, 20, 4)
    with torch.no_grad():
        ids = engine._upload(prompts, 2, 20)
        hidden = torch.nn.functional.embedding(ids, engine.embed)
        normed = engine._blocks(
            hidden.view(40, engine.hidden), engine.arange[:20], 20, 2, 0,
            engine._attend_prefill,
        )
        mine = torch.nn.functional.linear(
            normed.view(2, 20, -1)[:, -1, :], engine.embed
        ).float()

    # The reference-kernel path computes the same function op for op, so this
    # is exact. Anything less means a cast moved or an operand changed.
    worst = (mine - reference).abs().max().item()
    assert torch.equal(mine, reference), f"prefill logits drifted by {worst}"


def test_teacher_forced_replay(model_path):
    """The judge's rule, run locally: replay our own tokens through native Qwen.

    Every emitted token must be the argmax of the native logits at that
    position on our own prefix, or within the tie margin of it.
    """
    tie_margin = 2.0
    batch, length, steps = 2, 16, 10
    prompts = _prompts(batch, length, seed=13)

    engine = Engine(model_path)
    emitted = list(engine.generate(prompts, steps))

    full = torch.tensor(prompts, dtype=torch.int64)
    full = torch.cat([full, torch.tensor(emitted, dtype=torch.int64).T], dim=1)

    native = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    with torch.inference_mode():
        logits = native(input_ids=full, return_dict=True).logits.float()

    worst = 0.0
    for step in range(steps):
        at = length + step - 1  # the position whose logits chose this token
        row = logits[:, at, :]
        chosen = torch.tensor(emitted[step], dtype=torch.int64)
        gap = row.max(dim=-1).values - row.gather(1, chosen[:, None])[:, 0]
        worst = max(worst, gap.max().item())
    assert worst <= tie_margin, f"worst logit gap {worst} exceeds {tie_margin}"
    assert worst < 1e-3, f"gap {worst} is inside the margin but not near zero"


def test_fallback_is_intact(model_path):
    """The native model must survive the relayout; it is the only safety net."""
    engine = Engine(model_path)
    prompts = _prompts(2, 12, seed=5)
    assert list(engine._native_generate(prompts, 4)) == _native_tokens(
        model_path, prompts, 4
    )


def test_speculation_is_exact(model_path, monkeypatch):
    """Speculative decoding must emit the identical greedy stream."""
    import engine as engine_module

    monkeypatch.setattr(engine_module, "DRAFT", 4)
    monkeypatch.setattr(engine_module, "DRAFT_TARGET", 99.0)  # never governed
    tuned_rows = []
    choose = Engine._choose_matmuls

    def record_rows(self, rows):
        tuned_rows.append(rows)
        return choose(self, rows)

    monkeypatch.setattr(Engine, "_choose_matmuls", record_rows)
    prompts = _prompts(1, 24, seed=31)
    expected = _native_tokens(model_path, prompts, 12)

    engine = Engine(model_path)
    got = list(engine.generate(prompts, 12))
    # draft is chosen per shape in _ensure, so it is only meaningful after a call
    assert engine.draft == 4, "speculation should be live at batch 1"
    assert tuned_rows[-1] == 5, "verification must not select a batch-one GEMV"
    assert len(got) == 12
    assert got == expected
    assert engine.drafter.passes < 12, "no pass ever emitted more than one token"


def test_speculation_accepts_a_repeated_prompt(model_path, monkeypatch):
    """A prompt built from a repeated phrase should make the drafter land."""
    import engine as engine_module
    from kernels.ngram import NgramDrafter

    monkeypatch.setattr(engine_module, "DRAFT", 6)
    monkeypatch.setattr(engine_module, "DRAFT_TARGET", 99.0)
    phrase = [11, 22, 33, 44, 55, 66]
    prompts = [(phrase * 8)[:40]]
    expected = _native_tokens(model_path, prompts, 10)

    engine = Engine(model_path)
    got = list(engine.generate(prompts, 10))
    assert got == expected
    assert engine.drafter.rate > 1.0, "drafter never landed on a repetitive prompt"


def test_governor_holds_the_rate_down():
    """The governor must stop proposing once the running rate hits target."""
    from kernels.ngram import NgramDrafter

    drafter = NgramDrafter(order=2, draft=8, target=1.25)
    phrase = list(range(50)) * 4
    drafter.reset(phrase)
    for _ in range(40):
        proposal = drafter.propose()
        drafter.commit([phrase[0]] * (len(proposal) + 1) if proposal else [phrase[0]])
    assert drafter.rate <= 1.35, f"governor let the rate reach {drafter.rate}"


def test_short_verification_accept_reject_and_fallback(model_path, monkeypatch):
    """Mix accepted drafts, partial/full rejection and ordinary decode on one prefix."""
    import engine as engine_module
    from kernels.ngram import NgramDrafter

    monkeypatch.setattr(engine_module, "SHORT_DRAFT", 2)
    prompts = _prompts(1, 24, seed=71)
    expected = _native_tokens(model_path, prompts, 16)
    flat = [row[0] for row in expected]
    calls = []

    def scripted(drafter):
        at = len(drafter.context) - len(prompts[0])
        proposal = flat[at:at + 2]
        mode = len(calls) % 4
        calls.append(mode)
        if mode == 1 and len(proposal) > 1:
            proposal[1] ^= 1
        elif mode == 2 and proposal:
            proposal[0] ^= 1
        elif mode == 3:
            proposal = []
        return proposal

    monkeypatch.setattr(NgramDrafter, "propose", scripted)
    engine = Engine(model_path)
    got = list(engine.generate(prompts, len(expected)))
    assert got == expected
    stats = engine.short_verifier.stats
    assert stats["accepted"] > 0
    assert stats["verify_passes"] > 0
    assert stats["ordinary_passes"] > 0
    assert set(calls) == {0, 1, 2, 3}
    # A new call must overwrite the speculative cache suffix and reset state.
    calls.clear()
    assert list(engine.generate(prompts, len(expected))) == expected


def test_short_verification_rebuilds_after_shape_changes(model_path, monkeypatch):
    import engine as engine_module

    monkeypatch.setattr(engine_module, "SHORT_DRAFT", 2)
    engine = Engine(model_path)
    for batch, length, steps in ((1, 18, 7), (3, 13, 4), (1, 27, 9), (1, 27, 1)):
        prompts = _prompts(batch, length, seed=length)
        assert list(engine.generate(prompts, steps)) == _native_tokens(model_path, prompts, steps)
        assert (engine.short_verifier is not None) == (batch == 1)


def test_verification_tuner_keeps_each_projection_in_its_family(model_path, monkeypatch):
    """Verification widths must race only the weight family ordinary decode chose."""
    import engine as engine_module

    engine = Engine(model_path)
    packed = [object() for _ in engine.layers]
    engine.quantised = {"gate_up": packed, "qkv": [object()] * engine.n_layers}
    engine.families = {"qkv": "bf16", "o": "bf16", "gate_up": "fp8",
                       "down": "bf16", "lm_head": "bf16"}
    monkeypatch.setattr(engine_module, "HAVE_TRITON", True)
    monkeypatch.setattr(engine_module, "USE_FP8", True)
    monkeypatch.setattr(engine_module, "TUNE_MATMUL", True)
    calls = []

    def choose(rows, weights, *, packed, family, **kwargs):
        calls.append((rows, family, packed))
        if family == "fp8":
            return (lambda x, w: x), "fp8", "fp8 verification test"
        return torch.nn.functional.linear, False, "BF16 verification test"

    fusion = []
    monkeypatch.setattr(engine_module, "pick_matmul", choose)
    monkeypatch.setattr(engine, "_choose_mlp", lambda rows, family="bf16": fusion.append(family))
    engine._choose_matmuls(3, families=engine.families)
    assert [c[1] for c in calls] == ["bf16", "bf16", "fp8", "bf16", "bf16"]
    assert calls[2][2] is packed and engine.operand["gate_up"] is packed
    assert calls[0][2] is None, "a BF16 projection never sees FP8 operands"
    assert fusion == ["fp8"], "the fused MLP must stay in gate/up's family"
    assert engine.families["gate_up"] == "fp8", "verification must not rewrite the families"

    def refuse(rows, weights, *, packed, family, **kwargs):
        raise RuntimeError("no FP8 kernel passed")

    monkeypatch.setattr(engine_module, "pick_matmul", refuse)
    with pytest.raises(RuntimeError):
        engine._choose_matmuls(3, families=engine.families)


def test_zero_output_does_no_work():
    engine = Engine.__new__(Engine)
    assert list(engine.generate([[1, 2]], 0)) == []


@pytest.mark.parametrize("rows", [3, 5, 9, 17])
def test_cuda_mlp_rejects_unsupported_rows_before_launch(rows):
    """Rounded-up CUDA specializations would read/write beyond these buffers."""
    from kernels.cuda_mlp import gate_up_swiglu

    x = torch.empty(rows, 8, dtype=torch.bfloat16)
    w = torch.empty(16, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires 1, 2, 4, 8, or 16 rows"):
        gate_up_swiglu(x, w)


def test_mlp_fusion_must_beat_selected_projection(monkeypatch):
    """Beating plain cuBLAS must not replace an even faster selected runner."""
    from types import SimpleNamespace
    import engine as engine_module
    from kernels import timing

    engine = Engine.__new__(Engine)
    engine.hidden, engine.device = 4, "cpu"
    engine.fused_mlp = engine.fused_operand = None
    weight = torch.ones(8, 4, dtype=torch.bfloat16)
    engine.layers = [SimpleNamespace(gate_up=weight)]
    # A transposed operand makes accidentally timing F.linear observable.
    transposed = weight.t().contiguous()
    calls = []

    def projection(x, w):
        assert w is transposed
        calls.append("selected")
        return x @ w

    engine.matmul = {"gate_up": projection}
    engine.operand = {"gate_up": [transposed]}
    monkeypatch.setattr(engine_module, "cuda_mlp", SimpleNamespace(
        ready=lambda: True, CONFIGS=[None],
        gate_up_swiglu=lambda x, w, config: engine_module.swiglu(x @ w.t()),
    ))

    def clock(fn, operands, **kwargs):
        before = len(calls)
        for operand in operands:
            fn(operand)
        return 1.0 if len(calls) > before else 1.5

    monkeypatch.setattr(timing, "time_calls", clock)
    engine._choose_mlp(1)
    assert calls == ["selected"]
    assert engine.fused_mlp is None


def test_tiled_weight_layout_round_trips():
    """The tiled INT8 layout is a permutation of the row-major fragment order."""
    import importlib

    torch.manual_seed(0)
    cuda_fp8 = importlib.import_module("kernels.cuda_fp8")   # the package hides it without Triton

    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    packed = cuda_fp8.quantize(weight)
    rowmajor = cuda_fp8.prepare(packed, tiled=False)
    tiled = cuda_fp8.prepare(packed, tiled=True)
    assert tiled.tiled and not rowmajor.tiled
    assert tiled.weight.shape == rowmajor.weight.shape
    assert not torch.equal(tiled.weight, rowmajor.weight)
    assert torch.equal(tiled.row_major(), rowmajor.weight)
    # One tile is the 16 rows x 128 bytes of one group: half, j, g, t, byte.
    tile = tiled.weight.view(4, 2, 2, 2, 8, 4, 16)[1, 0]
    rows = rowmajor.weight.view(4, 16, 2, 2, 4, 16)[1, :, 0]   # tile 1, group 0: row, j, t, byte
    assert torch.equal(tile[0, 0, 3, 2], rows[3, 0, 2])        # half 0 -> rows 0-7
    assert torch.equal(tile[1, 1, 5, 1], rows[13, 1, 1])       # half 1 -> rows 8-15
    assert tiled.bytes_moved() == rowmajor.bytes_moved()
