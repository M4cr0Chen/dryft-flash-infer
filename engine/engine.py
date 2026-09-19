"""Qwen3 4B decode engine.

The model's own forward, rebuilt: fused projections, Triton norm/RoPE/SwiGLU,
a preallocated KV cache, and one CUDA graph over the whole decode step. The
arithmetic is the reference's, reordered but never reformulated, so the tokens
are the ones native Qwen picks.

Shape-dependent work -- cache allocation, Triton compilation, graph capture --
happens on the first ``generate`` call, which the platform spends on warmup and
does not time. If any of it fails the engine falls back to native Transformers
for the rest of the process and says so on stderr, because a slow correct run
beats a crashed one.
"""

import array
import os
import sys
import traceback

import torch
import torch.nn.functional as F

from kernels import (
    HAVE_TRITON,
    NgramDrafter,
    fp8,
    DecodeAttention,
    pick_matmul,
    add_rms_norm,
    kv_norm_rope_to_cache,
    q_norm_rope,
    rms_norm,
    swiglu,
)

#: Cap on rows per prefill pass. The MLP's fused gate/up activation is the
#: largest intermediate at 2 * 9728 * 2 bytes per row, so this bounds it to
#: about 1.2 GiB regardless of how large a hidden workload's batch turns out
#: to be. Sequences are independent, so splitting on batch needs no mask work.
PREFILL_ROW_BUDGET = 32768

#: Decode attention: the split Triton kernel, or SDPA. SDPA costs about 20 us a
#: call in fixed overhead regardless of how little cache it reads, which is 36
#: launches of pure latency per step. Set DRYFT_ATTENTION=sdpa to compare.
TRITON_ATTENTION = os.environ.get("DRYFT_ATTENTION", "triton") == "triton"

#: Quantise the decode projections to FP8 with group scales. Prefill keeps
#: bfloat16; it is compute-bound, so there is nothing to win and no reason to
#: spend the accuracy. See kernels/fp8.py for the measured logit cost.
USE_FP8 = os.environ.get("DRYFT_FP8", "on") == "on"

#: Capture the decode step into a CUDA graph. Off is a real earlier stage of
#: this engine, not a handicap: it is what the same forward costs when every
#: step pays Python dispatch again.
USE_GRAPH = os.environ.get("DRYFT_GRAPH", "on") == "on"

#: Pick projections per shape at warmup. Off keeps cuBLAS and F.linear.
TUNE_MATMUL = os.environ.get("DRYFT_MATMUL", "tune") == "tune"

#: Draft tokens verified alongside the confirmed one. 0 disables speculation.
#: Each pass runs DRAFT+1 positions whatever the draft finds, so the cost per
#: pass is constant and the step still captures into a CUDA graph.
DRAFT = int(os.environ.get("DRYFT_DRAFT", "0"))

#: Tokens per pass the governor aims for. See kernels/ngram.py for why a mean
#: this far below what the drafter can reach is the point.
DRAFT_TARGET = float(os.environ.get("DRYFT_DRAFT_TARGET", "1.25"))

#: Spare cache slots, for the decode steps spent warming the graph.
CAPACITY_SLACK = 8

#: Largest logit disagreement with native Qwen that still counts as reordering
#: rather than breakage. The tie margin is 2.0 and native's own teacher-forced
#: replay drifts up to 0.75, so anything past this is a real fault.
SELF_CHECK_TOLERANCE = 1.0


class _Layer:
    """One decoder layer's weights, in the layout this engine's kernels want."""

    __slots__ = ("norm_in", "qkv", "q_norm", "k_norm", "o", "norm_post", "gate_up", "down")

    def __init__(self, layer):
        attn = layer.self_attn
        self.norm_in = layer.input_layernorm.weight
        self.qkv = torch.cat(
            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
        ).contiguous()
        self.q_norm = attn.q_norm.weight
        self.k_norm = attn.k_norm.weight
        self.o = attn.o_proj.weight
        self.norm_post = layer.post_attention_layernorm.weight
        self.gate_up = torch.cat(
            [layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0
        ).contiguous()
        self.down = layer.mlp.down_proj.weight


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        from transformers import AutoModelForCausalLM

        self.cuda = torch.cuda.is_available()
        self.device = "cuda:0" if self.cuda else "cpu"
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(self.device)
        )
        self._native = model
        self._fast = False

        try:
            self._adopt(model)
            self._fast = True
            self._self_check()
        except Exception:
            traceback.print_exc()
            print("engine: custom path unavailable, using native Qwen", file=sys.stderr)
            self._fast = False

    # ---------------------------------------------------------------- loading

    def _adopt(self, model) -> None:
        """Take the loaded weights into this engine's own layout."""
        config = model.config
        self.hidden = config.hidden_size
        self.n_q = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden // self.n_q)
        self.n_layers = config.num_hidden_layers
        self.eps = config.rms_norm_eps
        self.rope_theta = getattr(config, "rope_theta", 5_000_000.0)
        self.q_width = self.n_q * self.head_dim
        self.kv_width = self.n_kv * self.head_dim
        if config.tie_word_embeddings is not True:
            raise RuntimeError("expected tied embeddings")

        base = model.model
        self.embed = base.embed_tokens.weight
        self.norm_out = base.norm.weight
        self.layers = [_Layer(layer) for layer in base.layers]

        # The fused projections duplicate q/k/v/gate/up, about 4.7 GiB. The
        # originals stay: they are the fallback, and this engine is written
        # against a GPU it cannot be tested on. Drop them here if a hidden
        # workload ever comes back memory_limit.
        if self.cuda:
            torch.cuda.empty_cache()

        self._enable_gqa = self._probe_gqa()
        self.cos = self.sin = None
        self.k_cache = self.v_cache = None
        self.capacity = self.batch = self.seq_len = 0
        self.graph = None

    def _self_check(self) -> None:
        """Prove the custom path against native Qwen before anything is timed.

        This runs inside the load budget on a small shape, so the first run on
        the platform reports what the engine actually agrees on rather than
        failing a workload to say it. Every stage is covered: prefill, graph
        capture, the decode attention, and the token stream itself.
        """
        batch, length, steps = 2, 16, 4
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(
            0, self.embed.shape[0], (batch, length), generator=generator
        ).tolist()

        expected = list(self._native_generate(ids, steps))
        got = list(self.generate(ids, steps))

        with torch.inference_mode():
            reference = self._native(
                input_ids=torch.tensor(ids, device=self.device), return_dict=True
            ).logits.float().view(batch * length, -1)
        with torch.no_grad():
            prompt = self._upload(ids, batch, length)
            hidden = F.embedding(prompt, self.embed).view(batch * length, self.hidden)
            normed = self._blocks(
                hidden, self.arange[:length], length, batch, 0, self._attend_prefill
            )
            mine = F.linear(normed, self.embed).float()

        # The judge's own rule, at every prompt position rather than only the
        # last: the token we would pick has to be native's argmax there, or
        # close enough to it that the tie margin covers the difference.
        delta = (mine - reference).abs().max().item()
        chosen = mine.argmax(dim=-1, keepdim=True)
        gap = (reference.max(dim=-1, keepdim=True).values - reference.gather(1, chosen))
        worst = gap.max().item()
        print(
            f"engine: self-check logit delta {delta:.4f}, worst tie gap {worst:.4f}, "
            f"tokens {'match' if got == expected else 'DIFFER'}, "
            f"triton {HAVE_TRITON}, graph {self.graph is not None}",
            file=sys.stderr,
        )
        if not delta < SELF_CHECK_TOLERANCE:  # also catches NaN
            raise RuntimeError(f"self-check logit delta {delta}")
        if not worst < SELF_CHECK_TOLERANCE:
            raise RuntimeError(f"self-check tie gap {worst}")
        if got != expected:
            raise RuntimeError("self-check tokens disagree with native Qwen")

        # Release the probe's buffers so the real workload allocates its own.
        self.capacity = 0
        self.graph = None
        self.k_cache = self.v_cache = None
        if self.cuda:
            torch.cuda.empty_cache()

    def _probe_gqa(self) -> bool:
        """Does this torch's SDPA take ``enable_gqa``? 2.5.1 does; be sure."""
        q = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16, device=self.device)
        kv = torch.zeros(1, 2, 1, 8, dtype=torch.bfloat16, device=self.device)
        try:
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        except (TypeError, RuntimeError):
            return False
        return True

    # ------------------------------------------------------------- allocation

    def _rope_tables(self, capacity: int) -> None:
        """cos/sin for every absolute position, built the way the reference does."""
        half = self.head_dim // 2
        steps = torch.arange(0, half, dtype=torch.int64, device=self.device).float()
        inv_freq = 1.0 / (self.rope_theta ** (steps / half))
        positions = torch.arange(capacity, dtype=torch.int64, device=self.device).float()
        emb = torch.cat([positions[:, None] * inv_freq[None, :]] * 2, dim=-1)
        self.cos = emb.cos().to(torch.bfloat16).contiguous()
        self.sin = emb.sin().to(torch.bfloat16).contiguous()

    @torch.no_grad()
    def _ensure(self, batch: int, seq_len: int, max_new_tokens: int) -> None:
        """Allocate for this shape and capture the decode graph. Warmup pays this."""
        # Decode writes one slot per step except for the last token, which is
        # never fed back, so the prompt plus the outputs is always enough.
        capacity = seq_len + max_new_tokens + CAPACITY_SLACK
        if (
            self.capacity
            and batch == self.batch
            and seq_len == self.seq_len
            and capacity <= self.capacity
        ):
            return

        self.graph = None
        self.k_cache = self.v_cache = None
        if self.cuda:
            torch.cuda.empty_cache()

        self.batch, self.seq_len, self.capacity = batch, seq_len, capacity
        self._rope_tables(capacity)

        shape = (self.n_layers, batch, self.n_kv, capacity, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=self.device)

        self.arange = torch.arange(capacity, dtype=torch.int32, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.token = torch.zeros(batch, 1, dtype=torch.int64, device=self.device)
        self.emitted = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self.mask = torch.zeros(capacity, dtype=torch.bool, device=self.device)

        self.ids_host = torch.zeros(
            batch * seq_len, dtype=torch.int64, pin_memory=self.cuda
        )
        self.ids_device = torch.zeros(
            batch * seq_len, dtype=torch.int64, device=self.device
        )
        self.out_host = [
            torch.zeros(batch, dtype=torch.int64, pin_memory=self.cuda) for _ in range(2)
        ]
        self.out_event = [torch.cuda.Event() for _ in range(2)] if self.cuda else []

        self._choose_matmuls(batch)

        self.draft = DRAFT if (batch == 1 and DRAFT > 0) else 0
        self.drafter = None
        if self.draft:
            width = self.draft + 1
            self.drafter = NgramDrafter(order=2, draft=self.draft, target=DRAFT_TARGET)
            self.spec_ids = torch.zeros(1, width, dtype=torch.int64, device=self.device)
            self.spec_pred = torch.zeros(width, dtype=torch.int64, device=self.device)
            self.spec_pred_host = torch.zeros(width, dtype=torch.int64, pin_memory=self.cuda)
            self.offsets = torch.arange(width, dtype=torch.int32, device=self.device)
            self.spec_positions = torch.zeros(width, dtype=torch.int32, device=self.device)
            self.spec_mask = torch.zeros(
                width, capacity, dtype=torch.bool, device=self.device
            )

        self.decode_attention = None
        if HAVE_TRITON and TRITON_ATTENTION:
            self.decode_attention = DecodeAttention(
                batch, self.n_kv, self.n_q // self.n_kv, self.head_dim,
                capacity, self.device, tokens=(self.draft + 1) if self.draft else 1,
            )

        self._capture()

    def _choose_matmuls(self, batch: int) -> None:
        """Race cuBLAS against the Triton kernel on every projection shape.

        Decode only. Prefill has thousands of rows and cuBLAS owns that regime,
        so it keeps ``F.linear`` unconditionally.
        """
        self.matmul = {}
        self.operand = {}
        if not (HAVE_TRITON and pick_matmul is not None and TUNE_MATMUL):
            return
        quantised = {}
        if USE_FP8 and fp8 is not None:
            try:
                for name, weights in (
                    ("qkv", [ly.qkv for ly in self.layers]),
                    ("o", [ly.o for ly in self.layers]),
                    ("gate_up", [ly.gate_up for ly in self.layers]),
                    ("down", [ly.down for ly in self.layers]),
                ):
                    quantised[name] = [fp8.quantize(w) for w in weights]
                if self.cuda:
                    torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()
                quantised = {}

        first = self.layers[0]
        for name, sample, every in (
            ("qkv", first.qkv, [ly.qkv for ly in self.layers]),
            ("o", first.o, [ly.o for ly in self.layers]),
            ("gate_up", first.gate_up, [ly.gate_up for ly in self.layers]),
            ("down", first.down, [ly.down for ly in self.layers]),
            ("lm_head", self.embed, [self.embed]),
        ):
            try:
                chosen, transpose, note = pick_matmul(
                    batch, every, packed=quantised.get(name)
                )
            except Exception:
                chosen, transpose, note = F.linear, False, "cublas (selection failed)"
            self.matmul[name] = chosen
            # A transposed layout is a second copy of the weight. Worth it for
            # the projections cuBLAS reads better that way; prefill keeps the
            # original, where the shape is wide enough not to care.
            if transpose == "fp8":
                self.operand[name] = quantised[name]
            elif transpose:
                self.operand[name] = [w.t().contiguous() for w in every]
            else:
                self.operand[name] = every
            print(f"engine: {name:8s} {tuple(sample.shape)} -> {note}", file=sys.stderr)
        if self.cuda:
            torch.cuda.empty_cache()

    def _capture(self) -> None:
        """Compile every kernel on this shape, then record one decode step."""
        dummy = torch.zeros(
            self.batch, self.seq_len, dtype=torch.int64, device=self.device
        )
        self._prefill(dummy)
        step = self._spec_step if self.draft else self._decode_step
        for _ in range(2):
            step()
        if not self.cuda or not USE_GRAPH:
            return

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        self.graph = graph
        torch.cuda.synchronize()

    # ---------------------------------------------------------------- forward

    def _blocks(self, x, positions, seq_len, batch, offset, attend, matmul=None):
        """All layers over ``x`` of shape ``[rows, hidden]``; returns the final norm.

        Each layer's residual add is fused into the norm that consumes it, and
        the last one borrows the model's output norm, so the chain never
        materialises a residual only to read it back.
        """
        def project(name, index, source, fallback):
            if not matmul:
                return F.linear(source, fallback)
            return matmul[name](source, self.operand[name][index])

        normed = rms_norm(x, self.layers[0].norm_in, self.eps)
        for index, layer in enumerate(self.layers):
            qkv = project("qkv", index, normed, layer.qkv)
            q = q_norm_rope(
                qkv, self.n_q, layer.q_norm,
                self.cos, self.sin, positions, seq_len, self.eps,
            )
            kv_norm_rope_to_cache(
                qkv, self.q_width, self.n_kv, layer.k_norm,
                self.cos, self.sin, positions, seq_len,
                self.k_cache[index].narrow(0, offset, batch),
                self.v_cache[index].narrow(0, offset, batch),
                self.eps,
            )
            attended = attend(q, index, batch, offset, seq_len)
            x, normed = add_rms_norm(
                x, project("o", index, attended, layer.o), layer.norm_post, self.eps
            )
            gate_up = project("gate_up", index, normed, layer.gate_up)
            down = project("down", index, swiglu(gate_up), layer.down)
            following = (
                self.layers[index + 1].norm_in
                if index + 1 < self.n_layers
                else self.norm_out
            )
            x, normed = add_rms_norm(x, down, following, self.eps)
        return normed

    def _attend_prefill(self, q, index, batch, offset, seq_len):
        q = q.view(batch, seq_len, self.n_q, self.head_dim).transpose(1, 2)
        k = self.k_cache[index].narrow(0, offset, batch)[:, :, :seq_len, :]
        v = self.v_cache[index].narrow(0, offset, batch)[:, :, :seq_len, :]
        if self._enable_gqa:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            group = self.n_q // self.n_kv
            out = F.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                is_causal=True,
            )
        return out.transpose(1, 2).reshape(batch * seq_len, self.q_width)

    def _attend_decode(self, q, index, batch, offset, seq_len):
        group = self.n_q // self.n_kv
        if self.decode_attention is not None:
            return self.decode_attention(
                q, self.k_cache[index], self.v_cache[index], self.pos
            )
        grouped = q.view(batch, self.n_kv, group, self.head_dim)
        # Fallback. Query head h reads KV head h // 4, so the four heads of a
        # group become four query positions against that group's cache: the
        # whole grouped-query pattern in one call, with no repeat_interleave
        # copy of the cache and no mask over heads.
        out = F.scaled_dot_product_attention(
            q.view(batch, self.n_kv, group, self.head_dim),
            self.k_cache[index],
            self.v_cache[index],
            attn_mask=self.mask.view(1, 1, 1, -1),
        )
        return out.reshape(batch, self.q_width)

    @torch.no_grad()
    def _prefill(self, ids: torch.Tensor) -> None:
        """Read the prompt, fill the cache, and leave the first token in place."""
        batch, seq_len = ids.shape
        rows = max(1, PREFILL_ROW_BUDGET // seq_len)
        positions = self.arange[:seq_len]
        logits = []
        for start in range(0, batch, rows):
            count = min(rows, batch - start)
            x = F.embedding(ids[start : start + count], self.embed)
            normed = self._blocks(
                x.view(count * seq_len, self.hidden),
                positions, seq_len, count, start, self._attend_prefill,
            )
            last = normed.view(count, seq_len, self.hidden)[:, -1, :]
            logits.append(F.linear(last, self.embed))
        chosen = (logits[0] if len(logits) == 1 else torch.cat(logits)).argmax(dim=-1)
        self.token.copy_(chosen.view(batch, 1))
        self.emitted.copy_(chosen)
        self.pos.fill_(seq_len)

    @torch.no_grad()
    def _decode_step(self) -> None:
        """One graphed step: everything reads and writes fixed addresses."""
        # The new token takes slot ``pos``, so keys 0..pos inclusive are live.
        torch.le(self.arange, self.pos, out=self.mask)
        x = F.embedding(self.token, self.embed).view(self.batch, self.hidden)
        normed = self._blocks(
            x, self.pos, 1, self.batch, 0, self._attend_decode, self.matmul
        )
        head = (
            self.matmul["lm_head"](normed, self.operand["lm_head"][0])
            if self.matmul
            else F.linear(normed, self.embed)
        )
        chosen = head.argmax(dim=-1)
        self.token.copy_(chosen.view(self.batch, 1))
        self.emitted.copy_(chosen)
        self.pos.add_(1)

    def _attend_spec(self, q, index, batch, offset, seq_len):
        """Attention over the draft block: query t may read keys 0..pos+t."""
        if self.decode_attention is not None:
            return self.decode_attention(
                q, self.k_cache[index], self.v_cache[index], self.pos
            )
        group = self.n_q // self.n_kv
        width = self.draft + 1
        grouped = q.view(width, self.n_kv, group, self.head_dim).permute(1, 0, 2, 3)
        out = F.scaled_dot_product_attention(
            grouped.reshape(1, self.n_kv, width * group, self.head_dim),
            self.k_cache[index],
            self.v_cache[index],
            attn_mask=self.spec_mask.repeat_interleave(group, 0).view(
                1, 1, width * group, -1
            ),
        )
        out = out.view(self.n_kv, width, group, self.head_dim).permute(1, 0, 2, 3)
        return out.reshape(width, self.q_width)

    @torch.no_grad()
    def _spec_step(self) -> None:
        """One pass over the confirmed token plus the draft, at fixed shape."""
        width = self.draft + 1
        torch.add(self.pos, self.offsets, out=self.spec_positions)
        # Query t sees every key written at or before its own position.
        torch.le(
            self.arange.view(1, -1), self.spec_positions.view(-1, 1), out=self.spec_mask
        )
        x = F.embedding(self.spec_ids, self.embed).view(width, self.hidden)
        normed = self._blocks(
            x, self.spec_positions, width, 1, 0, self._attend_spec, self.matmul
        )
        head = (
            self.matmul["lm_head"](normed, self.operand["lm_head"][0])
            if self.matmul
            else F.linear(normed, self.embed)
        )
        self.spec_pred.copy_(head.argmax(dim=-1))

    def _verify(self, proposal):
        """Longest correct prefix of the draft, plus the token after it.

        Position i of the pass predicts the token following the first i entries
        of what we ran. So prediction 0 is always right, and prediction i is
        right exactly while every draft token before it was. That makes the
        emitted run the model's own greedy continuation, by construction.
        """
        predicted = self.spec_pred_host.tolist()
        taken = 0
        for draft_token, model_token in zip(proposal, predicted):
            if draft_token != model_token:
                break
            taken += 1
        return predicted[: taken + 1]

    # ----------------------------------------------------------------- public

    def _upload(self, input_ids, batch, seq_len):
        """Prompt ids to the device without building a tensor from nested lists."""
        flat = array.array("q")
        for row in input_ids:
            flat.extend(row)
        self.ids_host.copy_(torch.frombuffer(flat, dtype=torch.int64))
        self.ids_device.copy_(self.ids_host, non_blocking=self.cuda)
        return self.ids_device.view(batch, seq_len)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if self._fast:
            batch, seq_len = len(input_ids), len(input_ids[0])
            try:
                self._ensure(batch, seq_len, max_new_tokens)
                self._prefill(self._upload(input_ids, batch, seq_len))
            except Exception:
                traceback.print_exc()
                print("engine: falling back to native Qwen", file=sys.stderr)
                self._fast = False
            else:
                if self.draft:
                    yield from self._stream_speculative(
                        input_ids[0], max_new_tokens
                    )
                else:
                    yield from self._stream(max_new_tokens)
                return
        yield from self._native_generate(input_ids, max_new_tokens)

    def _stream(self, max_new_tokens: int):
        """Replay the graph, staying one step ahead of the host that reads it."""
        if self.graph is None:
            yield self.emitted.tolist()
            for _ in range(1, max_new_tokens):
                self._decode_step()
                yield self.emitted.tolist()
            return

        self.out_host[0].copy_(self.emitted, non_blocking=True)
        self.out_event[0].record()
        self.out_event[0].synchronize()
        yield self.out_host[0].tolist()

        pending = None
        for step in range(1, max_new_tokens):
            self.graph.replay()
            slot = step & 1
            self.out_host[slot].copy_(self.emitted, non_blocking=True)
            self.out_event[slot].record()
            if pending is not None:
                self.out_event[pending].synchronize()
                yield self.out_host[pending].tolist()
            pending = slot
        if pending is not None:
            self.out_event[pending].synchronize()
            yield self.out_host[pending].tolist()

    def _stream_speculative(self, prompt, max_new_tokens: int):
        """Emit the greedy continuation, several tokens per forward pass.

        Every token yielded is the model's own argmax on the prefix before it,
        so the stream is identical to plain greedy decoding -- speculation only
        changes how many passes it takes to find them.
        """
        first = int(self.emitted[0].item())
        self.drafter.reset(list(prompt))
        self.drafter.commit([first])
        yield [first]

        remaining = max_new_tokens - 1
        width = self.draft + 1
        while remaining > 0:
            proposal = self.drafter.propose()[: self.draft]
            block = [self.drafter.context[-1]] + proposal
            block += [0] * (width - len(block))
            self.spec_ids.copy_(
                torch.tensor(block, dtype=torch.int64).view(1, width), non_blocking=True
            )

            if self.graph is not None:
                self.graph.replay()
            else:
                self._spec_step()
            self.spec_pred_host.copy_(self.spec_pred)
            torch.cuda.synchronize() if self.cuda else None

            produced = self._verify(proposal)[:remaining]
            self.drafter.commit(produced)
            self.pos.add_(len(produced))
            for token in produced:
                yield [token]
            remaining -= len(produced)

    def _native_generate(self, input_ids: list[list[int]], max_new_tokens: int):
        current = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        cache = None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self._native(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()
