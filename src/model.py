"""Llama 4-style decoder: iRoPE attention and mixture-of-experts feed-forward."""

from nnkit.activations import softcap, softcap_backward
from nnkit.base import (
    Array,
    Module,
    Param,
    chunked_causal_mask,
    document_causal_mask,
    sliding_window_mask,
)
from nnkit.layers import (
    Embedding,
    Linear,
    MoEFeedForward,
    MultiHeadAttention,
    RMSNorm,
    RoPE,
    TiedProjection,
)

from .config import ModelConfig


class Block(Module):
    """Pre-norm block: self-attention, then a mixture of experts.

        h = x + attn(norm1(x), mask)
        y = h + moe(norm2(h))

    A local block rotates queries and keys with RoPE and gets the local mask.
    A global block (iRoPE's NoPE layer) has no positions and sees the whole
    document, with temperature scaling instead. Llama 4 QK-normalises only
    the RoPE layers; with ``qk_norm_gain`` every layer is normalised.
    """

    def __init__(self, cfg: ModelConfig, is_global: bool, rope: RoPE) -> None:
        super().__init__()
        self.is_global = is_global
        self.attn = MultiHeadAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.dropout,
            rope=None if is_global else rope,
            n_kv_heads=cfg.n_kv_heads,
            qk_norm=cfg.qk_norm_gain or not is_global,
            qk_norm_gain=cfg.qk_norm_gain,
            attn_scale_alpha=cfg.attn_scale_alpha if is_global else 0.0,
            attn_scale_floor=cfg.local_size if is_global else None,
            gate=cfg.attn_gate,
        )
        self.norm1 = RMSNorm(cfg.d_model)
        self.moe = MoEFeedForward(
            cfg.d_model,
            cfg.n_experts,
            cfg.d_ff_expert,
            top_k=cfg.top_k,
            normalize_gates=cfg.normalize_gates,
            bias_update_rate=cfg.balance_rate,
            dropout=cfg.dropout,
            gate_input=cfg.moe_gate_input,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.register_modules(self.attn, self.norm1, self.moe, self.norm2)

    def forward(self, x: Array, mask: Array) -> Array:
        normed = self.norm1.forward(x)
        x = x + self.attn.forward(normed, normed, mask)

        return x + self.moe.forward(self.norm2.forward(x))

    def backward(self, dy: Array) -> Array:
        d = dy + self.norm2.backward(self.moe.backward(dy))
        d_q, d_kv = self.attn.backward(d)  # self-attention: both inputs are normed

        return d + self.norm1.backward(d_q + d_kv)


class Llama4Model(Module):
    """Token ids -> next-token logits: embedding, blocks, final RMSNorm, output head.

    Every ``cfg.nope_every``-th block is global (see ``ModelConfig.is_global``).
    Attention never crosses a document boundary of the packed batch; local
    blocks are further limited to a chunk or a sliding window.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.d_model, scale=False)
        rope = RoPE(cfg.d_model // cfg.n_heads, cfg.seq_len, cfg.rope_base)
        self.blocks = [Block(cfg, cfg.is_global(i), rope) for i in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model)
        self.head = (
            TiedProjection(self.embed)
            if cfg.tied_embeddings
            else Linear(cfg.d_model, cfg.vocab_size)
        )
        self.register_modules(self.embed, *self.blocks, self.norm, self.head)
        self._cache = None

    def masks(self, doc_ids: Array) -> tuple[Array, Array]:
        """(local, global) masks for ``doc_ids`` (batch, T), each (batch, 1, T, T)."""
        size, local = doc_ids.shape[1], self.cfg.local_size
        doc = document_causal_mask(doc_ids)
        if self.cfg.local_attn == "chunked":
            return doc & chunked_causal_mask(size, local), doc

        return doc & sliding_window_mask(size, local), doc

    def forward(self, ids: Array, doc_ids: Array) -> Array:
        """``ids``, ``doc_ids`` (batch, T) int -> logits (batch, T, vocab)."""
        local_mask, global_mask = self.masks(doc_ids)
        x = self.embed.forward(ids)
        for block in self.blocks:
            x = block.forward(x, global_mask if block.is_global else local_mask)
        logits = self.head.forward(self.norm.forward(x))
        if self.cfg.logit_cap:
            logits = softcap(logits, self.cfg.logit_cap)
        self._cache = (ids, logits)

        return logits

    def backward(self, d_logits: Array) -> None:
        """Accumulate every parameter's gradient from the loss gradient w.r.t. the logits."""
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        ids, logits = self._cache
        if self.cfg.logit_cap:
            d_logits = softcap_backward(logits, d_logits, self.cfg.logit_cap)
        d = self.norm.backward(self.head.backward(d_logits))
        for block in reversed(self.blocks):
            d = block.backward(d)

        self.embed.backward(d, ids)

    def update_bias(self) -> None:
        """Nudge every MoE layer's load-balancing bias; call after each optimizer step."""
        for block in self.blocks:
            block.moe.update_bias()

    def non_muon_matrices(self) -> list[Param]:
        """2-D parameters that belong to AdamW, not Muon: the embedding, an untied head, routers."""
        out = [self.embed.weight] + [block.moe.router.weight for block in self.blocks]
        if not self.cfg.tied_embeddings:
            out.append(self.head.weight)

        return out
