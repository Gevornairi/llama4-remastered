"""Model configuration: the faithful Llama 4 baseline and the remastered variant.

Every remastered change is one field, so an ablation is the llama4 preset with
one field replaced. Sizes are scaled down for TinyStories on one Mac; Llama 4's
ratios are kept (the NoPE temperature floor equals the local chunk size).
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    seq_len: int = 512
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    dropout: float = 0.0  # pretraining on fresh tokens needs none

    # iRoPE: every nope_every-th layer attends globally without RoPE,
    # the others locally with RoPE
    nope_every: int = 4
    rope_base: float = 500_000.0
    local_attn: str = "chunked"  # "chunked" (Llama 4) or "sliding"
    local_size: int = 128  # chunk length or window length
    attn_scale_alpha: float = 0.1  # NoPE layers' temperature scaling; 0 turns it off

    # False: no gain, after RoPE, RoPE layers only (Llama 4);
    # True: learned gain, before RoPE, every layer
    qk_norm_gain: bool = False

    # mixture of experts: a shared expert plus top_k of n_experts routed ones
    n_experts: int = 4
    d_ff_expert: int = 512
    top_k: int = 1
    moe_gate_input: bool = True  # Llama 4: the router score scales the expert input
    normalize_gates: bool = False
    balance_rate: float = 0.0  # bias balancing step per optimizer step; 0 = off

    tied_embeddings: bool = False

    # further ablations, off in both presets
    attn_gate: str = "none"  # "elementwise" or "headwise": gated attention
    logit_cap: float = 0.0  # soft-cap on the output logits; 0 = off

    def __post_init__(self) -> None:
        if self.local_attn not in ("chunked", "sliding"):
            raise ValueError(f"local_attn={self.local_attn!r}: 'chunked' or 'sliding'")
        if self.nope_every < 1:
            raise ValueError("nope_every must be >= 1")
        if self.normalize_gates and self.top_k == 1:
            # the one chosen gate is s / s = 1, so the router gets no gradient
            raise ValueError("normalize_gates needs top_k > 1")

    def is_global(self, layer: int) -> bool:
        """Every nope_every-th layer (1-based) is a global NoPE layer: 3, 7, ... for 4."""
        return (layer + 1) % self.nope_every == 0


LLAMA4 = ModelConfig()

REMASTERED = replace(
    LLAMA4,
    local_attn="sliding",
    qk_norm_gain=True,
    # same active (shared + routed) and total routed expert width as LLAMA4
    n_experts=8,
    d_ff_expert=256,
    top_k=3,
    moe_gate_input=False,
    normalize_gates=True,
    balance_rate=1e-3,
    tied_embeddings=True,
)

PRESETS = {"llama4": LLAMA4, "remastered": REMASTERED}
