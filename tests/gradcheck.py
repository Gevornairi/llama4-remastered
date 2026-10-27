"""Gradient check for the Llama 4 model: hand-written backward against reference gradients.

NNKIT_BACKEND=numpy NNKIT_DTYPE=float64 python -m tests.gradcheck   # finite differences
python -m tests.gradcheck                                           # MLX autodiff

Exits non-zero if any check fails. Tiny sizes, but every code path: local and
global blocks, both QK-norm placements, both MoE gatings, tied and untied heads,
the logit soft-cap and gated attention.
"""

from dataclasses import replace

import numpy as np
from nnkit.base import BACKEND, Param, asarray, seed, to_numpy, xp
from nnkit.losses import LabelSmoothingLoss
from nnkit.training import packed_batch

from src.config import LLAMA4, REMASTERED, ModelConfig
from src.model import Llama4Model

EOS = 1
# float64 finite differences are exact to ~1e-6; MLX's GPU matmuls to ~1e-3
TOLERANCE = 1e-2 if BACKEND == "mlx" else 1e-5


def tiny(cfg: ModelConfig, **changes) -> ModelConfig:
    """``cfg``'s choices at a size a gradcheck can afford: one local and one global block."""
    sizes = {
        "vocab_size": 11,
        "seq_len": 12,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 1,
        "nope_every": 2,
        "local_size": 4,  # several chunks, and t + 1 >= floor so the temperature varies
        "d_ff_expert": 8,
        "n_experts": min(cfg.n_experts, 4),
        "top_k": min(cfg.top_k, 2),
    }
    return replace(cfg, **sizes | changes)


def batch(cfg: ModelConfig):
    """Two rows cut from a packed stream, so the document mask matters."""
    stream = np.array([5, 9, 3, EOS, 7, 8, 2, 4, 6, EOS, 3, 10, 9, EOS] * 3)
    ids, targets, doc_ids = packed_batch(stream, np.array([0, 9]), cfg.seq_len, EOS)
    return asarray(ids), asarray(targets), asarray(doc_ids)


def reference_grads(params: list[Param], loss_of_model) -> list[np.ndarray]:
    """MLX autodiff, or central differences on NumPy."""
    if BACKEND == "mlx":

        def loss_of_values(values):
            for p, v in zip(params, values):
                p.value = v
            return loss_of_model()

        return [to_numpy(g) for g in xp.grad(loss_of_values)([p.value for p in params])]

    reference = []
    for p in params:
        num = np.zeros(p.shape)
        for ix in np.ndindex(*p.shape):
            old = p.value[ix]
            p.value[ix] = old + 1e-6
            loss_plus = loss_of_model().item()
            p.value[ix] = old - 1e-6
            loss_minus = loss_of_model().item()
            p.value[ix] = old
            num[ix] = (loss_plus - loss_minus) / 2e-6
        reference.append(num)
    return reference


def check_model(cfg: ModelConfig) -> float:
    """Largest gradient error relative to the largest gradient, over all parameters."""
    seed(0)
    model = Llama4Model(cfg)
    loss_fn = LabelSmoothingLoss(cfg.vocab_size, pad_id=0, smoothing=0.0)
    ids, targets, doc_ids = batch(cfg)
    params = model.parameters()

    def loss_of_model():
        return loss_fn(model.forward(ids, doc_ids), targets)[0]

    model.zero_grad()
    _, d_logits = loss_fn(model.forward(ids, doc_ids), targets)
    model.backward(d_logits)
    reference = reference_grads(params, loss_of_model)

    scale = max(np.abs(r).max() for r in reference)
    errors = [np.abs(to_numpy(p.grad) - r).max() for p, r in zip(params, reference)]
    return float(max(errors)) / scale


def check_layer_pattern() -> None:
    """Global blocks have no RoPE; Llama 4 QK-norms RoPE blocks only."""
    model = Llama4Model(LLAMA4)
    assert [b.is_global for b in model.blocks] == [False, False, False, True] * 2
    for b in model.blocks:
        assert (b.attn.rope is None) == b.is_global
    assert len(model.non_muon_matrices()) == 1 + LLAMA4.n_layers + 1
    tied = Llama4Model(REMASTERED)
    assert len(tied.non_muon_matrices()) == 1 + REMASTERED.n_layers
    assert tied.head.weight is tied.embed.weight


def check_masks() -> None:
    """Local = document & chunk (or window); global = document only."""
    cfg = tiny(LLAMA4)
    _, _, doc_ids = batch(cfg)
    for local_attn in ("chunked", "sliding"):
        model = Llama4Model(replace(cfg, local_attn=local_attn))
        local, glob = (to_numpy(m) for m in model.masks(doc_ids))
        assert local.shape == glob.shape == (2, 1, cfg.seq_len, cfg.seq_len)
        assert not (local & ~glob).any(), "local sees beyond the document mask"
        i, j = np.nonzero(local[0, 0])
        if local_attn == "chunked":
            assert (i // cfg.local_size == j // cfg.local_size).all()
        else:
            assert (i - j < cfg.local_size).all()


def main():
    ok = True
    variants = {
        "llama4": tiny(LLAMA4),
        "remastered": tiny(REMASTERED),
        "llama4 + logit cap": tiny(LLAMA4, logit_cap=3.0),
        "llama4 + gated attention": tiny(LLAMA4, attn_gate="elementwise"),
        "remastered, chunked": tiny(REMASTERED, local_attn="chunked"),
    }
    for name, cfg in variants.items():
        err = check_model(cfg)
        passed = err < TOLERANCE
        ok &= passed
        print(f"{name}: {err:.2e} {'ok' if passed else 'FAIL'}")
    for check in (check_layer_pattern, check_masks):
        try:
            check()
            print(f"{check.__name__}: ok")
        except Exception as e:  # noqa: BLE001 - report every failing check
            ok = False
            print(f"{check.__name__}: FAIL {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
