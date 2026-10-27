"""Sampling text from the model, one token at a time."""

import numpy as np
from nnkit.base import asarray, to_numpy

from ..model import Llama4Model


def generate(
    model: Llama4Model,
    prompt: list[int],
    max_new: int,
    eos: int,
    temperature: float = 0.8,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """The ids that continue ``prompt``, up to ``max_new`` of them or until EOS (not included).

    ``temperature=0`` is greedy. There is no KV cache: every step reruns the
    last ``seq_len`` tokens, which is fine for short samples during training.
    The prompt is one document, so every doc id is 0.
    """
    rng = rng if rng is not None else np.random.default_rng()
    was_training = model.training
    model.eval()
    ids, new = list(prompt), []
    for _ in range(max_new):
        window = np.array([ids[-model.cfg.seq_len :]], dtype=np.int32)
        logits = model.forward(asarray(window), asarray(np.zeros_like(window)))
        last = to_numpy(logits)[0, -1].astype(np.float64)
        if temperature == 0:
            token = int(last.argmax())
        else:
            p = np.exp((last - last.max()) / temperature)
            token = int(rng.choice(len(p), p=p / p.sum()))
        if token == eos:
            break
        ids.append(token)
        new.append(token)
    model.train(was_training)

    return new
