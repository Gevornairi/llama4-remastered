"""Optimizers for the two presets: AdamW alone, or Muon + AdamW, with their schedules."""

from collections.abc import Callable

import numpy as np
from nnkit.base import Array, xp
from nnkit.optimisers import (
    AdamW,
    CosineSchedule,
    Muon,
    WSDSchedule,
    split_muon_params,
)

from ..model import Llama4Model


class MuonAdamW:
    """Muon on the hidden weight matrices, AdamW on everything else, stepped together.

    Same interface as a single optimizer (``step``, ``zero_grad``, ``state``,
    ``load_state``, ``last_lr``), so the training loop and checkpoints don't
    care which is in use. Both follow one schedule: Muon's update RMS is
    matched to AdamW's, so one learning rate serves both. Gradient clipping
    applies to the AdamW group only; Muon's update size doesn't depend on the
    gradient's.
    """

    def __init__(self, muon: Muon, adamw: AdamW) -> None:
        self.muon, self.adamw = muon, adamw

    @property
    def last_lr(self) -> float:
        return self.adamw.last_lr

    def step(self) -> Array:
        """Update both groups; returns the grad norm over all parameters, before clipping."""
        muon_norm, adamw_norm = self.muon.step(), self.adamw.step()

        return xp.sqrt(muon_norm * muon_norm + adamw_norm * adamw_norm)

    def zero_grad(self) -> None:
        self.muon.zero_grad()
        self.adamw.zero_grad()

    def state(self) -> dict[str, int | np.ndarray]:
        """Both optimizers' state, keyed ``muon_*`` and ``adamw_*``."""
        return {f"muon_{k}": v for k, v in self.muon.state().items()} | {
            f"adamw_{k}": v for k, v in self.adamw.state().items()
        }

    def load_state(self, state) -> None:
        for prefix, opt in (("muon_", self.muon), ("adamw_", self.adamw)):
            opt.load_state(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            )


def make_schedule(
    kind: str, lr: float, warmup: int, steps: int, min_lr: float, decay_frac: float
) -> Callable[[int], float]:
    """Cosine decay (Llama 4) or warmup-stable-decay with the last ``decay_frac`` of steps decaying."""
    if kind == "cosine":
        return CosineSchedule(lr, warmup, steps, min_lr)

    return WSDSchedule(lr, warmup, steps, max(1, int(decay_frac * steps)), min_lr)


def make_optimizer(
    kind: str,
    model: Llama4Model,
    schedule: Callable[[int], float],
    weight_decay: float,
    clip_norm: float,
) -> AdamW | MuonAdamW:
    """AdamW on every parameter, or Muon on hidden matrices + AdamW on the rest."""
    # betas and eps of Llama-style pretraining, not the 2017 Transformer's (0.9, 0.98, 1e-9)
    adamw_args = {"beta2": 0.95, "eps": 1e-8, "clip_norm": clip_norm or None}
    if kind == "adamw":
        return AdamW(
            model.parameters(), schedule, weight_decay=weight_decay, **adamw_args
        )

    muon_params, adamw_params = split_muon_params(model, model.non_muon_matrices())
    return MuonAdamW(
        Muon(muon_params, schedule, weight_decay=weight_decay),
        AdamW(adamw_params, schedule, weight_decay=weight_decay, **adamw_args),
    )
