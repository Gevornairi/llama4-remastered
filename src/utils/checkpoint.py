"""Rebuild the model from a checkpoint (saving and loading itself is ``nnkit.training``)."""

import json
from pathlib import Path

from nnkit.training import load_checkpoint, read_checkpoint_metadata

from ..config import ModelConfig
from ..model import Llama4Model


def config_from_checkpoint(path: Path) -> ModelConfig:
    """The ModelConfig the checkpoint was trained with."""
    return ModelConfig(**json.loads(read_checkpoint_metadata(path)["config"]))


def load_model(path: Path) -> Llama4Model:
    """Build the model described by the checkpoint and load its weights, in eval mode.

    The architecture comes from the checkpoint, not the caller, so it always
    matches the run that produced it.
    """
    model = Llama4Model(config_from_checkpoint(path))
    load_checkpoint(path, model)
    model.eval()
    return model
