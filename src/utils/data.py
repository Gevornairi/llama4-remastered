"""Reading the packed token streams written by src.prepare_data."""

import json
from pathlib import Path

import numpy as np


def load_split(data_dir: str | Path, split: str) -> tuple[np.memmap, dict]:
    """One split's flat token stream (memory-mapped, not read into RAM) and meta.json."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    stream = np.memmap(data_dir / f"{split}.bin", dtype=meta["dtype"], mode="r")
    return stream, meta
