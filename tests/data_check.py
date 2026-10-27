"""Check the prepared data: the .bin + meta.json reader and the files themselves.

python -m tests.data_check

Exits non-zero if any check fails. The last check reads data/tinystories if it exists.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.utils.data import load_split

EOS = 1


def check_load_split() -> None:
    """A .bin + meta.json round trip through the memmap."""
    stream = np.array([5, 9, EOS, 7, 8, EOS, 4, EOS], dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "meta.json").write_text(
            json.dumps({"dtype": "uint16", "eos_id": EOS})
        )
        stream.tofile(Path(tmp) / "train.bin")
        loaded, meta = load_split(tmp, "train")
        np.testing.assert_array_equal(loaded, stream)
        assert loaded.dtype == np.uint16 and meta["eos_id"] == EOS
        del loaded  # release the memmap before the directory is removed


def check_prepared_data(data_dir: Path = Path("data/tinystories")) -> None:
    """The real files agree with meta.json: one EOS per document, ids inside the vocab."""
    if not (data_dir / "meta.json").exists():
        print(f"  (skipped: no {data_dir})")
        return
    for split in ("train", "valid"):
        stream, meta = load_split(data_dir, split)
        counts = meta["counts"][split]
        assert len(stream) == counts["tokens"], (split, len(stream))
        assert int((stream == meta["eos_id"]).sum()) == counts["docs"], split
        assert int(stream.max()) < meta["vocab_size"], split
        assert stream[-1] == meta["eos_id"], split


def main():
    ok = True
    for check in (check_load_split, check_prepared_data):
        try:
            check()
            print(f"{check.__name__}: ok")
        except Exception as e:  # noqa: BLE001 - report every failing check
            ok = False
            print(f"{check.__name__}: FAIL {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
