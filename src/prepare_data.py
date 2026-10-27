"""Download a text dataset, train a byte-level BPE and write packed token streams.

  python -m src.prepare_data                                   # TinyStories, 8k vocab
  python -m src.prepare_data --max-docs 20000 --out data/tiny  # quick subset

Writes to --out:
  tokenizer.json       byte-level BPE (PAD=0, EOS=1); any text encodes without unknowns
  {train,valid}.bin    flat uint16 token ids: every document, each followed by EOS
  meta.json            vocab size, special ids, document and token counts, settings

Documents are streamed from the Hugging Face hub, so nothing larger than one chunk is
held in memory. Training windows are cut from the flat stream later; a window may span
several documents, which the document mask keeps apart.
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<pad>", "<eos>"]  # ids 0, 1
PAD_ID, EOS_ID = 0, 1


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE: every byte is in the base alphabet, so no unknown tokens."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    for i, s in enumerate(SPECIALS):
        assert tokenizer.token_to_id(s) == i, f"{s} should have id {i}"
    return tokenizer


def documents(dataset: str, split: str, field: str):
    """Non-empty, stripped texts of one split, streamed in order."""
    for row in load_dataset(dataset, split=split, streaming=True):
        text = row[field].strip()
        if text:
            yield text


def write_stream(tokenizer: Tokenizer, docs, path: Path, chunk: int = 20_000) -> dict:
    """Encode docs in chunks and append them to path as uint16, each followed by EOS."""
    n_docs = n_tokens = 0
    with open(path, "wb") as f:
        while batch := list(itertools.islice(docs, chunk)):
            # encode_batch fans out over all CPU cores in Rust
            encoded = tokenizer.encode_batch(batch, add_special_tokens=False)
            ids = np.fromiter(
                (t for e in encoded for t in (*e.ids, EOS_ID)),
                dtype=np.uint16,
                count=sum(len(e.ids) + 1 for e in encoded),
            )
            f.write(ids.tobytes())
            n_docs += len(batch)
            n_tokens += len(ids)
            print(f"  {path.name}: {n_docs:,} docs, {n_tokens:,} tokens", end="\r")
    print()
    return {"docs": n_docs, "tokens": n_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--field", default="text", help="column holding the text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--valid-split",
        default="validation",
        help="'' to hold out the first --valid-docs train documents instead",
    )
    parser.add_argument("--valid-docs", type=int, default=10_000)
    parser.add_argument("--out", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--tokenizer-docs", type=int, default=200_000, help="train docs used for BPE"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="cap train docs (for a quick run)"
    )
    args = parser.parse_args()
    if args.vocab_size > np.iinfo(np.uint16).max + 1:
        parser.error("--vocab-size must fit in uint16 (<= 65536)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_docs = documents(args.dataset, args.train_split, args.field)
    if args.valid_split:
        valid = list(documents(args.dataset, args.valid_split, args.field))
    else:  # no validation split: hold out the first train documents
        valid = list(itertools.islice(train_docs, args.valid_docs))
    if args.max_docs is not None:
        train_docs = itertools.islice(train_docs, args.max_docs)

    # the BPE learns from the first train documents; they are written to train.bin too
    head = list(itertools.islice(train_docs, args.tokenizer_docs))
    print(
        f"loaded {len(valid):,} valid and {len(head):,} BPE docs in {time.time() - t0:.0f}s"
    )
    tokenizer = build_tokenizer(head, args.vocab_size)
    tokenizer.save(str(out / "tokenizer.json"))
    print(
        f"trained BPE ({tokenizer.get_vocab_size():,} types) at {time.time() - t0:.0f}s"
    )

    counts = {
        "valid": write_stream(tokenizer, iter(valid), out / "valid.bin"),
        "train": write_stream(
            tokenizer, itertools.chain(head, train_docs), out / "train.bin"
        ),
    }
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "dtype": "uint16",
        "counts": counts,
        "args": vars(args),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{counts}\ndone in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
    # A stream stopped early (--max-docs) can leave a pyarrow read in flight. At exit,
    # Arrow's thread pool waits for that read, but it can never take the GIL back, so
    # the process hangs. Every file is written and closed by now, so skip the teardown.
    sys.stdout.flush()
    os._exit(0)
