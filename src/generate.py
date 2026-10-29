"""Sample stories from a trained checkpoint.

  python -m src.generate --run full-remastered
  python -m src.generate --run full-llama4 --prompt "Tom found a key" --samples 3
  python -m src.generate --checkpoint runs/full-llama4/last.npz --temperature 0

The architecture comes from the checkpoint, so one command serves every preset
and ablation. The same --seed gives the same stories, for comparing runs.
"""

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .utils.checkpoint import load_model
from .utils.data import load_split
from .utils.generation import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tinystories")
    parser.add_argument("--run", default="remastered", help="reads runs/<run>/best.npz")
    parser.add_argument("--checkpoint", default=None, help="a .npz path; overrides --run")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.checkpoint or Path("runs") / args.run / "best.npz")
    if not path.exists():
        parser.error(f"no checkpoint at {path}")
    data = Path(args.data)
    _, meta = load_split(data, "valid")
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    model = load_model(path)
    prompt = tokenizer.encode(args.prompt).ids
    rng = np.random.default_rng(args.seed)
    for i in range(args.samples):
        new = generate(model, prompt, args.max_new, meta["eos_id"], args.temperature, rng)
        print(f"--- {path} | sample {i + 1}\n{tokenizer.decode(prompt + new)}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
