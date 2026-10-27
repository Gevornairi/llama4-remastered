"""Train the Llama 4-style model on the packed token streams built by prepare_data.

  python -m src.train --preset llama4                 # faithful baseline: AdamW + cosine
  python -m src.train --preset remastered             # every upgrade: Muon + WSD
  python -m src.train --preset llama4 --set tied_embeddings=true --run llama4-tied
  python -m src.train --run llama4 --resume           # continue runs/llama4/last.npz

``--set field=value`` replaces one ModelConfig field (an ablation); ``--optimizer``
and ``--schedule`` override the preset's training recipe. Every --eval-every
updates: validation loss, a sampled story and checkpoints (last.npz always,
best.npz when validation loss improves). Every log line also goes to
runs/<run>/log.jsonl, for comparing runs later.
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
from nnkit.base import asarray, materialize, seed, to_numpy
from nnkit.losses import LabelSmoothingLoss, z_loss
from nnkit.training import (
    Prefetcher,
    load_checkpoint,
    packed_batches,
    read_checkpoint_metadata,
    save_checkpoint,
)
from tokenizers import Tokenizer

from .config import PRESETS, ModelConfig
from .model import Llama4Model
from .utils.checkpoint import config_from_checkpoint
from .utils.data import load_split
from .utils.generation import generate
from .utils.optim import make_optimizer, make_schedule

# the training recipe that goes with each preset (README plan)
RECIPES = {"llama4": ("adamw", "cosine"), "remastered": ("muon", "wsd")}
PROMPT = "Once upon a time"


def parse_overrides(pairs: list[str]) -> dict:
    """``["top_k=2", "tied_embeddings=true"]`` -> typed ModelConfig field values."""
    types = {f.name: type(f.default) for f in fields(ModelConfig)}
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in types:
            raise SystemExit(f"--set {pair}: {key!r} is not a ModelConfig field")
        if types[key] is bool:
            if value.lower() not in ("true", "false"):
                raise SystemExit(f"--set {pair}: expected true or false")
            out[key] = value.lower() == "true"
        else:
            out[key] = types[key](value)
    return out


def validation_loss(model, loss_fn, valid, cfg, batch_size, eos, n_batches) -> float:
    """Mean loss over the first n_batches validation batches (always the same windows)."""
    model.eval()
    losses = []
    for ids, targets, doc_ids in itertools.islice(
        packed_batches(valid, cfg.seq_len, batch_size, eos), n_batches
    ):
        logits = model.forward(asarray(ids), asarray(doc_ids))
        losses.append(loss_fn(logits, asarray(targets))[0].item())
    model.train()

    return float(np.mean(losses))


def expert_loads(model: Llama4Model) -> np.ndarray:
    """(n_layers, n_experts): tokens routed to each expert since the last update_bias()."""
    return np.stack([to_numpy(block.moe.load) for block in model.blocks])


def expert_imbalance(loads: np.ndarray) -> float:
    """Busiest expert's tokens over the average, averaged over layers: 1 = even, n_experts = collapsed."""
    return float(np.mean(loads.max(axis=1) / np.maximum(loads.mean(axis=1), 1e-9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tinystories")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="llama4")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="replace one ModelConfig field; repeatable",
    )
    parser.add_argument("--run", default=None, help="run name; default = --preset")
    parser.add_argument("--resume", action="store_true")
    # optimisation (defaults: ~164M tokens, about a third of TinyStories)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default=None)
    parser.add_argument("--schedule", choices=["cosine", "wsd"], default=None)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=1, help="micro-batches per update")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--decay-frac", type=float, default=0.2, help="WSD: share of steps decaying"
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0, help="0 disables")
    parser.add_argument("--z-loss", type=float, default=0.0, help="z-loss coefficient")
    parser.add_argument("--seed", type=int, default=0)
    # monitoring
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=50)
    args = parser.parse_args()
    if args.accum < 1:
        parser.error("--accum must be at least 1")
    default_optimizer, default_schedule = RECIPES[args.preset]
    args.optimizer = args.optimizer or default_optimizer
    args.schedule = args.schedule or default_schedule
    args.run = args.run or args.preset

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "last.npz"
    resuming = args.resume and last.exists()

    data = Path(args.data)
    train, meta = load_split(data, "train")
    valid, _ = load_split(data, "valid")
    eos = meta["eos_id"]
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))

    if resuming:  # the architecture comes from the checkpoint, not the flags
        cfg = config_from_checkpoint(last)
    else:
        cfg = replace(
            PRESETS[args.preset],
            vocab_size=meta["vocab_size"],
            **parse_overrides(args.overrides),
        )
    seed(args.seed)
    model = Llama4Model(cfg)
    loss_fn = LabelSmoothingLoss(cfg.vocab_size, meta["pad_id"], smoothing=0.0)
    schedule = make_schedule(
        args.schedule, args.lr, args.warmup, args.steps, args.min_lr, args.decay_frac
    )
    opt = make_optimizer(
        args.optimizer, model, schedule, args.weight_decay, args.clip_norm
    )

    step, best_loss = 0, float("inf")
    if resuming:
        saved = json.loads(read_checkpoint_metadata(last)["args"])
        if saved["optimizer"] != args.optimizer:  # their states don't fit each other
            raise SystemExit(
                f"{last} was trained with --optimizer {saved['optimizer']}; "
                f"resume with the same one (got {args.optimizer})"
            )
        resumed = load_checkpoint(last, model, opt)
        step, best_loss = int(resumed["step"]), float(resumed["best_loss"])
        print(f"resumed {args.run} at step {step} (best valid loss {best_loss:.3f})")

    tokens_per_step = args.batch_size * cfg.seq_len * args.accum
    n_params = sum(p.value.size for p in model.parameters())
    print(
        f"{args.run}: {args.preset} preset, {args.optimizer} + {args.schedule} | "
        f"{n_params / 1e6:.2f}M params | {len(train) / 1e6:.0f}M train tokens | "
        f"{tokens_per_step:,} tokens/step, {args.steps:,} steps"
    )
    if args.overrides and not resuming:
        print(f"overrides: {parse_overrides(args.overrides)}")

    batches = Prefetcher(
        lambda rng: packed_batches(train, cfg.seq_len, args.batch_size, eos, rng),
        seed=args.seed + step,  # resuming continues with a fresh, reproducible order
    )
    log_file = (run_dir / "log.jsonl").open("a")
    prompt = tokenizer.encode(PROMPT).ids

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    # running totals since the last log line
    log_t0, log_loss, log_z, log_n = time.time(), 0.0, 0.0, 0
    log_load = np.zeros((cfg.n_layers, cfg.n_experts))
    epoch = 0  # set by every batch; the accumulation loop always runs at least once
    model.train()
    while step < args.steps:
        model.zero_grad()
        for _ in range(args.accum):
            epoch, (ids, targets, doc_ids) = batches.next()
            targets = asarray(targets)
            logits = model.forward(asarray(ids), asarray(doc_ids))
            loss, d_logits = loss_fn(logits, targets)
            if args.z_loss:
                z, d_z = z_loss(logits, targets, meta["pad_id"], args.z_loss)
                d_logits = d_logits + d_z
                log_z += z.item()
            model.backward(d_logits * (1.0 / args.accum))
            materialize(loss, [p.grad for p in model.parameters()])
            log_loss += loss.item()
            log_n += 1
        grad_norm = opt.step()
        step += 1
        loads = expert_loads(model)  # read before update_bias() resets the counts
        log_load += loads

        if step % args.log_every == 0:
            dt = time.time() - log_t0
            record = {
                "step": step,
                "epoch": epoch,
                "loss": log_loss / log_n,
                "lr": opt.last_lr,
                "grad_norm": grad_norm.item(),
                "imbalance": expert_imbalance(loads),  # this step's routing
                # each layer's share of tokens per expert, over the whole interval
                "expert_share": (log_load / log_load.sum(axis=1, keepdims=True))
                .round(4)
                .tolist(),
                "tokens_per_s": tokens_per_step * args.log_every / dt,
            }
            if args.z_loss:
                record["z_loss"] = log_z / log_n
            log(record)
            print(
                f"step {step:6d} | ep {epoch} | loss {record['loss']:.3f} | "
                f"lr {record['lr']:.2e} | gnorm {record['grad_norm']:.2f} | "
                f"experts {record['imbalance']:.2f}x | "
                f"{record['tokens_per_s']:,.0f} tok/s",
                flush=True,
            )
            log_t0, log_loss, log_z, log_n = time.time(), 0.0, 0.0, 0
            log_load[:] = 0
        model.update_bias()  # resets the expert load counts

        if step % args.eval_every == 0 or step == args.steps:
            t0 = time.time()
            v_loss = validation_loss(
                model, loss_fn, valid, cfg, args.batch_size, eos, args.eval_batches
            )
            is_best = v_loss < best_loss
            best_loss = min(best_loss, v_loss)
            progress = {
                "step": step,
                "best_loss": best_loss,
                "config": json.dumps(asdict(cfg)),
                "args": json.dumps(vars(args)),
            }
            save_checkpoint(last, model, opt, **progress)
            if is_best:
                save_checkpoint(run_dir / "best.npz", model, opt, **progress)
            # same seed every time, so samples are comparable across evals
            story = tokenizer.decode(
                prompt + generate(model, prompt, 150, eos, rng=np.random.default_rng(0))
            )
            log({"step": step, "valid_loss": v_loss, "sample": story})
            print(
                f"== eval step {step}: valid loss {v_loss:.3f} "
                f"(ppl {np.exp(v_loss):.1f}){' (best)' if is_best else ''} | "
                f"{time.time() - t0:.0f}s\n   {story}",
                flush=True,
            )
            log_t0 = time.time()

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
