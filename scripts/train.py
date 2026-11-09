"""Train tsfusion from the command line - single GPU or all GPUs.

    # one GPU
    python scripts/train.py --data data/prepared --output-dir runs/exp1

    # all 8 GPUs of one machine (DistributedDataParallel, one process per GPU)
    torchrun --standalone --nproc_per_node=8 scripts/train.py --data data/prepared --output-dir runs/exp1_ddp

    # 8 independent experiments at once (one per GPU): see scripts/run_grid.sh

``--data`` is a directory written by ``tsfusion.save_dataset`` (``train_*.npy``,
``val_*.npy`` and ``meta.json``) or ``synthetic`` for the demo data.
``--batch-size`` is PER GPU: with DDP the global batch is batch_size x number of GPUs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfusion import (ModelConfig, Preprocessor, RawData, Trainer, TrainConfig,  # noqa: E402
                      cleanup_distributed, default_target_mode, group_columns_by_prefix,
                      is_main_process, load_dataset, long_df_to_array, regression_metrics)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="synthetic", help="dataset dir from save_dataset, or 'synthetic'")
    p.add_argument("--n-records", type=int, default=6000, help="synthetic data only")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None, help="epoch_XXX.pt checkpoint to continue from")
    # objective
    p.add_argument("--loss", choices=["mse", "tweedie"], default="tweedie")
    p.add_argument("--tweedie-power", type=float, default=1.5)
    # model
    p.add_argument("--fusion", choices=["late", "early", "hybrid"], default="late")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--patch-len", type=int, default=3)
    p.add_argument("--patch-stride", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256, help="per GPU")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=10, help="early stopping, in epochs")
    p.add_argument("--min-delta", type=float, default=0.0, help="early stopping: min RMSE improvement")
    p.add_argument("--train-eval-max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", help="'cuda' (GPU; one per process with torchrun) or 'cpu'")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def load_data(args):
    """Return (train RawData, validation RawData, n_special_codes, snapshot_group_sizes)."""
    if args.data != "synthetic":
        splits, meta = load_dataset(args.data, mmap=True)   # read chunk by chunk from disk
        return splits["train"], splits["val"], meta["n_special_codes"], meta["snapshot_group_sizes"]

    from tsfusion.synthetic import make_synthetic_dataset
    d = make_synthetic_dataset(n_records=args.n_records, n_vars=8, n_special_codes=3, seed=0)
    groups = group_columns_by_prefix(d["snapshot_ple_df"].columns)
    cols = [c for _, cs in groups for c in cs]

    def to_raw(ids):
        kw = dict(value_cols=d["var_names"], record_ids=ids, time_steps=d["time_steps"])
        x, _ = long_df_to_array(d["ts_df"], **kw)
        miss, _ = long_df_to_array(d["missing_df"], fill_value=1, **kw)
        spec, _ = long_df_to_array(d["special_df"], fill_value=0, **kw)
        return RawData(x, miss, spec.astype(np.int64), d["snapshot_ple_df"].loc[ids, cols].to_numpy(),
                       d["y"].loc[ids].to_numpy())
    return to_raw(d["train_ids"]), to_raw(d["val_ids"]), d["n_special_codes"], [len(c) for _, c in groups]


def main(argv=None) -> None:
    args = parse_args(argv)
    train_raw, val_raw, n_special, group_sizes = load_data(args)

    # every process fits the same (deterministic) preprocessor on the same training data
    prep = Preprocessor(n_special, group_sizes, target_mode=default_target_mode(args.loss)).fit(train_raw)
    train, val = prep.transform(train_raw), prep.transform(val_raw)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = math.ceil((len(train) // world) / args.batch_size)   # per GPU
    model_cfg = ModelConfig(
        n_vars=train_raw.x.shape[2], n_steps=train_raw.x.shape[1], n_special_codes=n_special,
        snapshot_group_sizes=group_sizes, patch_len=args.patch_len, patch_stride=args.patch_stride,
        d_model=args.d_model, n_heads=args.n_heads, n_ts_layers=args.n_layers,
        dropout=args.dropout, head_dropout=args.dropout, missing_fusion=args.fusion)
    train_cfg = TrainConfig(
        loss=args.loss, tweedie_power=args.tweedie_power, batch_size=args.batch_size,
        max_epochs=args.epochs, lr=args.lr, min_lr=args.min_lr, scheduler=args.scheduler,
        warmup_steps=int(round(args.warmup_epochs * steps_per_epoch)),
        early_stopping_patience=args.patience, early_stopping_min_delta=args.min_delta,
        train_eval_max_samples=args.train_eval_max_samples,
        seed=args.seed, device=args.device, use_amp=not args.no_amp, output_dir=args.output_dir)

    trainer = Trainer(model_cfg, train_cfg, prep)
    start = trainer.resume(args.resume) if args.resume else 1
    history = trainer.fit(train, val, start_epoch=start)
    pred = trainer.predict(val)                       # collective: every rank calls it

    if is_main_process():
        np.save(os.path.join(args.output_dir, "val_predictions.npy"), pred)
        best = history.loc[history["val_rmse"].idxmin()]
        summary = {"best_epoch": int(best["epoch"]), "epochs_run": int(history["epoch"].max()),
                   "n_gpus": world, **{k: float(v) for k, v in regression_metrics(val_raw.y, pred).items()}}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary))
    cleanup_distributed()


if __name__ == "__main__":
    main()
