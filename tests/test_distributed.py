"""Multi-GPU (DDP) logic, tested with 2 CPU processes (gloo backend) via torchrun."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tsfusion import InMemoryLoader, RawData, load_dataset, save_dataset
from tsfusion.data import ProcessedData

REPO = Path(__file__).resolve().parents[1]


def test_loader_shards_reproduce_single_gpu_batches():
    n, bs, world = 103, 8, 4
    data = ProcessedData(torch.arange(n, dtype=torch.float32).view(n, 1, 1),
                         torch.zeros(n, 1, 1, dtype=torch.uint8), None, None, None)
    single = list(InMemoryLoader(data, bs * world, shuffle=True, seed=3))
    shards = [list(InMemoryLoader(data, bs, shuffle=True, seed=3, rank=r, world_size=world))
              for r in range(world)]
    assert len({len(s) for s in shards}) == 1                 # same #batches on every rank
    seen = []
    for k in range(len(shards[0])):
        union = sorted(int(v) for s in shards for v in s[k]["x"].view(-1))
        seen += union
        assert set(union) <= set(int(v) for v in single[k]["x"].view(-1))
    assert len(seen) == len(set(seen)) == n // world * world   # disjoint, <= world-1 dropped


def test_save_load_dataset(tmp_path):
    rng = np.random.default_rng(0)
    raw = RawData(rng.normal(size=(5, 11, 3)), np.zeros((5, 11, 3)), np.zeros((5, 11, 3), int),
                  rng.random((5, 4)), rng.gamma(2, 1, 5))
    save_dataset(str(tmp_path), {"train": raw, "val": raw.subset(np.arange(2))}, 2, [4])
    splits, meta = load_dataset(str(tmp_path))
    assert meta["n_special_codes"] == 2 and meta["snapshot_group_sizes"] == [4]
    np.testing.assert_array_equal(splits["val"].x, raw.x[:2])
    np.testing.assert_array_equal(splits["train"].y, raw.y)


def _run(args, nproc=None, timeout=600):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable] + (["-m", "torch.distributed.run", "--standalone",
                               f"--nproc_per_node={nproc}"] if nproc else []) + \
          ["scripts/train.py", "--data", "synthetic", "--n-records", "600", "--device", "cpu",
           "--d-model", "16", "--n-heads", "2"] + args
    res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, res.stderr[-3000:]
    return res


needs_dist = pytest.mark.skipif(not torch.distributed.is_available(), reason="no torch.distributed")


@needs_dist
def test_ddp_matches_single_process(tmp_path):
    common = ["--epochs", "3", "--dropout", "0", "--scheduler", "cosine", "--seed", "1"]
    _run(common + ["--batch-size", "64", "--output-dir", str(tmp_path / "one")])
    _run(common + ["--batch-size", "32", "--output-dir", str(tmp_path / "two")], nproc=2)
    a = pd.read_csv(tmp_path / "one" / "metrics.csv")
    b = pd.read_csv(tmp_path / "two" / "metrics.csv")
    np.testing.assert_allclose(a["val_rmse"], b["val_rmse"], rtol=1e-4)
    np.testing.assert_allclose(a["train_loss"], b["train_loss"], rtol=1e-4)
    # rank 0 alone writes: one header + one line per epoch, one log line per epoch
    log = (tmp_path / "two" / "train.log").read_text().splitlines()
    assert sum(" | epoch " in line for line in log) == 3
    assert json.loads((tmp_path / "two" / "summary.json").read_text())["n_gpus"] == 2
    assert np.load(tmp_path / "two" / "val_predictions.npy").shape == (120,)


@needs_dist
def test_ddp_early_stopping_and_resume(tmp_path):
    out = str(tmp_path / "es")
    # min-delta so large that only epoch 1 counts as an improvement -> stop after 1 + 2 epochs
    _run(["--epochs", "10", "--patience", "2", "--min-delta", "1e12", "--output-dir", out], nproc=2)
    assert pd.read_csv(Path(out) / "metrics.csv")["epoch"].tolist() == [1, 2, 3]
    _run(["--epochs", "4", "--patience", "50", "--output-dir", out,
          "--resume", str(Path(out) / "epoch_002.pt")], nproc=2)
    assert pd.read_csv(Path(out) / "metrics.csv")["epoch"].tolist() == [1, 2, 3, 4]
    assert sorted(f for f in os.listdir(out) if f.startswith("epoch_")) == \
        ["epoch_001.pt", "epoch_002.pt", "epoch_003.pt", "epoch_004.pt"]
