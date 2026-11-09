"""Pick tuned hyper-parameters from a study and save the final model bundle."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import optuna

from .storage import copy_study_to_local, default_local_dir, is_fuse_path, normalize_path
from .trainer import TrainResult
from .tuner import select_best_trial

# Keys of a saved trial's params that train_xgb_model sets itself (from the loss / device).
_DERIVED_KEYS = ("objective", "eval_metric", "device", "tree_method")


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop objective / eval_metric / device / tree_method; keep everything that was tuned
    or fixed (incl. max_bin, subsample, huber_slope, tweedie_variance_power)."""
    return {k: v for k, v in params.items() if k not in _DERIVED_KEYS}


def load_tuned_params(
    study_dir: str | Path,
    trial: int | None = None,
    max_overfit: float | None = None,
    local_db_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hyper-parameters of a tuning trial, read from ``<study_dir>/trials/trial_XXXXX/metrics.json``.

    ``trial=None`` picks the best trial of the study (lowest loss on the Pareto front with
    overfit index <= ``max_overfit``, same rule as the tuner). ``study_dir`` is
    ``<output_dir>/<study_name>``; on a Databricks volume the study DB is copied to local
    disk first. Returns a dict with ``trial``, ``loss``, ``params`` (ready for
    ``train_xgb_model``), ``best_iteration``, ``train_metrics``, ``valid_metrics``,
    ``overfit_index``.
    """
    study_dir = normalize_path(study_dir)
    if trial is None:
        db = study_dir / "study.db"
        if not db.exists():
            raise FileNotFoundError(f"No study at {db}")
        if is_fuse_path(db):
            local = Path(local_db_dir or default_local_dir()) / f"{study_dir.name}_readonly.db"
            storage = copy_study_to_local(db, local)
        else:
            storage = f"sqlite:///{db.resolve()}"
        study = optuna.load_study(study_name=study_dir.name, storage=storage)
        trial = select_best_trial(study, max_overfit).number

    res = TrainResult.load(study_dir / "trials" / f"trial_{trial:05d}" / "metrics.json")
    return {
        "trial": trial,
        "loss": res.loss,
        "params": clean_params(res.params),
        "best_iteration": res.best_iteration,
        "train_metrics": res.train_metrics,
        "valid_metrics": res.valid_metrics,
        "overfit_index": res.overfit_index,
    }


def save_model_bundle(
    booster,
    result: TrainResult,
    model_dir: str | Path,
    model_card: dict[str, Any],
) -> Path:
    """Write ``booster.ubj``, ``metrics.json`` and ``model_card.json`` to ``model_dir``.

    Files are written to a local temp dir first and then copied whole, which also works
    on Databricks volumes / DBFS.
    """
    model_dir = normalize_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        booster.save_model(tmp / "booster.ubj")
        result.booster_path = str(model_dir / "booster.ubj")
        result.save(tmp / "metrics.json")
        (tmp / "model_card.json").write_text(json.dumps(model_card, indent=2, default=str))
        for f in tmp.iterdir():
            shutil.copy(f, model_dir / f.name)
    return model_dir
