"""Train a single Dask XGBoost model and save its booster + metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import xgboost as xgb
from distributed import Client
from xgboost import dask as dxgb

from .losses import LossSpec

DEFAULT_EVAL_METRICS = ("rmse", "mae", "mphe")


@dataclass
class TrainResult:
    loss: str
    params: dict[str, Any]
    loss_metric: str
    best_iteration: int
    num_boosted_rounds: int
    train_metrics: dict[str, float]  # all eval metrics at best_iteration, train set
    valid_metrics: dict[str, float]  # all eval metrics at best_iteration, validation set
    overfit_index: float  # |valid - train| / |train| on loss_metric
    avg_target_train: float
    avg_pred_train: float
    avg_target_valid: float
    avg_pred_valid: float
    train_seconds: float
    booster_path: str | None = None
    history: dict[str, dict[str, list[float]]] = field(default_factory=dict, repr=False)

    @property
    def loss_value(self) -> float:
        return self.valid_metrics[self.loss_metric]

    def summary(self) -> dict[str, Any]:
        """Flat, JSON-friendly dict without the per-round history."""
        d = asdict(self)
        d.pop("history")
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=float))

    @classmethod
    def load(cls, path: str | Path) -> "TrainResult":
        return cls(**json.loads(Path(path).read_text()))


def overfit_index(tr_metric: float, ho_metric: float, eps: float = 1e-12) -> float:
    """abs(ho - tr) / tr. ``abs(tr)`` is used in the denominator because some metrics
    (e.g. XGBoost's tweedie-nloglik) can be negative."""
    return abs(ho_metric - tr_metric) / max(abs(tr_metric), eps)


def _mean(collection) -> float:
    return float(collection.mean().compute())


def train_xgb_model(
    client: Client,
    dtrain: dxgb.DaskQuantileDMatrix,
    dvalid: dxgb.DaskQuantileDMatrix,
    params: dict[str, Any],
    loss: LossSpec,
    X_train,
    y_train,
    X_valid,
    y_valid,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 50,
    eval_metrics: Sequence[str] = DEFAULT_EVAL_METRICS,
    use_gpu: bool = True,
    save_dir: str | Path | None = None,
    verbose_eval: int | bool = False,
) -> tuple[xgb.Booster, TrainResult]:
    """Train one model with early stopping on the validation set.

    ``params`` holds the tree / regularisation params (e.g. from Optuna); objective,
    eval metrics and device are filled in from ``loss``. Early stopping monitors the loss
    metric on ``valid`` and the returned booster is truncated to the best iteration.
    If ``save_dir`` is given, ``booster.ubj`` and ``metrics.json`` are written there.
    """
    full_params = {
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        **loss.fixed_params,
        **params,
        "objective": loss.objective,
    }
    full_params["eval_metric"] = loss.eval_metrics(full_params, eval_metrics)
    loss_metric = loss.loss_metric_name(full_params)

    early_stop = xgb.callback.EarlyStopping(
        rounds=early_stopping_rounds,
        metric_name=loss_metric,
        data_name="valid",
        maximize=False,
        save_best=True,
    )
    t0 = time.perf_counter()
    output = dxgb.train(
        client,
        full_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        callbacks=[early_stop],
        verbose_eval=verbose_eval,
    )
    train_seconds = time.perf_counter() - t0
    booster: xgb.Booster = output["booster"]
    history: dict[str, dict[str, list[float]]] = output["history"]

    n_rounds = len(history["valid"][loss_metric])
    best_it = int(booster.attributes().get("best_iteration", n_rounds - 1))
    train_metrics = {m: float(v[best_it]) for m, v in history["train"].items()}
    valid_metrics = {m: float(v[best_it]) for m, v in history["valid"].items()}

    # Booster is already truncated to the best iteration (save_best=True).
    pred_train = dxgb.inplace_predict(client, booster, X_train)
    pred_valid = dxgb.inplace_predict(client, booster, X_valid)

    result = TrainResult(
        loss=loss.name,
        params={k: v for k, v in full_params.items()},
        loss_metric=loss_metric,
        best_iteration=best_it,
        num_boosted_rounds=booster.num_boosted_rounds(),
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        overfit_index=overfit_index(train_metrics[loss_metric], valid_metrics[loss_metric]),
        avg_target_train=_mean(y_train),
        avg_pred_train=_mean(pred_train),
        avg_target_valid=_mean(y_valid),
        avg_pred_valid=_mean(pred_valid),
        train_seconds=train_seconds,
        history={ds: {m: [float(x) for x in v] for m, v in ms.items()} for ds, ms in history.items()},
    )
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        booster_path = save_dir / "booster.ubj"
        booster.save_model(booster_path)
        result.booster_path = str(booster_path)
        result.save(save_dir / "metrics.json")
    return booster, result
