"""Optuna multi-objective (loss, overfit index) tuning of Dask GPU XGBoost, resumable."""

from __future__ import annotations

import functools
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import optuna
import pandas as pd
from distributed import Client
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState

from .data import build_dmatrices
from .losses import LossSpec, SpaceEntry, get_loss
from .storage import (
    copy_study_to_local,
    default_local_dir,
    is_fuse_path,
    normalize_path,
    save_study_to_volume,
)
from .trainer import DEFAULT_EVAL_METRICS, TrainResult, overfit_index, train_xgb_model

DEFAULT_SEARCH_SPACE: dict[str, SpaceEntry] = {
    "max_depth": ("int", 3, 12, {}),
    "learning_rate": ("float", 5e-3, 0.3, {"log": True}),
    "gamma": ("float", 1e-8, 10.0, {"log": True}),
    "min_child_weight": ("float", 1e-2, 100.0, {"log": True}),
    "max_delta_step": ("float", 0.0, 10.0, {}),  # 0 = no constraint
    "reg_alpha": ("float", 1e-8, 10.0, {"log": True}),
    "reg_lambda": ("float", 1e-3, 100.0, {"log": True}),
}

DEFAULT_FIXED_PARAMS: dict[str, Any] = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 256,
}

DataLoader = Callable[[], tuple[Any, Any, Any, Any]]


def tpe_gamma(n: int, fraction: float = 0.1, max_n: int = 25) -> int:
    """Number of trials TPE treats as "good": min(ceil(fraction * n), max_n)."""
    return min(int(math.ceil(fraction * n)), max_n)


def select_best_trial(study: optuna.Study, max_overfit: float | None = None) -> FrozenTrial:
    """Pick one trial from the Pareto front.

    Lowest loss among Pareto trials whose overfit index <= ``max_overfit``; if none
    qualifies (or no threshold is given), the Pareto trial with the lowest loss.
    """
    pareto = study.best_trials
    if not pareto:
        raise RuntimeError("No completed trials yet.")
    if max_overfit is not None:
        ok = [t for t in pareto if t.values[1] <= max_overfit]
        if ok:
            return min(ok, key=lambda t: t.values[0])
    return min(pareto, key=lambda t: t.values[0])


class XGBOptunaTuner:
    """Bayesian (TPE) multi-objective tuning for one loss function.

    Objectives (both minimised):
      1. the loss metric on the validation set at the best iteration
         (rmse for MSE, mphe for Huber, tweedie-nloglik@p for Tweedie), and
      2. overfit index = |valid - train| / |train| of that same metric.

    Every trial is committed to a SQLite study (``<output_dir>/<study_name>/study.db``)
    as soon as it finishes, and its booster + metrics are written to
    ``trials/trial_XXXXX/``. Re-running ``optimize`` with the same ``output_dir`` and
    ``study_name`` resumes the study: trials interrupted by a crash (e.g. a GPU error that
    killed the kernel) are marked failed and re-queued with the same hyper-parameters.

    ``data_loader`` is a zero-argument callable returning
    ``(X_train, y_train, X_valid, y_valid)`` Dask collections. It is called lazily, and
    again after a cluster restart (a restart wipes all data held by the workers).
    Assumes a single process drives a given study at a time.

    Storage:
      * ``storage`` given (e.g. ``"postgresql://..."`` or an Optuna storage object): used as is.
      * otherwise SQLite at ``<study_dir>/study.db``. If ``output_dir`` is on a Databricks
        FUSE mount (``/Volumes``, ``/dbfs``, ``/Workspace``; SQLite fails there with
        "disk I/O error"), or ``sync_db=True``, Optuna uses a local copy
        ``<local_db_dir>/<study_name>.db`` (default dir ``/local_disk0/tmp/optuna``):
        ``study.db`` is copied volume -> local when the tuner is created and
        local -> volume after every trial.
    """

    def __init__(
        self,
        client: Client,
        data_loader: DataLoader,
        loss: str | LossSpec = "tweedie",
        output_dir: str | Path = "optuna_runs",
        study_name: str | None = None,
        search_space: dict[str, SpaceEntry] | None = None,
        fixed_params: dict[str, Any] | None = None,
        num_boost_round: int = 5000,
        early_stopping_rounds: int = 50,
        eval_metrics: Sequence[str] = DEFAULT_EVAL_METRICS,
        opt_metric: str | None = None,
        n_startup_trials: int = 20,
        gamma: Callable[[int], int] | None = None,
        gamma_fraction: float = 0.1,
        gamma_max: int = 25,
        multivariate: bool = True,
        seed: int = 42,
        use_gpu: bool = True,
        enable_categorical: bool = False,
        max_overfit: float | None = None,
        verbose: bool = True,
        storage: str | optuna.storages.BaseStorage | None = None,
        sync_db: bool | None = None,
        local_db_dir: str | Path | None = None,
    ):
        self.client = client
        self.data_loader = data_loader
        self.loss = get_loss(loss) if isinstance(loss, str) else loss
        self.study_name = study_name or f"xgb_{self.loss.name}"
        self.study_dir = normalize_path(output_dir) / self.study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)

        self.volume_db = self.study_dir / "study.db"
        self.local_db: Path | None = None  # set when the DB lives on local disk
        if storage is not None:
            self.storage = storage
        elif sync_db or (sync_db is None and is_fuse_path(self.study_dir)):
            self.local_db = Path(local_db_dir or default_local_dir()) / f"{self.study_name}.db"
            self.storage = copy_study_to_local(self.volume_db, self.local_db)
        else:
            self.storage = f"sqlite:///{(self.study_dir / 'study.db').resolve()}"
        # URL (or storage object) to pass to optuna.load_study for this study.
        self.storage_url = self.storage

        space = dict(DEFAULT_SEARCH_SPACE if search_space is None else search_space)
        for name, entry in self.loss.extra_space.items():
            space.setdefault(name, entry)
        self.search_space = space
        self.fixed_params = {**DEFAULT_FIXED_PARAMS, **(fixed_params or {})}

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metrics = tuple(eval_metrics)
        self.opt_metric = opt_metric  # None -> the loss metric
        self.n_startup_trials = n_startup_trials
        self.gamma = gamma or functools.partial(tpe_gamma, fraction=gamma_fraction, max_n=gamma_max)
        self.multivariate = multivariate
        self.seed = seed
        self.use_gpu = use_gpu
        self.enable_categorical = enable_categorical
        self.max_overfit = max_overfit
        self.verbose = verbose

        self._data = None
        self._dmatrices = None
        self.study: optuna.Study | None = None

    # ------------------------------------------------------------------ data
    def _ensure_data(self):
        if self._data is None:
            self._data = self.data_loader()
        if self._dmatrices is None:
            self._dmatrices = build_dmatrices(
                self.client,
                *self._data,
                max_bin=self.fixed_params["max_bin"],
                enable_categorical=self.enable_categorical,
            )
        return self._data, self._dmatrices

    def _reset_data(self):
        self._data = None
        self._dmatrices = None
        gc.collect()

    # ----------------------------------------------------------------- study
    def _make_sampler(self, n_existing: int) -> optuna.samplers.TPESampler:
        # Offset the seed by the number of existing trials so a resumed study does not
        # replay the same random start-up points.
        return optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            gamma=self.gamma,
            multivariate=self.multivariate,
            seed=None if self.seed is None else self.seed + n_existing,
        )

    def load_study(self) -> optuna.Study:
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=["minimize", "minimize"],
            load_if_exists=True,
        )
        prev_loss = study.user_attrs.get("loss")
        if prev_loss is not None and prev_loss != self.loss.name:
            raise ValueError(
                f"Study {self.study_name!r} was created for loss {prev_loss!r}, not "
                f"{self.loss.name!r}. Use a different study_name."
            )
        study.set_user_attr("loss", self.loss.name)
        study.set_user_attr("opt_metric", self.opt_metric or "loss_metric")
        study.set_metric_names([self.opt_metric or "loss_metric", "overfit_index"])
        study.sampler = self._make_sampler(len(study.trials))
        self.study = study
        self.sync_storage()
        return study

    def sync_storage(self) -> None:
        """Copy the local study DB to the volume (no-op unless the DB lives on local disk)."""
        if self.local_db is not None:
            try:
                save_study_to_volume(self.local_db, self.volume_db)
            except OSError as e:  # a failed copy must not kill the tuning run
                self._log(f"[warn] could not copy {self.local_db} to {self.volume_db}: {e}")

    def _recover(self, study: optuna.Study, max_retries_per_trial: int) -> None:
        """Fail trials left RUNNING by a crash and re-queue failed trials' params."""
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study._storage.set_trial_user_attr(t._trial_id, "fail_reason", "interrupted")
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
            self._log(f"[recover] trial {t.number} was interrupted -> marked FAIL")

        trials = study.get_trials(deepcopy=False)
        already_retried = {t.user_attrs["retry_of"] for t in trials if "retry_of" in t.user_attrs}
        for t in trials:
            if (
                t.state != TrialState.FAIL
                or t.number in already_retried
                or not t.params
                or t.user_attrs.get("fail_reason") == "non_finite"
            ):
                continue
            n_retry = t.user_attrs.get("retry_count", 0)
            if n_retry >= max_retries_per_trial:
                continue
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number, "retry_count": n_retry + 1})
            self._log(f"[recover] re-queued params of failed trial {t.number} (retry {n_retry + 1})")

    # ------------------------------------------------------------- objective
    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        params = dict(self.fixed_params)
        for name, (kind, low, high, opts) in self.search_space.items():
            if kind == "int":
                params[name] = trial.suggest_int(name, int(low), int(high), **opts)
            elif kind == "float":
                params[name] = trial.suggest_float(name, low, high, **opts)
            else:
                raise ValueError(f"Unknown search-space kind {kind!r} for {name!r}")
        if "tweedie_variance_power" in params:
            params["tweedie_variance_power"] = round(params["tweedie_variance_power"], 4)
        return params

    def _objective(self, trial: optuna.Trial) -> tuple[float, float]:
        params = self._suggest(trial)
        self.sync_storage()  # persist the RUNNING trial's params so a crash can retry them
        (X_tr, y_tr, X_va, y_va), (dtrain, dvalid) = self._ensure_data()
        trial_dir = self.study_dir / "trials" / f"trial_{trial.number:05d}"
        trial.set_user_attr("artifact_dir", str(trial_dir))
        try:
            booster, res = train_xgb_model(
                self.client, dtrain, dvalid, params, self.loss,
                X_tr, y_tr, X_va, y_va,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metrics=self.eval_metrics,
                use_gpu=self.use_gpu,
                save_dir=trial_dir,
            )
        except Exception as e:
            trial.set_user_attr("fail_reason", f"{type(e).__name__}: {e}"[:500])
            raise
        del booster

        metric = self.opt_metric or res.loss_metric
        tr, ho = res.train_metrics[metric], res.valid_metrics[metric]
        ofi = overfit_index(tr, ho)
        trial.set_user_attr("loss_metric", res.loss_metric)
        trial.set_user_attr("best_iteration", res.best_iteration)
        trial.set_user_attr("avg_target_valid", res.avg_target_valid)
        trial.set_user_attr("avg_pred_valid", res.avg_pred_valid)
        trial.set_user_attr("avg_target_train", res.avg_target_train)
        trial.set_user_attr("avg_pred_train", res.avg_pred_train)
        for m in res.valid_metrics:
            trial.set_user_attr(f"train_{m}", res.train_metrics[m])
            trial.set_user_attr(f"valid_{m}", res.valid_metrics[m])
        if not (math.isfinite(ho) and math.isfinite(ofi)):
            trial.set_user_attr("fail_reason", "non_finite")
            return float("nan"), float("nan")  # Optuna marks the trial FAIL
        return ho, ofi

    def _after_trial(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """Runs after every trial: persist a CSV snapshot and log progress."""
        self.sync_storage()
        study.trials_dataframe().to_csv(self.study_dir / "trials.csv", index=False)
        if trial.state == TrialState.COMPLETE:
            best = select_best_trial(study, self.max_overfit)
            self._log(
                f"[trial {trial.number}] loss={trial.values[0]:.6g} overfit={trial.values[1]:.4f} "
                f"best_iter={trial.user_attrs.get('best_iteration')} | current best: trial "
                f"{best.number} loss={best.values[0]:.6g} overfit={best.values[1]:.4f}"
            )
        else:
            self._log(f"[trial {trial.number}] {trial.state.name}: {trial.user_attrs.get('fail_reason')}")

    # -------------------------------------------------------------- optimize
    def n_complete(self, study: optuna.Study | None = None) -> int:
        study = study or self.study
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))

    def optimize(
        self,
        n_trials: int,
        max_restarts: int = 3,
        max_retries_per_trial: int = 2,
        restart_cluster_on_error: bool = True,
    ) -> pd.DataFrame:
        """Run until the study holds ``n_trials`` COMPLETE trials (counting earlier runs).

        On an exception inside a trial (e.g. a CUDA error), optionally restart the Dask
        workers, reload the data, re-queue that trial's params and continue, up to
        ``max_restarts`` times. If the Python process itself dies, simply re-run:
        completed trials are kept and the interrupted one is retried.
        Returns ``report_best()`` for the selected best trial.
        """
        study = self.load_study()
        self._recover(study, max_retries_per_trial)
        self.sync_storage()
        where = f"{self.storage} (copied to {self.volume_db})" if self.local_db else self.storage
        self._log(f"[study {self.study_name}] {self.n_complete(study)}/{n_trials} trials complete; storage={where}")

        restarts = 0
        while (remaining := n_trials - self.n_complete(study)) > 0:
            try:
                study.optimize(
                    self._objective,
                    n_trials=remaining,
                    callbacks=[self._after_trial, MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
                    gc_after_trial=True,
                )
            except KeyboardInterrupt:
                self.sync_storage()
                self._log("Interrupted by user; progress is saved. Re-run optimize() to resume.")
                raise
            except Exception as e:  # GPU / worker failures
                restarts += 1
                self.sync_storage()
                self._log(f"[error] {type(e).__name__}: {e} (restart {restarts}/{max_restarts})")
                if restarts > max_restarts:
                    raise
                if restart_cluster_on_error:
                    self.client.restart()
                self._reset_data()
                self._recover(study, max_retries_per_trial)
                self.sync_storage()
                continue
            if n_trials - self.n_complete(study) >= remaining:  # nothing completed (all NaN)
                self._log("[warn] no trial completed in the last batch; stopping.")
                break

        return self.report_best()

    # ---------------------------------------------------------------- report
    def best_trial(self, max_overfit: float | None = None) -> FrozenTrial:
        if self.study is None:
            self.load_study()
        return select_best_trial(self.study, self.max_overfit if max_overfit is None else max_overfit)

    def trial_result(self, trial: FrozenTrial | int) -> TrainResult:
        number = trial if isinstance(trial, int) else trial.number
        return TrainResult.load(self.study_dir / "trials" / f"trial_{number:05d}" / "metrics.json")

    def best_result(self, max_overfit: float | None = None) -> TrainResult:
        return self.trial_result(self.best_trial(max_overfit))

    def load_best_booster(self, max_overfit: float | None = None):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(self.best_result(max_overfit).booster_path)
        return booster

    def pareto_front(self) -> pd.DataFrame:
        rows = [
            {"trial": t.number, "loss": t.values[0], "overfit_index": t.values[1], **t.params}
            for t in self.study.best_trials
        ]
        return pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)

    def report_best(self, max_overfit: float | None = None) -> pd.DataFrame:
        """Print and return the evaluation metrics of the selected best trial."""
        trial = self.best_trial(max_overfit)
        res = self.trial_result(trial)
        table = pd.DataFrame({"train": res.train_metrics, "valid": res.valid_metrics})
        table.loc["avg_target"] = [res.avg_target_train, res.avg_target_valid]
        table.loc["avg_prediction"] = [res.avg_pred_train, res.avg_pred_valid]
        table.loc["avg_pred / avg_target"] = table.loc["avg_prediction"] / table.loc["avg_target"]
        if self.verbose:
            print(f"\n===== Best trial for study '{self.study_name}' (loss={self.loss.name}) =====")
            print(f"trial #{trial.number}  | {self.n_complete()} complete trials, {len(self.study.best_trials)} on Pareto front")
            print(f"objective metric : {self.opt_metric or res.loss_metric} = {trial.values[0]:.6g}")
            print(f"overfit index    : {trial.values[1]:.4f}")
            print(f"best_iteration   : {res.best_iteration}  ({res.train_seconds:.1f}s)")
            print(f"booster          : {res.booster_path}")
            print("params           :", json.dumps(trial.params, default=float))
            print(table.to_string(float_format=lambda v: f"{v:.6g}"))
        return table

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
