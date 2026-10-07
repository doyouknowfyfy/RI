# RI

Distributed GPU XGBoost regression with resumable Optuna multi-objective hyper-parameter tuning,
for a positive, right-skewed target with extreme values.

## Layout

```
xgb_optuna_gpu/
├── cluster.py   make_client(): LocalCUDACluster (GPU) or LocalCluster (CPU smoke tests)
├── data.py      load_dask_data() -> dask_cudf collections; build_dmatrices() -> DaskQuantileDMatrix
├── losses.py    MSE / pseudo-Huber / Tweedie specs (objective, loss metric, extra tunables)
├── trainer.py   train_xgb_model(): one dask.xgboost run + early stopping + saved booster/metrics
├── storage.py   copy the study DB between /local_disk0 and a volume (Databricks)
└── tuner.py     XGBOptunaTuner: TPE multi-objective study, resume, best-model report
notebooks/demo_xgb_optuna_gpu.ipynb   end-to-end demo
tests/test_smoke.py                   CPU tests (losses, resume after crash / GPU error, synced SQLite)
```

## What it does

| Item | Implementation |
|---|---|
| Data | `dask_cudf.read_parquet/csv` → persisted → `xgboost.dask.DaskQuantileDMatrix` (valid uses `ref=dtrain`); built once, reused by every trial |
| Training | `xgboost.dask.train`, `tree_method="hist"`, `device="cuda"` |
| Losses | `mse` → `reg:squarederror` / `rmse`; `huber` → `reg:pseudohubererror` / `mphe` (`huber_slope` fixed, set per dataset); `tweedie` → `reg:tweedie` / `tweedie-nloglik@p` with `tweedie_variance_power` tuned in [1.05, 1.95] |
| Eval metrics | `["rmse", "mae", "mphe"]` + the loss metric, which goes last |
| Early stopping | 50 rounds on the loss metric on the validation set; booster truncated to the best iteration |
| Saved per model | `booster.ubj` + `metrics.json` (params, train/valid metrics at best iteration, overfit index, avg target & avg prediction for train/valid, full history) |
| Tuned | `max_depth, learning_rate, gamma, min_child_weight, max_delta_step, reg_alpha, reg_lambda` (ranges in `tuner.DEFAULT_SEARCH_SPACE`) |
| Sampler | `TPESampler(n_startup_trials=..., gamma=min(ceil(fraction*n), max))` |
| Objectives | minimise (1) valid loss metric, (2) overfit index `|valid - train| / |train|` on the same metric |
| Persistence | SQLite study committed after each trial + `trials.csv` snapshot; re-run resumes (on Databricks volumes the DB runs on `/local_disk0` and is copied to the volume) |
| Best model | lowest loss on the Pareto front with overfit ≤ `max_overfit` (otherwise lowest loss) |

## Quick start

```python
import functools
from xgb_optuna_gpu import XGBOptunaTuner, load_dask_data, make_client

client = make_client(use_gpu=True)
loader = functools.partial(load_dask_data, "data/train", "data/valid", "y", use_gpu=True, client=client)

tuner = XGBOptunaTuner(client, loader, loss="tweedie", output_dir="optuna_runs",
                       n_startup_trials=15, gamma_fraction=0.1, max_overfit=0.1)
tuner.optimize(n_trials=100)   # prints the best model's metrics at the end
```

### Resuming

`n_trials` is the total number of completed trials the study should reach. If a run stops:

* **Exception inside a trial** (e.g. CUDA OOM): `optimize` restarts the Dask workers, calls `data_loader` again,
  re-queues that trial's hyper-parameters and continues (`max_restarts`, default 3).
* **Kernel/process dies**: run the same code again. Completed trials are kept, the trial left `RUNNING` is
  marked failed and its params re-queued, and only the remaining trials run.

Only one process should drive a given study at a time.

### Databricks: `sqlite3.OperationalError: disk I/O error`

SQLite needs file locking and random writes, and Databricks FUSE mounts (`/Volumes`, `/dbfs`,
`/Workspace`) support neither. So the study runs on local disk and only whole-file copies go to the volume:

1. creating the tuner copies `<output_dir>/<study_name>/study.db` to `/local_disk0/tmp/optuna/<study_name>.db`
   (if the volume copy is newer than the local one, or there is no local copy),
2. Optuna reads and writes only the local file,
3. the local file is `shutil.copy`-ed back to the volume at the start and end of every trial.

This happens automatically when `output_dir` is on a FUSE mount:

```python
tuner = XGBOptunaTuner(client, loader, loss="tweedie",
                       output_dir="/Volumes/main/ml/artifacts/optuna_runs")
tuner.optimize(n_trials=100)     # run again after a crash / cluster restart to continue
# force it on any path: sync_db=True; choose the local dir: local_db_dir="/local_disk0/tmp/optuna"
# or use a server DB instead of SQLite: storage="postgresql://user:pw@host/db"
```

To continue a saved study with your own objective, without the tuner:

```python
from xgb_optuna_gpu import load_study_from_volume, save_study_to_volume

volume_db = "/Volumes/main/ml/artifacts/optuna_runs/xgb_tweedie/study.db"
study, local_db = load_study_from_volume("xgb_tweedie", volume_db)    # copy to /local_disk0, load, fix crashed trials
study.sampler = optuna.samplers.TPESampler(n_startup_trials=15, seed=42 + len(study.trials))
try:
    study.optimize(objective, n_trials=20, callbacks=[lambda st, tr: save_study_to_volume(local_db, volume_db)])
finally:
    save_study_to_volume(local_db, volume_db)
```

Don't open `study.db` on the volume directly with `optuna.load_study`; you'll get the same error.
Copy it to local disk first (`copy_study_to_local`).

## Notes

* XGBoost has no `mph` metric. This code uses `mphe` (mean pseudo-Huber error), which is what XGBoost calls it.
* `mphe` uses `huber_slope` (default 1.0). Set `huber_slope` to fit your target's scale.
* `tweedie-nloglik@p` can be negative, so the overfit index divides by `|train|`. Its values also differ
  between different `p`. If you want a scale that doesn't depend on `p` for the objectives, pass `opt_metric="rmse"` (training and early stopping still
  use the Tweedie loss).
* Use several files or row groups per split (at least one per GPU) so every GPU gets data.

## Tests

```
pip install -e .[dev]
pytest -q tests        # CPU only, about 30 s
```
