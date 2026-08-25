# tsfm — time-series transformer for regression

```
time series (B, C, T)
   -> Conv1d local feature extractor / patch tokenizer      (B, C, N, d)
   -> systematic-missing embedding (additive | concat)
   -> transformer, attention ACROSS FEATURES                (B, C, N, d)
   -> masked pooling  ->  time-series embedding             (B, d_ts)
   -> LATE FUSION with the snapshot embedding               (B, d_ts + d_snap)
   -> regression head                                       (B, n_targets)
```

`B` records, `C` features (conv input channels), `T` time steps, `N` patch tokens.

## Install

```bash
pip install -r requirements.txt      # or: pip install -e .
pytest -q                            # 84 tests (GPU ones auto-skip on CPU)
```

Start with **`notebooks/demo_timeseries_transformer.ipynb`** — it runs the whole
pipeline on synthetic credit-card-like data and shows where to plug in your own.

## Quick start

```python
from tsfm import DataConfig, TokenizerConfig, TrainConfig, Trainer, TSFormerRegressor, prepare_datasets
from tsfm.data import SystematicMissingSpec

data_cfg = DataConfig(
    key="account_id", time_col="month_idx",
    ts_features=[f"f{i}" for i in range(8)],
    target_cols=["y"],
    snapshot_cat=["segment"], snapshot_num=["income", "tenure_months"],
    time_order="oldest_first",
    systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"),
)
prep = prepare_datasets(ts_long, snapshot, target, data_cfg,
                        splits={"train": train_ids, "holdout": holdout_ids, "oot": oot_ids})

model = TSFormerRegressor(prep.model_config(
    tokenizer=TokenizerConfig(in_channels=prep.n_features, conv_channels=32,
                              kernel_size=5, stride=2, d_model=64)))

trainer = Trainer(model, TrainConfig(epochs=30, device="cuda"), prep.target_scaler)
history = trainer.fit(prep.datasets["train"], prep.datasets["holdout"], prep.datasets["oot"])
report  = trainer.final_report(prep.datasets)
```

## Layout

| file | contents |
|---|---|
| `tsfm/config.py` | every hyper-parameter, as dataclasses |
| `tsfm/data.py` | the three-table processor, missing-mask logic, `TSDataset` |
| `tsfm/modules/tokenizer.py` | grouped Conv1d patch tokenizer + exact mask pooling |
| `tsfm/modules/missing.py` | learned systematic-missing embedding (`add` / `concat`) |
| `tsfm/modules/encoder.py` | axial transformer (feature / time / factorized / full) + masked pooling |
| `tsfm/modules/snapshot.py` | static categorical + numeric branch |
| `tsfm/modules/head.py` | late-fusion regression head |
| `tsfm/model.py` | `TSFormerRegressor` — the assembled network |
| `tsfm/metrics.py` | RMSE, MAE, R², NMAD |
| `tsfm/trainer.py` | training loop, per-epoch logging, holdout-RMSE selection |
| `tsfm/simulate.py` | synthetic panel generator for the demo and tests |

## Design notes

**Tokenizer.** `groups=C` keeps each feature's filters private, so every token
belongs to exactly one feature — that is what makes attention *across features*
meaningful. `pad_to_cover=True` guarantees the last (usually most recent) time
step lands inside a patch window; without it a geometry like `T=24, k=5, s=2`
silently drops month 24.

**Two kinds of missing.**

| kind | meaning | handling |
|---|---|---|
| random | value exists but was not recorded | zero-fill after scaling (= the training mean) + observed-mask conv channel |
| systematic | the period does not exist for this record (5-month-old card in a 24-month window) | learned embedding, combined with the patch token by `add` or `concat` |

The systematic mask is pushed through the tokenizer's *exact* geometry
(`pool_mask`), so each token gets the **fraction** of its window that is missing —
patches straddling the boundary are handled continuously rather than by a hard
flag. Tokens that are 100% systematic are excluded from attention and pooling.

**Metrics.** RMSE / MAE / R² / NMAD on train, holdout and OOT every epoch.
Selection is on the **lowest holdout RMSE**; OOT is reported, never used to
choose. NMAD defaults to `1.4826 · median(|r − median(r)|)` — a robust,
outlier-insensitive residual spread in the target's own units. All metrics are
computed after inverse-transforming to the original target scale.
