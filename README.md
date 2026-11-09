# RI - Time Series

`tsfusion` is a PyTorch code base (one GPU by default, several GPUs with `torchrun`) for a regression transformer that combines

* a **multivariate time series** (records × T=11 steps × V variables) with missing and special values,
* **snapshot variables** that are already piecewise-linear encoded (PLE),
* a **positive, right-skewed target** trained with **MSE** or a **tunable Tweedie** loss.

```
x (B,T,V) ─► sliding patches per variable ─► value tokens ─► [variable-attention ─► time-attention] × L ─► CLS_regular ─┐
state     ─► same patches ─► ONE embedding per missing / special code ─► [same factorized encoder]    ─► CLS_special ─┼─► head ─► ŷ
snapshot PLE ─► per-variable linear embedding ─► [CLS] + transformer                                 ─► CLS_snapshot ─┘
```

| notebook | content |
|---|---|
| [`notebooks/01_demo_late_fusion_cls.ipynb`](notebooks/01_demo_late_fusion_cls.ipynb) | **Requested design**, end to end: data structures, missing/special treatment, patching, model, MSE training, Tweedie training with a search over `p`, reloading the best checkpoint |
| [`notebooks/02_alternative_value_aware_tokens.ipynb`](notebooks/02_alternative_value_aware_tokens.ipynb) | Alternative: missing/special embeddings injected **into the value tokens** (early fusion), plus a hybrid. Controlled multi-seed comparison with the requested design |
| [`notebooks/03_demo_early_fusion.ipynb`](notebooks/03_demo_early_fusion.ipynb) | **Recommended early-fusion design**, end to end with the same steps as 01. Adds value-aware token construction, inspection of the learned missing/special embeddings, and a side-by-side with the 01 runs |
| [`notebooks/04_visualize_early_fusion.ipynb`](notebooks/04_visualize_early_fusion.ipynb) | **Inside the early-fusion model**: cell embeddings of missing/special values, CLS and snapshot-token embedding spaces, attention rollout and per-stage attention for one record, relevance of missing/special patches across records, snapshot attention, and what the prediction depends on (branch ablation, permutation importance, state probe) |
| [`notebooks/05_combined_state_input.ipynb`](notebooks/05_combined_state_input.ipynb) | **One combined missing/special matrix as input** (0 regular, 1 missing, 2..K+1 special), with unchanged embedding and fusion. Covers uint8 storage, chunked and in-place preprocessing, memory-mapped data, memory measured and projected to 2M + 1M × 11 × 460, training, and the learned code embeddings |

The notebooks were executed on CPU with synthetic data, so they use small models and few epochs. On a GPU they automatically use `d_model=64` and up to 150 epochs.

## Layout

```
tsfusion/
  config.py     ModelConfig / TrainConfig dataclasses (serialised into every checkpoint)
  data.py       DataFrame -> array helpers, RawData / ProcessedData, Preprocessor, TargetScaler, InMemoryLoader
  model.py      SlidingPatcher, factorized grid encoder with 2-D CLS, state / value / snapshot tokenizers, TSFusionTransformer
  losses.py     MSELoss, TweedieLoss (log link, 1 <= p <= 3), tweedie_deviance
  metrics.py    RMSE, MAE, R2, NMAD (original target scale)
  trainer.py    Trainer (one GPU or DDP, AMP, per-epoch logs, checkpoints, early stopping, resume), load_checkpoint, predict_with
  distributed.py  multi-GPU helpers (DDP process group, gather / broadcast)
  interpret.py  attention recording + rollout, missing/special relevance, embedding maps, ablation / permutation / state probes
  viz.py        matplotlib figures for interpret results (state-aware colours and glyphs)
  synthetic.py  synthetic data in the same layout as the real data
scripts/        train.py (command line, one GPU or torchrun) and run_grid.sh (one experiment per GPU)
tests/          pytest suite (63 tests incl. 2-process DDP, attention rollout, combined-state, memory and no-append-storage checks, CPU, about 1 min)
notebooks/      the three demo notebooks
```

## Quick start

```python
from tsfusion import *

# 1) arrays, all in the SAME record order
#    x: (N, 11, V) float32
#    state: (N, 11, V) uint8, ONE matrix: 0 regular, 1 missing, 2..K+1 special value 1..K
#           (or pass the two matrices instead: RawData(x, missing, special, snapshot, y))
#    snapshot: (N, sum of PLE bins) float32, y: (N,)
groups = group_columns_by_prefix(snapshot_ple_df.columns)            # [(feature, [cols]), ...]
cols = [c for _, cs in groups for c in cs]
group_sizes = [len(cs) for _, cs in groups]
train_raw = RawData(x_tr, state=state_tr, snapshot=snapshot_ple_df.loc[train_ids, cols].to_numpy(), y=y_tr)
val_raw   = RawData(x_va, state=state_va, snapshot=snapshot_ple_df.loc[val_ids, cols].to_numpy(), y=y_va)

# 2) preprocessing, fitted on train only (works in chunks; inplace=True reuses the input arrays)
loss = "tweedie"                                                     # or "mse"
prep = Preprocessor(n_special_codes=K, snapshot_group_sizes=group_sizes,
                    target_mode=default_target_mode(loss)).fit(train_raw)
train = prep.transform(train_raw, inplace=True)                      # train_raw is consumed
val = prep.transform(val_raw, inplace=True)

# 3) model + training on one GPU
model_cfg = ModelConfig(n_vars=V, n_steps=11, n_special_codes=K, snapshot_group_sizes=group_sizes,
                        missing_fusion="late")                      # "late" = requested 3-CLS design
train_cfg = TrainConfig(loss=loss, tweedie_power=1.5, device="cuda:0", output_dir="runs/exp1")
trainer = Trainer(model_cfg, train_cfg, prep)
history = trainer.fit(train, val)                                    # DataFrame, one row per epoch
y_hat = trainer.predict(val)                                         # original scale, best epoch

# 4) later / elsewhere
model, prep, cfg, ckpt = load_checkpoint("runs/exp1/best_model.pt", device="cuda:0")
y_new = predict_with(model, prep, cfg, prep.transform(new_raw))
```

`long_df_to_array` / `wide_df_to_array` build the `(N, T, V)` arrays from long `(record_id, time_step) × variables` frames or from wide `var_t1 … var_t11` frames. `encode_special_values(x_raw, {-999: 1, -998: 2}, special_flag)` turns sentinel values into codes.

## How the requirements map to the code

| requirement | implementation |
|---|---|
| single GPU | `TrainConfig.device="cuda:0"` by default; the data can be kept on the GPU (`data_on_device`). Mixed precision is bf16 when supported, otherwise fp16 with GradScaler. Optional `torch.compile`. Several GPUs: see below. |
| sliding patching per variable | `SlidingPatcher`: `(B,T,V) → (B,V,Np,P)`; `patch_len`, `patch_stride`; right-padding so the latest step is always covered |
| one embedding for missing and for each special value | the state code is 0 regular, 1 missing, 1+k special k. `StatePatchEmbedding` holds one `nn.Embedding(2+K, ·)` shared by all variables and steps |
| attention across variables, then across time, with CLS | `FactorizedBlock` + `GridEncoder` on a `(V+1)×(Np+1)` grid. Row 0 holds per-patch CLS tokens, column 0 per-variable CLS tokens, and the global CLS sits at (0,0) |
| snapshot embedding | `SnapshotEncoder`: per-variable linear embedding of the PLE vector (FT-Transformer tokenizer), then [CLS] and a transformer. A raw variable is a group of size 1 |
| head input = 3 CLS tokens | `missing_fusion="late"`: `head(concat[CLS_regular, CLS_special, CLS_snapshot])` |
| MSE and tunable Tweedie | `MSELoss` (standardised target), `TweedieLoss(p)` (log link, target / train mean). Tune `p` on validation |
| per-epoch train and holdout RMSE / MAE / R2 / NMAD | `Trainer.fit` computes them in eval mode, on the original scale, every epoch; written to `metrics.csv`, `train.log` and the returned DataFrame |
| save the model every epoch | `epoch_XXX.pt`: model, optimizer, scheduler and scaler state, configs, fitted preprocessor. Loadable with `torch.load(weights_only=True)` |
| model selected by lowest holdout RMSE | `best_model.pt`; the best weights are restored at the end of `fit` |
| early stopping | `early_stopping_patience` / `early_stopping_min_delta` on validation RMSE |

## Using several GPUs

`device="cuda:0"` means GPU number 0. One GPU already runs all of its own cores, so a single-GPU run leaves the other GPUs idle. There are two ways to use them, and they suit different goals.

**A. Several experiments at once, one per GPU** (best for tuning). Each GPU trains its own configuration: a different Tweedie power, seed, fusion mode or model size. There is no communication between GPUs, so 8 GPUs give 8 times the experiments, and each run behaves exactly like a single-GPU run.

```bash
bash scripts/run_grid.sh data/prepared runs/grid                 # edit CONFIGS in the script
EXTRA_ARGS="--epochs 60 --scheduler cosine" bash scripts/run_grid.sh data/prepared runs/grid
```

**B. One run spread over all GPUs (DDP)** (best when a single run on large data is too slow). Launch the same script with `torchrun`; no code change is needed:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py --data data/prepared --output-dir runs/ddp \
    --batch-size 256 --lr 2e-3
```

* One process per GPU (`cuda:LOCAL_RANK`). Each GPU takes a shard of every global batch, and gradients are averaged across GPUs.
* Evaluation is also sharded, and the predictions are gathered, so the metrics, best-epoch selection and early stopping are identical on every GPU.
* Only the first process (rank 0) writes `train.log`, `metrics.csv` and the checkpoints. Checkpoints load exactly like single-GPU ones.
* **`--batch-size` is per GPU**: the global batch is `batch_size × #GPUs`. With dropout off, 2 GPUs × 64 follows the same training path as 1 GPU × 128 (tested to within 1e-7 relative).
* A larger global batch means fewer optimizer steps per epoch. Either raise the learning rate (start with about `sqrt(#GPUs) ×`; with 8 GPUs, try 2–3×) and re-check the warm-up, or keep the global batch unchanged by dividing the per-GPU batch by the number of GPUs.
* `--warmup-epochs`, `--patience` and the schedulers keep their meaning in epochs.

**The data for the script** is written once from Python with `save_dataset("data/prepared", {"train": train_raw, "val": val_raw}, n_special_codes=K, snapshot_group_sizes=group_sizes)`. This creates plain `.npy` files plus `meta.json`, which every process reads. `--data synthetic` runs the demo data.

Which to use: the model is small (tens to hundreds of thousands of parameters), so DDP pays off only when one epoch over your data takes long. For choosing `p`, seeds or architecture, option A gives the most work per GPU hour. Notebooks always run on one GPU; DDP needs `torchrun`, so it runs through the script.

## Running on Databricks

* **Runtime:** use Databricks Runtime 18 LTS ML (GPU) and don't reinstall PyTorch. The code runs on PyTorch ≥ 2.1.
* **`output_dir` on a Volume** (`/Volumes/...`), `/dbfs/...` or `/Workspace/...`: these locations don't support appending to a file or out-of-order writes (zip files such as `torch.save` checkpoints). The trainer detects such paths and writes logs, metrics and checkpoints on the local disk first (`/local_disk0/tmp/...`). It then copies each finished file to `output_dir` in one write (`TrainConfig.local_staging="auto"`, `staging_dir`).
* **Large data:** keep `x` as float32 and `state` as uint8, preprocess in place or from memory-mapped files (notebook 05), and let the trainer stream batches to the GPU.
* **Debugging a CUDA error:** set `os.environ["CUDA_LAUNCH_BLOCKING"] = "1"` in a fresh Python process before the first GPU use (`dbutils.library.restartPython()` first). The traceback then points at the operation that failed. Remove it afterwards, because it slows training a lot.

## Code review: what was wrong in the first draft and is now fixed

I reviewed the first version myself and had an independent review run on it. That review checked the factorized attention against a naive loop, the masks, the Tweedie gradients, checkpoint loading and resume. Every fix below has a regression test in `tests/test_tsfusion.py`.

1. **`long_df_to_array` sorted string time labels alphabetically**, so `t1, t10, t11, t2, …` silently scrambled time order. It now requires explicit `time_steps` for non-numeric labels. It also raises if none of the requested rows exist (for example a record-id dtype mismatch, which used to turn everything into "missing") and warns about absent rows.
2. **PLE bins could be assigned to the wrong variable** when the PLE columns of different features were interleaved, and the grouping regex merged raw columns like `x_2023` into one group. The pattern now requires `_bin<j>`, there is a warning for interleaving, and the docs and notebooks build the matrix in the returned column order.
3. **Sentinel matching used a relative tolerance**, so `-999.005` and `99999.5` counted as special. It is now exact.
4. **The model output was rounded to bf16 on the GPU.** bf16 keeps about 3 significant digits, so predictions, metrics and the best-epoch choice carried noise of up to about 1–2%. The head now runs in fp32.
5. **After resuming with the cosine schedule, the first step used the wrong learning rate.** It is restored now.
6. **Stale files from an earlier run in the same `output_dir` could be reloaded as the "best" model.** The best weights now live in memory. A fresh run clears the old checkpoints (or raises if `overwrite_output_dir=False`), and `resume()` drops files from the abandoned epochs.
7. **Weight decay was not applied to some projection matrices** (a name-matching bug). Fixed.
8. **The model now zeroes values at missing or special cells itself**, so a sentinel such as -999 cannot leak into the value pathway even if preprocessing is skipped.
9. Smaller fixes:
   * `patch_stride > patch_len` (which skips time steps) is rejected;
   * `+inf` in raw snapshot columns is handled;
   * a `tweedie_dev` metric is logged, because Tweedie loss values are not comparable across `p`;
   * the default `eval_batch_size` is lower;
   * there is an optional `gradient_checkpointing` and an optional log transform for heavy-tailed inputs (`Preprocessor(ts_log_vars=[...])`).

## Concerns about the task

1. **Selecting by RMSE on a heavy-tailed target.** RMSE on the original scale is driven by a few extreme records. The best epoch, early stopping and the LR schedule will all react to them, and RMSE structurally favours MSE-trained models over Tweedie ones. Watch MAE, NMAD and the Tweedie deviance too, and make sure RMSE really is the business objective.
2. **Validation reuse.** Choosing the epoch, `p` and the architecture on the same holdout makes the holdout numbers optimistic. Keep an untouched test split for the final figure.
3. **Tweedie with a strictly positive target.** `1 < p < 2` assumes a point mass at zero. The loss is still a consistent estimator of the mean for every `p` and only changes the weighting, but `p ≈ 2` (Gamma) to 3 is the natural family for y > 0. `p` must be tuned, not learned.
4. **MSE choices.** MSE on the raw (standardised) target is what RMSE measures, but extreme values dominate the gradient (we clip gradients). MSE on `log1p(y)` (`target_mode="log1p_standard"`) is more stable, but back-transformed predictions are biased low for the mean (they are closer to the median), which hurts RMSE.
5. **Late fusion cannot relate a missing value to the observed values around it**, because the two branches only meet in the head. Notebook 02 shows the token-level alternative. Run the comparison on your data.
6. **Short series.** With T=11, sliding patches give few tokens: 9 with `P=3, S=1`, 5 with `P=4, S=2`. `patch_len=1` (one token per cell) is affordable and worth trying.
7. **Memory scales with the number of records and the number of variables.** Preprocessing works in chunks, so its working memory is under 1 GB at the default chunk size. The processed data costs 5 bytes per cell (float32 value + uint8 state code); with `transform(..., inplace=True)` it reuses the input arrays, and with `load_dataset(..., mmap=True)` the raw arrays stay on disk. For 2M + 1M records × 11 × 460 that is about 78 GB of RAM at the peak, down from about 607 GB with the earlier separate-matrix, whole-array preprocessing (measured in notebook 05). On the GPU, the data stays in CPU memory and goes over one batch at a time unless it needs less than 30% of the free GPU memory (`data_on_device="auto"`). Activations: each grid branch has `(V+1)(Np+1)` tokens per record, and attention across variables is O(V²). For V in the hundreds, reduce `batch_size` / `eval_batch_size` or `d_model`, use a larger stride, enable `gradient_checkpointing`, and keep AMP on. `gpu_peak_mem_mb` is logged each epoch.
8. **Leakage risk upstream.** PLE bins, scaling statistics and the target transform must be fitted on the training split only. The code does this for its own steps, but your PLE vectors come from upstream.
9. **Data contract.**
   * Missing snapshot values must be encoded explicitly (NaN raises an error).
   * A cell flagged both missing and special is treated as special (with a warning).
   * Time steps must have the same meaning and order for every record.
10. **Benchmark against GBDT.** On tabular-style data of moderate size, a LightGBM/XGBoost model with a Tweedie objective on hand-built time-series features is a strong baseline. Compare against it before investing in tuning the transformer.
11. **NMAD definition.** The code uses `1.4826 · median(|e − median(e)|)` with `e = ŷ − y`, and `TrainConfig(nmad_relative=True)` uses relative errors instead. Check that this matches the definition your team uses.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q tests        # includes 2-process DDP tests on CPU (gloo backend)
```
