import numpy as np
import pytest

from tsfm import (
    DataConfig, TrainConfig, Trainer, TSFormerRegressor, TokenizerConfig, TransformerConfig,
    prepare_datasets,
)
from tsfm.data import SystematicMissingSpec
from tsfm.metrics import compute_metrics, mae, nmad, r2_score, rmse
from tsfm.simulate import make_synthetic_panel


# --------------------------------- metrics ---------------------------------- #
def test_metrics_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["nmad"] == 0 and m["r2"] == 1


def test_rmse_mae_known_values():
    y, p = np.array([0.0, 0.0, 0.0]), np.array([1.0, -1.0, 2.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(6 / 3))
    assert mae(y, p) == pytest.approx(4 / 3)


def test_r2_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=200), rng.normal(size=200)
    assert r2_score(y, p) == pytest.approx(sk.r2_score(y, p))


def test_nmad_is_robust_to_outliers():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    p = y.copy()
    p[:20] += 50.0                                    # 1% wild outliers
    assert rmse(y, p) > 4.0                           # RMSE blows up
    assert nmad(y, p) < 0.05                          # NMAD barely moves


def test_nmad_recovers_sigma_for_gaussian_residuals():
    rng = np.random.default_rng(1)
    y = np.zeros(200_000)
    p = rng.normal(scale=2.0, size=200_000)
    assert nmad(y, p) == pytest.approx(2.0, rel=0.02)  # 1.4826 * MAD -> sigma


def test_nmad_relative_variant():
    y = np.array([10.0, 20.0, 40.0])
    p = np.array([11.0, 22.0, 44.0])                   # 10% error everywhere
    assert nmad(y, p, variant="relative") == pytest.approx(1.4826 * 0.1)
    with pytest.raises(ValueError):
        nmad(y, p, variant="nope")


def test_multi_target_metrics():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(100, 2))
    p = y + rng.normal(scale=0.1, size=(100, 2))
    m = compute_metrics(y, p, target_names=["a", "b"], per_target=True)
    assert "rmse__a" in m and "r2__b" in m
    assert m["rmse"] == pytest.approx(np.mean([m["rmse__a"] ** 2, m["rmse__b"] ** 2]) ** 0.5, rel=1e-6)


def test_metric_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((10, 1)), np.zeros((10, 2)))


# --------------------------------- training --------------------------------- #
@pytest.fixture(scope="module")
def prepared():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=600, n_features=4, seq_len=12, seed=7)
    cfg = DataConfig(
        key="account_id", time_col="month_idx", ts_features=[f"f{i}" for i in range(4)],
        target_cols=["y"], snapshot_cat=["cat0", "cat1"], snapshot_num=["num0", "num1"],
        systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"),
    )
    return prepare_datasets(ts, snap, tgt, cfg, splits)


def _small_model(prep):
    return TSFormerRegressor(prep.model_config(
        tokenizer=TokenizerConfig(in_channels=prep.n_features, conv_channels=8,
                                  kernel_size=3, stride=2, d_model=16),
        transformer=TransformerConfig(n_layers=1, n_heads=2, d_ff=32),
    ))


def _train_cfg(tmp_path, **kw):
    base = dict(epochs=3, batch_size=64, device="cpu", amp=False, num_workers=0,
                pin_memory=False, log_every_epoch=False, early_stopping_patience=None,
                ckpt_path=str(tmp_path / "best.pt"))
    base.update(kw)
    return TrainConfig(**base)


def test_history_logs_all_splits_and_metrics(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path), prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    for split in ("train", "holdout", "oot"):
        for metric in ("rmse", "mae", "r2", "nmad"):
            assert f"{split}_{metric}" in hist.columns
    assert len(hist) == 3 and hist.index.name == "epoch"
    assert np.isfinite(hist.to_numpy(dtype=float)).all()


def test_best_model_is_the_lowest_holdout_rmse(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path, epochs=4), prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    assert tr.best_epoch_ == int(hist["holdout_rmse"].idxmin())
    # the restored weights reproduce that epoch's holdout RMSE
    assert tr.evaluate(prepared.datasets["holdout"])["rmse"] == pytest.approx(
        hist["holdout_rmse"].min(), rel=1e-5)


def test_oot_never_drives_selection(prepared, tmp_path):
    """Selection must depend only on holdout, even if OOT disagrees."""
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path, epochs=4), prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    assert tr.best_epoch_ == int(hist["holdout_rmse"].idxmin())


def test_checkpoint_roundtrip(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path), prepared.target_scaler)
    tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    before = tr.evaluate(prepared.datasets["oot"])["rmse"]
    fresh = Trainer(_small_model(prepared), _train_cfg(tmp_path), prepared.target_scaler)
    fresh.load_checkpoint()
    assert fresh.evaluate(prepared.datasets["oot"])["rmse"] == pytest.approx(before, rel=1e-5)


def test_early_stopping_triggers(prepared, tmp_path):
    tr = Trainer(_small_model(prepared),
                 _train_cfg(tmp_path, epochs=50, early_stopping_patience=1, lr=1e-9),
                 prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"])
    assert len(hist) < 50


def test_metrics_are_in_original_target_units(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path, epochs=1), prepared.target_scaler)
    y_true, y_pred = tr.predict(prepared.datasets["holdout"])
    assert y_true.mean() > 10                       # ~100, not the scaled ~0
    m = tr.evaluate(prepared.datasets["holdout"])
    assert m["rmse"] == pytest.approx(rmse(y_true, y_pred), rel=1e-6)


def test_model_learns_something(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path, epochs=8, lr=3e-3),
                 prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    assert hist["holdout_rmse"].iloc[-1] < hist["holdout_rmse"].iloc[0]
    assert hist["holdout_r2"].max() > 0.1


def test_final_report_shape(prepared, tmp_path):
    tr = Trainer(_small_model(prepared), _train_cfg(tmp_path, epochs=1), prepared.target_scaler)
    tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    rep = tr.final_report(prepared.datasets)
    assert list(rep.index) == ["train", "holdout", "oot"]
    assert {"rmse", "mae", "r2", "nmad", "n"} <= set(rep.columns)


def test_dim_mismatch_is_caught_early(prepared, tmp_path):
    bad = TSFormerRegressor(prepared.model_config(
        tokenizer=TokenizerConfig(in_channels=prepared.n_features, d_model=16),
        transformer=TransformerConfig(n_layers=1, n_heads=2, d_ff=16)))
    bad.cfg.seq_len += 1                                # pretend it wants a longer window
    tr = Trainer(bad, _train_cfg(tmp_path), prepared.target_scaler)
    with pytest.raises(ValueError, match="time steps"):
        tr.fit(prepared.datasets["train"], prepared.datasets["holdout"])
