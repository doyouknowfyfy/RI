"""CPU smoke tests (no GPU needed): run with ``pytest -q tests``."""

import functools
import json

import dask.dataframe as dd
import optuna
import pytest
from optuna.trial import TrialState

import xgb_optuna_gpu.tuner as tuner_mod
from xgb_optuna_gpu import XGBOptunaTuner, load_dask_data, make_client, make_synthetic_data


@pytest.fixture(scope="module")
def client():
    c = make_client(use_gpu=False, n_workers=2)
    yield c
    c.close()


@pytest.fixture(scope="module")
def loader(client, tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    tr, va = make_synthetic_data(n_train=3000, n_valid=1000, n_features=6)
    dd.from_pandas(tr, npartitions=4).to_parquet(d / "train.parquet")
    dd.from_pandas(va, npartitions=4).to_parquet(d / "valid.parquet")
    return functools.partial(
        load_dask_data, str(d / "train.parquet"), str(d / "valid.parquet"), "y",
        use_gpu=False, client=client,
    )


def _tuner(client, loader, out, loss, **kw):
    return XGBOptunaTuner(
        client, loader, loss=loss, output_dir=out, use_gpu=False,
        num_boost_round=200, n_startup_trials=2, verbose=False, **kw,
    )


@pytest.mark.parametrize("loss", ["mse", "huber", "tweedie"])
def test_each_loss(client, loader, tmp_path, loss):
    extra = {"fixed_params": {"huber_slope": 3.0}} if loss == "huber" else {}
    t = _tuner(client, loader, tmp_path, loss, **extra)
    table = t.optimize(n_trials=3)
    assert t.n_complete() == 3
    res = t.best_result()
    assert list(res.valid_metrics)[-1] == res.loss_metric  # loss metric drives early stopping
    assert {"rmse", "mae", "mphe"} <= set(res.valid_metrics)
    assert res.avg_target_valid > 0 and res.avg_pred_valid > 0
    assert (t.study_dir / "trials.csv").exists()
    assert t.load_best_booster().num_boosted_rounds() == res.best_iteration + 1
    assert "avg_prediction" in table.index
    if loss == "tweedie":
        assert res.loss_metric.startswith("tweedie-nloglik@")


def test_resume_after_crash_and_gpu_error(client, loader, tmp_path, monkeypatch):
    t = _tuner(client, loader, tmp_path, "tweedie")
    t.optimize(n_trials=2)

    # Simulate a kernel crash mid-trial: a trial left RUNNING in storage.
    study = optuna.load_study(study_name=t.study_name, storage=t.storage_url)
    orphan = study.ask()
    orphan.suggest_int("max_depth", 3, 12)

    # Simulate one GPU error on the next training call.
    real_train = tuner_mod.train_xgb_model
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated CUDA error")
        return real_train(*a, **k)

    monkeypatch.setattr(tuner_mod, "train_xgb_model", flaky)
    t2 = _tuner(client, loader, tmp_path, "tweedie")
    t2.optimize(n_trials=5, restart_cluster_on_error=False)

    trials = t2.study.get_trials()
    assert t2.n_complete() == 5
    states = {tr.number: tr.state for tr in trials}
    assert states[orphan.number] == TrialState.FAIL
    retry_of = [tr.user_attrs.get("retry_of") for tr in trials]
    assert orphan.number in retry_of  # interrupted trial re-queued
    failed = [tr for tr in trials if tr.user_attrs.get("fail_reason", "").startswith("RuntimeError")]
    assert len(failed) == 1 and failed[0].number in retry_of  # GPU-error trial re-queued
    retried = next(tr for tr in trials if tr.user_attrs.get("retry_of") == failed[0].number)
    assert retried.params == failed[0].params
    json.loads((t2.study_dir / "trials" / f"trial_{retried.number:05d}" / "metrics.json").read_text())


def test_fuse_path_detection():
    from xgb_optuna_gpu.storage import is_fuse_path, normalize_path

    assert is_fuse_path("/dbfs/tmp/x") and is_fuse_path("dbfs:/tmp/x")
    assert is_fuse_path("/Volumes/cat/schema/vol/runs") and is_fuse_path("/Workspace/Users/a/runs")
    assert not is_fuse_path("/local_disk0/tmp") and not is_fuse_path("/dbfsx/a")
    assert str(normalize_path("dbfs:/a/b")) == "/dbfs/a/b"


def test_synced_sqlite_resume(client, loader, tmp_path):
    """Simulates Databricks: live DB on local disk, snapshot in output_dir, local disk wiped."""
    out, local = tmp_path / "volume", tmp_path / "local_disk"
    t = _tuner(client, loader, out, "mse", sync_db=True, local_db_dir=local)
    assert str(local) in t.storage_url and t.local_db == local / f"{t.study_name}.db"
    t.optimize(n_trials=2)

    snapshot = t.study_dir / "study.db"
    assert snapshot.exists()
    assert len(optuna.load_study(study_name=t.study_name, storage=f"sqlite:///{snapshot}").trials) == 2

    for f in local.iterdir():  # cluster restart wipes the driver's local disk
        f.unlink()
    t2 = _tuner(client, loader, out, "mse", sync_db=True, local_db_dir=local)
    t2.optimize(n_trials=3)
    assert t2.n_complete() == 3
    assert len(optuna.load_study(study_name=t.study_name, storage=f"sqlite:///{snapshot}").trials) == 3


def test_load_study_from_volume(tmp_path):
    """Standalone helper: volume copy -> local disk, interrupted trial marked FAIL + retried."""
    from xgb_optuna_gpu import load_study_from_volume, save_study_to_volume

    volume_db, local_db = tmp_path / "volume" / "study.db", tmp_path / "local" / "study.db"
    local_db.parent.mkdir()
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{local_db}", directions=["minimize"] * 2)
    objective = lambda tr: (tr.suggest_float("x", 0, 1), tr.suggest_float("y", 0, 1))
    study.optimize(objective, n_trials=3)
    crashed = study.ask()
    crashed.suggest_float("x", 0, 1)
    save_study_to_volume(local_db, volume_db)
    local_db.unlink()  # cluster restart

    study, local = load_study_from_volume("s", volume_db, local_db)
    assert local == local_db and local_db.exists()
    assert study.trials[crashed.number].state == TrialState.FAIL
    study.optimize(objective, n_trials=1, callbacks=[lambda st, tr: save_study_to_volume(local, volume_db)])
    retry = study.trials[-1]
    assert retry.state == TrialState.COMPLETE and retry.params["x"] == crashed.params["x"]
    saved = optuna.load_study(study_name="s", storage=f"sqlite:///{volume_db}")
    assert len(saved.trials) == 5
