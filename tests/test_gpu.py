"""GPU-only checks.  Skipped automatically when CUDA is unavailable."""

import numpy as np
import pytest
import torch

from tsfm import (
    DataConfig, TokenizerConfig, TrainConfig, Trainer, TSFormerRegressor,
    TransformerConfig, prepare_datasets,
)
from tsfm.data import SystematicMissingSpec
from tsfm.simulate import make_synthetic_panel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def prepared():
    ts, snap, tgt, splits = make_synthetic_panel(n_records=800, n_features=4, seq_len=12, seed=11)
    cfg = DataConfig(key="account_id", time_col="month_idx",
                     ts_features=[f"f{i}" for i in range(4)], target_cols=["y"],
                     snapshot_cat=["cat0"], snapshot_num=["num0"],
                     systematic=SystematicMissingSpec(method="tenure", tenure_col="tenure_months"))
    return prepare_datasets(ts, snap, tgt, cfg, splits)


def _model(prep):
    return TSFormerRegressor(prep.model_config(
        tokenizer=TokenizerConfig(in_channels=prep.n_features, conv_channels=8,
                                  kernel_size=3, stride=2, d_model=16),
        transformer=TransformerConfig(n_layers=2, n_heads=2, d_ff=32)))


@pytest.mark.parametrize("amp_dtype", ["bf16", "fp16"])
def test_amp_training_is_finite(prepared, tmp_path, amp_dtype):
    tr = Trainer(_model(prepared),
                 TrainConfig(epochs=2, batch_size=128, device="cuda", amp=True,
                             amp_dtype=amp_dtype, num_workers=0, log_every_epoch=False,
                             early_stopping_patience=None, ckpt_path=str(tmp_path / "b.pt")),
                 prepared.target_scaler)
    hist = tr.fit(prepared.datasets["train"], prepared.datasets["holdout"], prepared.datasets["oot"])
    assert np.isfinite(hist.to_numpy(dtype=float)).all()


def test_cpu_and_gpu_predictions_agree(prepared, tmp_path):
    """fp32 forward must be device-independent to within float tolerance."""
    from tsfm import make_dataloader

    model = _model(prepared).eval()
    batch = next(iter(make_dataloader(prepared.datasets["holdout"], 32, pin_memory=False)))
    with torch.no_grad():
        cpu_out = model(batch)
        model.cuda()
        gpu_out = model({k: v.cuda() for k, v in batch.items()}).cpu()
    torch.testing.assert_close(cpu_out, gpu_out, rtol=1e-4, atol=1e-4)
