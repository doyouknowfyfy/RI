"""tsfm -- Conv1d-tokenised transformer for multivariate time-series regression.

::

    (B, C, T) -> Conv1d patch tokenizer -> systematic-missing embedding
              -> transformer (attention across features) -> pooled TS embedding
              -> late fusion with snapshot embedding -> regression head

Quick start
-----------
>>> from tsfm import DataConfig, prepare_datasets, TSFormerRegressor, Trainer, TrainConfig
>>> prep = prepare_datasets(ts_df, snap_df, target_df, data_cfg, splits)
>>> model = TSFormerRegressor(prep.model_config())
>>> hist = Trainer(model, TrainConfig(), prep.target_scaler).fit(
...     prep.datasets["train"], prep.datasets["holdout"], prep.datasets["oot"])
"""

from .config import (
    HeadConfig,
    MissingConfig,
    ModelConfig,
    SnapshotConfig,
    TokenizerConfig,
    TrainConfig,
    TransformerConfig,
)
from .data import (
    DataConfig,
    DataProcessor,
    PreparedData,
    SystematicMissingSpec,
    TSDataset,
    infer_wide_columns,
    make_dataloader,
    prepare_datasets,
    wide_to_panel,
)
from .metrics import compute_metrics, mae, nmad, r2_score, rmse
from .model import TSFormerRegressor
from .trainer import Trainer
from .utils import resolve_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "TokenizerConfig",
    "MissingConfig",
    "TransformerConfig",
    "SnapshotConfig",
    "HeadConfig",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "SystematicMissingSpec",
    "DataProcessor",
    "PreparedData",
    "TSDataset",
    "prepare_datasets",
    "make_dataloader",
    "wide_to_panel",
    "infer_wide_columns",
    "TSFormerRegressor",
    "Trainer",
    "compute_metrics",
    "rmse",
    "mae",
    "r2_score",
    "nmad",
    "set_seed",
    "resolve_device",
    "__version__",
]
