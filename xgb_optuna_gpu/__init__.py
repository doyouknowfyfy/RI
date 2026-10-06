"""Distributed (Dask) GPU XGBoost regression + Optuna multi-objective tuning."""

from .cluster import make_client
from .data import build_dmatrices, load_dask_data, make_synthetic_data
from .losses import LOSSES, LossSpec, get_loss
from .storage import SyncedSQLiteStorage, is_fuse_path
from .trainer import TrainResult, train_xgb_model
from .tuner import XGBOptunaTuner

__all__ = [
    "LOSSES",
    "LossSpec",
    "TrainResult",
    "SyncedSQLiteStorage",
    "XGBOptunaTuner",
    "build_dmatrices",
    "get_loss",
    "is_fuse_path",
    "load_dask_data",
    "make_client",
    "make_synthetic_data",
    "train_xgb_model",
]
