"""Distributed (Dask) GPU XGBoost regression + Optuna multi-objective tuning."""

from .cluster import make_client
from .data import build_dmatrices, load_dask_data, make_synthetic_data
from .losses import LOSSES, LossSpec, get_loss
from .storage import copy_study_to_local, is_fuse_path, load_study_from_volume, save_study_to_volume
from .trainer import TrainResult, train_xgb_model
from .tuner import XGBOptunaTuner

__all__ = [
    "LOSSES",
    "LossSpec",
    "TrainResult",
    "XGBOptunaTuner",
    "build_dmatrices",
    "copy_study_to_local",
    "get_loss",
    "is_fuse_path",
    "load_dask_data",
    "load_study_from_volume",
    "make_client",
    "make_synthetic_data",
    "save_study_to_volume",
    "train_xgb_model",
]
