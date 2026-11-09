"""Distributed (Dask) GPU XGBoost regression + Optuna multi-objective tuning."""

from .cluster import make_client
from .data import build_dmatrices, load_dask_data, load_dask_split, make_synthetic_data
from .evaluate import decile_table, predict, regression_metrics, to_numpy
from .final_fit import clean_params, load_tuned_params, save_model_bundle
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
    "clean_params",
    "copy_study_to_local",
    "decile_table",
    "get_loss",
    "is_fuse_path",
    "load_dask_data",
    "load_dask_split",
    "load_study_from_volume",
    "load_tuned_params",
    "make_client",
    "make_synthetic_data",
    "predict",
    "regression_metrics",
    "save_model_bundle",
    "save_study_to_volume",
    "to_numpy",
    "train_xgb_model",
]
