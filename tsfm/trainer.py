"""Training loop with per-epoch metrics for train / holdout / OOT.

Model selection: **lowest holdout RMSE** (configurable via
``TrainConfig.monitor``).  The OOT split is *never* used for selection or early
stopping -- it is scored every epoch purely so you can watch the temporal
generalisation gap.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataclasses import replace

from .config import TrainConfig
from .data import TSDataset, make_dataloader
from .metrics import METRIC_MODES, compute_metrics
from .utils import resolve_device, set_seed

__all__ = ["Trainer", "EpochLog"]

_BATCH_KEYS = ("x", "obs_mask", "sys_mask", "cat", "num", "num_missing")


class EpochLog(list):
    """History of per-epoch metric dicts, with a ``to_frame()`` view."""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self).set_index("epoch")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device, non_blocking: bool = True):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


class Trainer:
    """Train / evaluate a :class:`~tsfm.model.TSFormerRegressor`.

    Parameters
    ----------
    model :
        The network.  Moved to ``cfg.device`` here.
    cfg :
        :class:`~tsfm.config.TrainConfig`.
    target_scaler :
        The scaler returned by the data processor.  Predictions are
        inverse-transformed with it before metrics, so RMSE/MAE/NMAD are always
        reported in the **original target units**.
    target_names :
        Used for the per-target metric breakdown.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        target_scaler=None,
        target_names: Optional[Sequence[str]] = None,
        nmad_variant: str = "residual",
    ):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.target_scaler = target_scaler
        self.target_names = list(target_names) if target_names else None
        self.nmad_variant = nmad_variant

        # keep an uncompiled handle: torch.compile wraps the module and would
        # otherwise prefix every checkpoint key with "_orig_mod."
        self.model_raw = self.model
        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.criterion = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(delta=cfg.huber_delta),
        }[cfg.loss]

        self.amp_enabled = bool(cfg.amp) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
        if self.amp_enabled and self.amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("[tsfm] bf16 not supported on this GPU -- using fp16.")
            self.amp_dtype = torch.float16
        # GradScaler is only needed for fp16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        )
        self.history = EpochLog()
        self.best_state_: Optional[Dict[str, torch.Tensor]] = None
        self.best_score_: float = math.inf
        self.best_epoch_: int = -1

    def _check_dims(self, ds: TSDataset) -> None:
        """Fail loudly (and early) on a model/dataset shape mismatch."""
        cfg = getattr(self.model_raw, "cfg", None)
        if cfg is None:
            return
        if ds.n_features != cfg.tokenizer.in_channels:
            raise ValueError(
                f"dataset has {ds.n_features} features but the model expects "
                f"{cfg.tokenizer.in_channels} (TokenizerConfig.in_channels)"
            )
        if ds.seq_len != cfg.seq_len:
            raise ValueError(
                f"dataset has {ds.seq_len} time steps but the model was built for "
                f"{cfg.seq_len} (ModelConfig.seq_len)"
            )
        if ds.y.shape[1] != cfg.head.n_targets:
            raise ValueError(
                f"dataset has {ds.y.shape[1]} targets but the model predicts "
                f"{cfg.head.n_targets} (HeadConfig.n_targets)"
            )

    # ------------------------------------------------------------------ #
    def _build_scheduler(self, steps_per_epoch: int):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1), eta_min=cfg.min_lr
            ), "epoch"
        if cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.monitor_mode,
                factor=0.5,
                patience=max(1, (cfg.early_stopping_patience or 10) // 2),
                min_lr=cfg.min_lr,
            ), "plateau"
        if cfg.scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps_per_epoch
            ), "step"
        return None, "none"

    def _warmup_lr(self, epoch: int) -> None:
        cfg = self.cfg
        if cfg.warmup_epochs and epoch < cfg.warmup_epochs:
            scale = (epoch + 1) / cfg.warmup_epochs
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.lr * scale

    # ------------------------------------------------------------------ #
    def train_one_epoch(self, loader: DataLoader, scheduler=None, sched_mode: str = "none") -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            y = batch["y"]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                pred = self.model(batch)
                loss = self.criterion(pred, y)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scheduler is not None and sched_mode == "step":
                scheduler.step()
            bs = y.shape[0]
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, data, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(y_true, y_pred)`` in the **original** target units."""
        loader = self._as_loader(data, batch_size)
        self.model.eval()
        preds, trues = [], []
        for batch in loader:
            batch = _to_device(batch, self.device)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                out = self.model(batch)
            preds.append(out.float().cpu())
            trues.append(batch["y"].float().cpu())
        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(trues).numpy()
        if self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred)
            y_true = self.target_scaler.inverse_transform(y_true)
        return y_true, y_pred

    @torch.no_grad()
    def evaluate(
        self, data, batch_size: Optional[int] = None, per_target: bool = False
    ) -> Dict[str, float]:
        y_true, y_pred = self.predict(data, batch_size)
        return compute_metrics(
            y_true,
            y_pred,
            nmad_variant=self.nmad_variant,
            target_names=self.target_names,
            per_target=per_target,
        )

    def _as_loader(self, data, batch_size: Optional[int] = None) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        return make_dataloader(
            data,
            batch_size or self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and self.device.type == "cuda",
            persistent_workers=self.cfg.persistent_workers,
        )

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train: TSDataset,
        holdout: TSDataset,
        oot: Optional[TSDataset] = None,
        extra_eval: Optional[Dict[str, TSDataset]] = None,
    ) -> pd.DataFrame:
        """Train, logging RMSE/MAE/R2/NMAD on every split each epoch.

        Returns the history as a ``DataFrame``.  The best (lowest holdout RMSE)
        weights are restored into ``self.model`` at the end and written to
        ``TrainConfig.ckpt_path``.
        """
        cfg = self.cfg
        self._check_dims(train)
        if cfg.scheduler == "onecycle" and cfg.warmup_epochs:
            print("[tsfm] OneCycleLR has its own warmup -- ignoring warmup_epochs.")
            cfg = replace(cfg, warmup_epochs=0)
            self.cfg = cfg
        train_loader = make_dataloader(
            train,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            drop_last=len(train) > cfg.batch_size,   # keep BatchNorm happy
            persistent_workers=cfg.persistent_workers,
        )
        eval_sets: Dict[str, TSDataset] = {}
        if cfg.eval_on_train:
            eval_sets["train"] = train
        eval_sets["holdout"] = holdout
        if oot is not None:
            eval_sets["oot"] = oot
        if extra_eval:
            eval_sets.update(extra_eval)
        eval_loaders = {k: self._as_loader(v) for k, v in eval_sets.items()}

        scheduler, sched_mode = self._build_scheduler(max(len(train_loader), 1))
        mode = METRIC_MODES.get(cfg.monitor, cfg.monitor_mode)
        sign = 1.0 if mode == "min" else -1.0
        self.best_score_, self.best_epoch_, bad_epochs = math.inf, -1, 0

        for epoch in range(cfg.epochs):
            self._warmup_lr(epoch)
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, scheduler, sched_mode)

            row: Dict[str, Any] = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "seconds": None,
            }
            for name, loader in eval_loaders.items():
                for k, v in self.evaluate(loader).items():
                    row[f"{name}_{k}"] = v
            row["seconds"] = time.time() - t0
            self.history.append(row)

            score = sign * row[f"holdout_{cfg.monitor}"]
            improved = score < self.best_score_ - 1e-12
            if improved:
                self.best_score_, self.best_epoch_, bad_epochs = score, epoch, 0
                self.best_state_ = {
                    k: v.detach().cpu().clone() for k, v in self.model_raw.state_dict().items()
                }
                if cfg.ckpt_path:
                    torch.save(
                        {
                            "state_dict": self.best_state_,
                            "epoch": epoch,
                            f"holdout_{cfg.monitor}": row[f"holdout_{cfg.monitor}"],
                        },
                        cfg.ckpt_path,
                    )
            else:
                bad_epochs += 1

            if cfg.log_every_epoch:
                print(self._format_row(row, improved))

            if scheduler is not None:
                if sched_mode == "epoch" and epoch >= cfg.warmup_epochs:
                    scheduler.step()
                elif sched_mode == "plateau":
                    scheduler.step(row[f"holdout_{cfg.monitor}"])

            if cfg.early_stopping_patience and bad_epochs >= cfg.early_stopping_patience:
                print(
                    f"[tsfm] early stopping at epoch {epoch} "
                    f"(no holdout {cfg.monitor} improvement for {bad_epochs} epochs)"
                )
                break

        if self.best_state_ is not None:
            self.model_raw.load_state_dict(self.best_state_)
            print(
                f"[tsfm] restored best model from epoch {self.best_epoch_} "
                f"(holdout {cfg.monitor} = {sign * self.best_score_:.4f})"
            )
        return self.history.to_frame()

    def load_checkpoint(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load weights saved by :meth:`fit` (or any compatible checkpoint)."""
        ckpt = torch.load(path or self.cfg.ckpt_path, map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model_raw.load_state_dict(state)
        self.model_raw.to(self.device)
        return ckpt if isinstance(ckpt, dict) else {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_row(row: Dict[str, Any], improved: bool) -> str:
        parts = [f"epoch {row['epoch']:3d}", f"loss {row['train_loss']:.4f}"]
        for split in ("train", "holdout", "oot"):
            if f"{split}_rmse" in row:
                parts.append(
                    f"{split}: rmse {row[f'{split}_rmse']:.4f} "
                    f"mae {row[f'{split}_mae']:.4f} "
                    f"r2 {row[f'{split}_r2']:.4f} "
                    f"nmad {row[f'{split}_nmad']:.4f}"
                )
        parts.append(f"{row['seconds']:.1f}s")
        return " | ".join(parts) + ("  *" if improved else "")

    def final_report(self, splits: Dict[str, TSDataset], per_target: bool = False) -> pd.DataFrame:
        """Metrics for the restored best model on every split."""
        rows = []
        for name, ds in splits.items():
            m = self.evaluate(ds, per_target=per_target)
            m["split"] = name
            m["n"] = len(ds)
            rows.append(m)
        cols = ["split", "n", "rmse", "mae", "r2", "nmad"]
        df = pd.DataFrame(rows)
        return df[cols + [c for c in df.columns if c not in cols]].set_index("split")
