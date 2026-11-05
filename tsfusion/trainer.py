"""Training loop with per-epoch metrics, checkpointing and early stopping.

One GPU by default. Launched with ``torchrun --nproc_per_node=N`` it trains with
DistributedDataParallel on N GPUs (see :mod:`tsfusion.distributed`): each GPU processes a
shard of every global batch, gradients are averaged, evaluation is sharded and gathered,
and only rank 0 writes logs / CSV / checkpoints.

Per epoch
  1. one pass over the training data (mixed precision on GPU, gradient clipping)
  2. eval-mode metrics (RMSE / MAE / R2 / NMAD on the ORIGINAL scale) on the training
     data (or a fixed random subset) and on the validation data
  3. LR scheduler step, one CSV row + log line, checkpoint ``epoch_XXX.pt``
  4. if validation RMSE improved: ``best_model.pt``; otherwise patience counter += 1
  5. stop when the patience is exhausted; the best weights are restored at the end
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import ModelConfig, TrainConfig
from .data import InMemoryLoader, Preprocessor, ProcessedData
from .distributed import (all_gather_equal, all_reduce_sum, barrier, broadcast_object,
                          init_distributed)
from .losses import build_loss, tweedie_deviance
from .metrics import regression_metrics
from .model import TSFusionTransformer, count_parameters


# =============================================================================
# Utilities
# =============================================================================
def _grad_scaler(enabled: bool):
    """Loss scaling for fp16. ``torch.amp.GradScaler`` exists from PyTorch 2.3; older
    versions (e.g. older Databricks ML runtimes) only have ``torch.cuda.amp.GradScaler``."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


NONFINITE_CHECK_EVERY = 50      # steps between checks of the running loss for NaN / inf


def fits_on_gpu(need_bytes: int, free_bytes: int, fraction: float) -> bool:
    """``data_on_device="auto"``: copy the data to the GPU only if it takes at most
    ``fraction`` of the currently free GPU memory (the rest is left for activations)."""
    return need_bytes <= fraction * free_bytes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """``cuda:<i>`` if available, else CPU (with a warning)."""
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available - falling back to CPU (slow).")
            return torch.device("cpu")
        idx = dev.index if dev.index is not None else 0
        if idx >= torch.cuda.device_count():
            raise ValueError(f"{name} requested but only {torch.cuda.device_count()} GPU(s) visible")
        torch.cuda.set_device(idx)
        return torch.device("cuda", idx)
    return dev


def _make_logger(output_dir: str, verbose: bool, enabled: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"tsfusion.{os.path.abspath(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):          # avoid duplicate handlers when re-running cells
        logger.removeHandler(h)
        h.close()
    if not enabled:                          # non-zero DDP ranks stay silent
        logger.addHandler(logging.NullHandler())
        return logger
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


_NO_DECAY_PARAMS = {"var_emb", "pos_emb", "cls_global", "cls_var", "cls_time", "cls"}


def _param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on biases, LayerNorm weights, position / variable / state embeddings
    and CLS tokens; decay on all projection matrices."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if p.ndim < 2 or leaf in _NO_DECAY_PARAMS or name.endswith("state_emb.weight"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _to_builtin(obj):
    """Make metrics JSON / ``torch.load(weights_only=True)`` friendly."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    """Train :class:`TSFusionTransformer` on one GPU, or on several with DDP.

    Example::

        trainer = Trainer(model_cfg, train_cfg, preprocessor)
        history = trainer.fit(train_data, val_data)     # pandas DataFrame, one row / epoch
        y_hat = trainer.predict(val_data)                 # original scale

    Under ``torchrun`` every process runs the same script; ``fit``, ``evaluate``,
    ``predict`` and ``resume`` are collective (call them on all ranks). Their results are
    identical on every rank; files are written by rank 0 only.
    """

    MONITOR = "val_rmse"

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 preprocessor: Preprocessor, model: Optional[TSFusionTransformer] = None):
        self.model_cfg, self.cfg, self.prep = model_cfg, train_cfg, preprocessor
        self.loss_fn = build_loss(train_cfg)
        tmode = preprocessor.target_scaler.mode
        if tmode not in self.loss_fn.compatible_target_modes:
            raise ValueError(f"loss {train_cfg.loss!r} is not compatible with target mode "
                             f"{tmode!r}; use one of {self.loss_fn.compatible_target_modes}")
        if train_cfg.loss == "tweedie" and train_cfg.tweedie_power >= 2.0 \
                and preprocessor.target_scaler.scale <= 0:
            raise ValueError("Tweedie with p >= 2 needs a strictly positive target")

        use_cuda = torch.device(train_cfg.device).type == "cuda" and torch.cuda.is_available()
        self.rank, self.world_size, local_rank = init_distributed(use_cuda)
        self.distributed = self.world_size > 1
        self.is_main = self.rank == 0

        set_seed(train_cfg.seed)                 # identical initial weights on every rank
        self.device = resolve_device(f"cuda:{local_rank}" if self.distributed and use_cuda
                                     else train_cfg.device)
        self.model = (model or TSFusionTransformer(model_cfg)).to(self.device)
        train_module: nn.Module = self.model
        if self.distributed:
            # the model's only buffer is a constant mask, so no buffer syncing is needed
            sync_kw = ({"forward_sync_buffers": False}
                       if "forward_sync_buffers" in inspect.signature(DDP.__init__).parameters
                       else {"broadcast_buffers": False})
            train_module = DDP(self.model, **sync_kw,
                               device_ids=[self.device.index] if self.device.type == "cuda" else None)
            set_seed(train_cfg.seed + self.rank)     # different dropout masks per GPU
        compile_ = torch.compile if train_cfg.compile_model else (lambda m: m)
        self._train_fwd = compile_(train_module)      # DDP-wrapped: gradients are all-reduced
        self._eval_fwd = compile_(self.model) if self.distributed else self._train_fwd

        # mixed precision: bf16 if the GPU supports it (no loss scaling needed), else fp16
        self.use_amp = train_cfg.use_amp and self.device.type == "cuda"
        if train_cfg.amp_dtype == "bf16" or (train_cfg.amp_dtype == "auto" and self.use_amp
                                             and torch.cuda.is_bf16_supported()):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = _grad_scaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.optimizer = torch.optim.AdamW(_param_groups(self.model, train_cfg.weight_decay),
                                           lr=train_cfg.lr)
        self.scheduler = None            # built in fit() once steps/epoch is known
        self.global_step = 0
        self.best_metric = math.inf
        self.best_epoch = -1
        self.history: list = []
        self._best_state: Optional[Dict[str, torch.Tensor]] = None   # in-memory best weights
        self._resume_scheduler_state = None

        os.makedirs(train_cfg.output_dir, exist_ok=True)
        self.logger = _make_logger(train_cfg.output_dir, train_cfg.verbose, enabled=self.is_main)

    # ------------------------------------------------------------------ helpers
    def _data_on_gpu(self, *datasets: ProcessedData) -> bool:
        """Copy these datasets to the GPU once (True) or stream them per batch (False)."""
        mode = self.cfg.data_on_device
        if self.device.type != "cuda":
            return False
        if mode != "auto":
            return bool(mode)
        free, _ = torch.cuda.mem_get_info(self.device)
        return fits_on_gpu(sum(d.nbytes() for d in datasets), free, self.cfg.gpu_data_fraction)

    def _prepare(self, *datasets: ProcessedData):
        """Place datasets according to ``data_on_device``; returns them in the same order."""
        on_gpu = self._data_on_gpu(*datasets)
        placed = [d.to(self.device) if on_gpu else d for d in datasets]
        self._data_where = "GPU" if on_gpu else (
            "CPU, streamed to the GPU per batch" if self.device.type == "cuda" else "CPU")
        return placed[0] if len(placed) == 1 else placed

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward(self, batch: Dict[str, torch.Tensor], train: bool = False) -> torch.Tensor:
        fwd = self._train_fwd if train else self._eval_fwd
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.use_amp):
            return fwd(batch["x"], batch["state"], batch.get("snapshot"))

    def _build_scheduler(self, steps_per_epoch: int):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            total = max(1, cfg.max_epochs * steps_per_epoch)
            warm = cfg.warmup_steps
            min_ratio = cfg.min_lr / cfg.lr

            def lr_lambda(step):
                if step < warm:
                    return (step + 1) / max(1, warm)
                prog = min(1.0, (step - warm) / max(1, total - warm))
                return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        if cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=cfg.plateau_factor,
                patience=cfg.plateau_patience, min_lr=cfg.min_lr)
        return None

    def _manual_warmup(self) -> None:
        """Linear warm-up for the 'plateau' / 'none' schedulers (cosine has it built in)."""
        if self.cfg.scheduler == "cosine" or self.global_step >= self.cfg.warmup_steps:
            return
        scale = (self.global_step + 1) / self.cfg.warmup_steps
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.lr * scale

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------ train / eval
    def train_one_epoch(self, loader: InMemoryLoader, epoch: int) -> float:
        self.model.train()
        loader.set_epoch(epoch)
        # the running loss stays on the device: a .item() per step would make the CPU wait
        # for every step and stop it preparing the next batch in the meantime
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        count = 0
        n_steps = len(loader)
        for i, batch in enumerate(loader):
            batch = self._batch_to_device(batch)
            self._manual_warmup()
            out = self._forward(batch, train=True)
            loss = self.loss_fn(out, batch["y"])
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.cfg.scheduler == "cosine":
                self.scheduler.step()
            self.global_step += 1
            n = batch["y"].shape[0]
            total += loss.detach().double() * n
            count += n
            log_now = self.cfg.log_every_n_steps and (i + 1) % self.cfg.log_every_n_steps == 0
            if log_now or (i + 1) % NONFINITE_CHECK_EVERY == 0 or i + 1 == n_steps:
                running = total.item() / count             # one sync every few steps
                if not math.isfinite(running):
                    raise FloatingPointError(
                        f"non-finite training loss in epoch {epoch} (by step {i + 1}); try a "
                        "lower lr, fp32 (use_amp=False) or check the inputs")
                if log_now:
                    self.logger.info(f"  epoch {epoch} step {i + 1}/{n_steps} "
                                     f"loss {running:.5f} lr {self._current_lr():.2e}")
        total_f, count_f = total.item(), float(count)
        if self.distributed:                 # running loss over all GPUs (logging only)
            t = all_reduce_sum(torch.tensor([total_f, count_f], dtype=torch.float64,
                                            device=self.device))
            total_f, count_f = t[0].item(), t[1].item()
        return total_f / max(count_f, 1)

    @torch.no_grad()
    def _predict_link(self, data: ProcessedData, indices: Optional[np.ndarray] = None
                      ) -> Tuple[torch.Tensor, float]:
        """Raw network output (link scale) for all requested records and the mean loss (if
        targets are present). With DDP each rank predicts one contiguous shard; the shards
        are gathered so every rank returns the full, identically ordered result."""
        self.model.eval()
        idx = np.arange(len(data)) if indices is None else np.asarray(indices)
        n = len(idx)
        mine = idx
        if self.distributed:
            chunk = math.ceil(n / self.world_size)
            padded = np.concatenate([idx, np.full(chunk * self.world_size - n, idx[-1])])
            mine = padded[self.rank * chunk:(self.rank + 1) * chunk]
        loader = InMemoryLoader(data, self.cfg.eval_batch_size, shuffle=False, indices=mine,
                                device=self.device)
        out = torch.cat([self._forward(self._batch_to_device(b)).float() for b in loader])
        if self.distributed:
            out = all_gather_equal(out)[:n]
        loss = float("nan")
        if data.y is not None:
            y = data.y[torch.as_tensor(idx, device=data.y.device)].to(out.device)
            loss = self.loss_fn(out, y).item()
        return out, loss

    def _to_original_scale(self, link_out: torch.Tensor) -> np.ndarray:
        z = self.loss_fn.inverse_link(link_out).cpu().numpy()
        return self.prep.target_scaler.inverse_transform(z)

    def evaluate(self, data: ProcessedData, indices: Optional[np.ndarray] = None
                 ) -> Tuple[Dict[str, float], np.ndarray]:
        """Loss (transformed space) + RMSE / MAE / R2 / NMAD on the ORIGINAL scale."""
        if data.y_raw is None:
            raise ValueError("evaluate() needs targets")
        link, loss = self._predict_link(data, indices)
        pred = self._to_original_scale(link)
        y = data.y_raw if indices is None else data.y_raw[indices]
        metrics = {"loss": loss, **regression_metrics(y, pred, self.cfg.nmad_relative)}
        if self.cfg.loss == "tweedie":
            # the loss value carries a p-dependent constant; the deviance (0 = perfect) is
            # comparable across runs that use different p
            metrics["tweedie_dev"] = tweedie_deviance(y, pred, self.cfg.tweedie_power) \
                if np.isfinite(pred).all() else float("nan")
        return metrics, pred

    def predict(self, data: ProcessedData) -> np.ndarray:
        """Predictions on the original target scale."""
        data = self._prepare(data)
        link, _ = self._predict_link(data)
        return self._to_original_scale(link)

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint(self, epoch: int, metrics: Dict[str, float], with_optimizer: bool) -> dict:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "model_config": self.model_cfg.to_dict(),
            "train_config": self.cfg.to_dict(),
            "preprocessor": self.prep.state_dict(),
            "metrics": _to_builtin(metrics),
            "best_val_rmse": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
        }
        if with_optimizer:
            ckpt["optimizer_state"] = self.optimizer.state_dict()
            ckpt["scaler_state"] = self.scaler.state_dict()
            if self.scheduler is not None:
                ckpt["scheduler_state"] = self.scheduler.state_dict()
        return ckpt

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float],
                        with_optimizer: bool = True) -> None:
        if not self.is_main:                 # weights are identical on all ranks
            return
        tmp = path + ".tmp"
        torch.save(self._checkpoint(epoch, metrics, with_optimizer), tmp)
        os.replace(tmp, path)          # atomic: never leaves a half-written checkpoint

    def resume(self, path: str) -> int:
        """Restore model / optimizer / scheduler from a per-epoch checkpoint; returns the
        epoch to start from. Call before ``fit(..., start_epoch=...)``."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self._resume_scheduler_state = ckpt.get("scheduler_state")
        self.scheduler = None                          # rebuilt (and restored) in fit()
        self.global_step = ckpt["global_step"]
        self.best_metric, self.best_epoch = ckpt["best_val_rmse"], ckpt["best_epoch"]
        out, epoch = self.cfg.output_dir, ckpt["epoch"]
        # read everything first (all ranks), then let rank 0 rewrite the files
        self._best_state, best = None, None
        best_src = os.path.join(out, f"epoch_{self.best_epoch:03d}.pt")
        if self.best_epoch >= 1 and os.path.exists(best_src):
            best = torch.load(best_src, map_location="cpu", weights_only=True)
            self._best_state = best["model_state"]
        hist_path = os.path.join(out, "metrics.csv")
        hist = None
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            hist = hist[hist["epoch"] <= epoch]
            self.history = hist.to_dict("records")
        barrier()
        if self.is_main:
            # forget everything an abandoned run wrote after the resumed epoch
            for f in os.listdir(out):
                m = re.fullmatch(r"epoch_(\d+)\.pt", f)
                if m and int(m.group(1)) > epoch:
                    os.remove(os.path.join(out, f))
            if best is not None:
                for k in ("optimizer_state", "scaler_state", "scheduler_state"):
                    best.pop(k, None)
                torch.save(best, os.path.join(out, "best_model.pt"))  # consistent with resume
            if hist is not None:
                hist.to_csv(hist_path, index=False)    # drop rows after the resumed epoch
        barrier()
        return epoch + 1

    def _start_fresh(self) -> None:
        """Reset run state and clear outputs of a previous run in ``output_dir``."""
        self.history, self.best_metric, self.best_epoch, self._best_state = [], math.inf, -1, None
        out = self.cfg.output_dir
        stale = [f for f in os.listdir(out)
                 if re.fullmatch(r"epoch_\d+\.pt|best_model\.pt|metrics\.csv", f)]
        if stale and not self.cfg.overwrite_output_dir:
            raise FileExistsError(f"{out} already contains {len(stale)} checkpoint/metric files; "
                                  "use another output_dir, resume(), or overwrite_output_dir=True")
        if self.is_main:
            for f in stale:
                os.remove(os.path.join(out, f))
            if stale:
                self.logger.info(f"removed {len(stale)} files of a previous run in {out}")
        barrier()

    # ------------------------------------------------------------------ main loop
    def fit(self, train_data: ProcessedData, val_data: ProcessedData,
            start_epoch: int = 1) -> pd.DataFrame:
        cfg = self.cfg
        if train_data.y is None or val_data.y is None:
            raise ValueError("train and validation data need targets")
        train_data, val_data = self._prepare(train_data, val_data)
        train_loader = InMemoryLoader(train_data, cfg.batch_size, shuffle=True, seed=cfg.seed,
                                      rank=self.rank, world_size=self.world_size,
                                      device=self.device)
        if start_epoch == 1:
            self._start_fresh()
        if self.scheduler is None:
            saved_lrs = [g["lr"] for g in self.optimizer.param_groups]
            self.scheduler = self._build_scheduler(len(train_loader))
            if self._resume_scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(self._resume_scheduler_state)
                # LambdaLR's constructor overwrote the lr with lr * lambda(0): restore it
                for g, lr in zip(self.optimizer.param_groups, saved_lrs):
                    g["lr"] = lr
                self._resume_scheduler_state = None

        train_eval_idx = None
        if cfg.train_eval_max_samples and cfg.train_eval_max_samples < len(train_data):
            rng = np.random.default_rng(cfg.seed)
            train_eval_idx = np.sort(rng.choice(len(train_data), cfg.train_eval_max_samples,
                                                replace=False))

        if self.is_main:
            with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
                json.dump({"model": self.model_cfg.to_dict(), "train": cfg.to_dict(),
                           "n_gpus": self.world_size,
                           "target_scaler": self.prep.target_scaler.state_dict()}, f, indent=2)
        csv_path = os.path.join(cfg.output_dir, "metrics.csv")

        amp = str(self.amp_dtype).split('.')[-1] if self.use_amp else "off"
        gpus = (f"gpus={self.world_size} (DDP, global batch {cfg.batch_size * self.world_size}) "
                if self.distributed else "")
        self.logger.info(f"device={self.device} {gpus}amp={amp} "
                         f"params={count_parameters(self.model):,} train={len(train_data):,} "
                         f"val={len(val_data):,} loss={cfg.loss}"
                         + (f"(p={cfg.tweedie_power})" if cfg.loss == "tweedie" else "")
                         + f" fusion={self.model_cfg.missing_fusion}")
        size = train_data.nbytes() + val_data.nbytes()
        size_txt = f"{size / 2**30:.2f} GiB" if size >= 2**30 else f"{size / 2**20:.1f} MiB"
        self.logger.info(f"data: {size_txt} (train + val), kept on {self._data_where}")
        bad_epochs = 0
        for row in self.history:      # non-empty only after resume(): rebuild the counter
            bad_epochs = 0 if row["epoch"] == self.best_epoch else bad_epochs + 1

        for epoch in range(start_epoch, cfg.max_epochs + 1):
            t0 = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_loss = self.train_one_epoch(train_loader, epoch)
            train_m, _ = self.evaluate(train_data, train_eval_idx)
            val_m, _ = self.evaluate(val_data)
            lr = self._current_lr()
            if cfg.scheduler == "plateau" and self.global_step >= cfg.warmup_steps:
                v = val_m["rmse"]
                self.scheduler.step(v if math.isfinite(v) else math.inf)

            improved = math.isfinite(val_m["rmse"]) and \
                val_m["rmse"] < self.best_metric - cfg.early_stopping_min_delta
            improved = bool(broadcast_object(improved))   # same decision on every GPU
            if improved:
                self.best_metric, self.best_epoch, bad_epochs = val_m["rmse"], epoch, 0
                self._best_state = {k: v.detach().to("cpu", copy=True)
                                    for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1

            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                   **{f"train_eval_{k}" if k == "loss" else f"train_{k}": v
                      for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()},
                   "best_epoch": self.best_epoch, "is_best": improved,
                   "epoch_time_s": time.time() - t0}
            if self.device.type == "cuda":
                row["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
            row = _to_builtin(row)
            self.history.append(row)
            if self.is_main:
                self._append_csv(csv_path, row)

            if cfg.save_every_epoch:
                self.save_checkpoint(os.path.join(cfg.output_dir, f"epoch_{epoch:03d}.pt"),
                                     epoch, row, with_optimizer=cfg.save_optimizer_state)
            if improved:
                self.save_checkpoint(os.path.join(cfg.output_dir, "best_model.pt"), epoch, row,
                                     with_optimizer=False)

            self.logger.info(
                f"epoch {epoch:3d} | lr {lr:.2e} | train loss {train_loss:.4f} "
                f"rmse {train_m['rmse']:.4f} mae {train_m['mae']:.4f} r2 {train_m['r2']:.4f} "
                f"nmad {train_m['nmad']:.4f} | val loss {val_m['loss']:.4f} "
                f"rmse {val_m['rmse']:.4f} mae {val_m['mae']:.4f} r2 {val_m['r2']:.4f} "
                f"nmad {val_m['nmad']:.4f} | {'*best*' if improved else f'{bad_epochs} bad'} "
                f"| {row['epoch_time_s']:.1f}s")

            if bad_epochs >= cfg.early_stopping_patience:
                self.logger.info(f"early stopping at epoch {epoch}: no improvement in val RMSE "
                                 f"for {bad_epochs} epochs (best {self.best_metric:.4f} "
                                 f"at epoch {self.best_epoch})")
                break

        barrier()
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            self.logger.info(f"restored best model from epoch {self.best_epoch} "
                             f"(val RMSE {self.best_metric:.4f})")
        else:
            warnings.warn("validation RMSE was never finite - keeping the last weights")
        return pd.DataFrame(self.history)

    @staticmethod
    def _append_csv(path: str, row: dict) -> None:
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)


# =============================================================================
# Loading for inference
# =============================================================================
def load_checkpoint(path: str, device: str = "cpu"
                    ) -> Tuple[TSFusionTransformer, Preprocessor, TrainConfig, dict]:
    """Rebuild model + preprocessor from a checkpoint (``best_model.pt`` or ``epoch_XXX.pt``)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model_cfg = ModelConfig(**ckpt["model_config"])
    train_cfg = TrainConfig(**ckpt["train_config"])
    model = TSFusionTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    prep = Preprocessor.from_state_dict(ckpt["preprocessor"])
    return model, prep, train_cfg, ckpt


@torch.no_grad()
def predict_with(model: TSFusionTransformer, prep: Preprocessor, train_cfg: TrainConfig,
                 data: ProcessedData, batch_size: int = 4096) -> np.ndarray:
    """Original-scale predictions from a model restored with :func:`load_checkpoint`."""
    loss_fn = build_loss(train_cfg)
    device = next(model.parameters()).device
    model.eval()
    outs = []
    for batch in InMemoryLoader(data, batch_size, device=device):
        outs.append(model(batch["x"], batch["state"], batch.get("snapshot")).float())
    z = loss_fn.inverse_link(torch.cat(outs)).cpu().numpy()
    return prep.target_scaler.inverse_transform(z)
