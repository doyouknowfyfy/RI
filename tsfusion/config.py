"""Configuration dataclasses for the model and the training loop.

Both configs are plain dataclasses so they can be serialised into checkpoints
with ``dataclasses.asdict`` and restored with ``Config(**d)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

# How missing / special values are combined with regular values.
#   "late"   : the design requested in the spec. Three CLS tokens are concatenated
#              before the head: [CLS regular-values | CLS missing/special | CLS snapshot].
#   "early"  : missing/special embeddings are injected into the *same* tokens as the
#              regular values (value-aware tokens). Two CLS tokens: [CLS ts | CLS snapshot].
#   "hybrid" : both of the above (value-aware tokens + a separate missing/special CLS).
MISSING_FUSION_MODES = ("late", "early", "hybrid")


@dataclass
class ModelConfig:
    # ---- data shape -------------------------------------------------------------
    n_vars: int                                   # V: number of time-series variables
    n_steps: int = 11                             # T: number of time steps
    n_special_codes: int = 0                      # K: special values are coded 1..K
    snapshot_group_sizes: List[int] = field(default_factory=list)
    # ^ one entry per snapshot variable = length of its PLE vector (use 1 for a raw
    #   numeric variable). Empty list -> no snapshot branch.

    # ---- tokenizer (sliding patches) -------------------------------------------
    patch_len: int = 3
    patch_stride: int = 1
    patch_pad_end: bool = True                    # replicate-pad so the last step is covered
    patch_include_mask: bool = True               # late mode: feed "is regular" bits with the
                                                  # values so an imputed 0 is distinguishable
                                                  # from a real 0.
    state_emb_dim: int = 16                       # size of the missing/special embedding

    # ---- transformer -------------------------------------------------------------
    d_model: int = 64
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.0
    n_ts_layers: int = 2                          # factorized (variable -> time) blocks
    n_special_layers: int = 1                     # blocks in the missing/special branch
    n_snapshot_layers: int = 1                    # blocks in the snapshot branch
    mask_empty_patches: bool = True               # late mode: patches without any regular
                                                  # value are hidden from attention
    mask_regular_patches: bool = True             # special branch: patches without any
                                                  # missing/special value are hidden
    gradient_checkpointing: bool = False          # trade compute for memory (many variables)

    missing_fusion: str = "late"

    # ---- head --------------------------------------------------------------------
    head_hidden: int = 128
    head_dropout: float = 0.1

    def __post_init__(self) -> None:
        self.snapshot_group_sizes = [int(s) for s in self.snapshot_group_sizes]
        if self.missing_fusion not in MISSING_FUSION_MODES:
            raise ValueError(f"missing_fusion must be one of {MISSING_FUSION_MODES}, "
                             f"got {self.missing_fusion!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_vars < 1 or self.n_steps < 1:
            raise ValueError("n_vars and n_steps must be >= 1")
        if self.patch_len < 1 or self.patch_stride < 1:
            raise ValueError("patch_len and patch_stride must be >= 1")
        if self.patch_stride > self.patch_len:
            raise ValueError("patch_stride > patch_len would skip time steps")
        if self.patch_len > self.n_steps:
            raise ValueError(f"patch_len ({self.patch_len}) > n_steps ({self.n_steps})")
        if any(s < 1 for s in self.snapshot_group_sizes):
            raise ValueError("every snapshot group size must be >= 1")
        if self.n_special_codes < 0:
            raise ValueError("n_special_codes must be >= 0")

    @property
    def n_states(self) -> int:
        """0 = regular, 1 = missing, 2..K+1 = special code 1..K."""
        return 2 + self.n_special_codes

    def to_dict(self) -> dict:
        return asdict(self)


LOSSES = ("mse", "tweedie")
SCHEDULERS = ("plateau", "cosine", "none")


@dataclass
class TrainConfig:
    # ---- objective ---------------------------------------------------------------
    loss: str = "mse"                             # "mse" | "tweedie"
    tweedie_power: float = 1.5                    # 1 <= p <= 3 (p=2 is Gamma); tune on validation

    # ---- optimisation ------------------------------------------------------------
    batch_size: int = 256
    eval_batch_size: int = 1024
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200                       # linear LR warm-up (optimizer steps)
    scheduler: str = "plateau"                    # "plateau" | "cosine" | "none"
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    min_lr: float = 1e-6
    grad_clip: Optional[float] = 1.0

    # ---- early stopping / model selection (monitored metric: validation RMSE) ----
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # ---- hardware: one GPU ---------------------------------------------------------
    device: str = "cuda:0"                        # falls back to CPU (with a warning)
    use_amp: bool = True                          # mixed precision on the GPU
    amp_dtype: str = "auto"                       # "auto" | "bf16" | "fp16"
    data_on_device: bool = True                   # copy the whole dataset to GPU once
    compile_model: bool = False                   # torch.compile (PyTorch >= 2)

    # ---- bookkeeping -------------------------------------------------------------
    seed: int = 42
    output_dir: str = "runs/experiment"
    overwrite_output_dir: bool = True             # fresh run: delete old checkpoints/metrics
                                                  # in output_dir (False -> raise instead)
    save_every_epoch: bool = True
    save_optimizer_state: bool = True             # needed to resume training
    train_eval_max_samples: Optional[int] = None  # eval-mode metrics on (a subset of) train
    nmad_relative: bool = False                   # NMAD on relative instead of absolute errors
    log_every_n_steps: int = 0                    # 0 = only per-epoch logs
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.loss == "tweedie" and not (1.0 <= self.tweedie_power <= 3.0):
            raise ValueError("tweedie_power must be in [1, 3]")
        if self.amp_dtype not in ("auto", "bf16", "fp16"):
            raise ValueError("amp_dtype must be 'auto', 'bf16' or 'fp16'")

    def to_dict(self) -> dict:
        return asdict(self)
