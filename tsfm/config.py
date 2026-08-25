"""Configuration dataclasses for the time-series transformer regressor.

Every knob the user asked to be tunable lives here so that a run can be fully
described by a single (serialisable) config object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Sequence

__all__ = [
    "TokenizerConfig",
    "MissingConfig",
    "TransformerConfig",
    "SnapshotConfig",
    "HeadConfig",
    "ModelConfig",
    "TrainConfig",
]


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
@dataclass
class TokenizerConfig:
    """Conv1d patch tokenizer.

    The input is ``(B, C, T)``: ``C`` features (= conv input channels), ``T``
    time steps.  A strided convolution slides along the time axis and emits
    ``N`` patch tokens per feature.

    Parameters
    ----------
    in_channels:
        Number of input time-series features ``C``.
    conv_channels:
        Width of the per-feature local extractor (the conv ``out_channels``
        *per feature*).  The grouped conv therefore has
        ``out_channels = in_channels * conv_channels``.
    kernel_size, stride, dilation:
        Patch geometry along the time axis.  ``N = floor((T + 2*pad - dil*(k-1)
        - 1) / stride) + 1``.
    n_layers:
        Number of stacked (strided) conv layers.  ``>1`` gives a deeper local
        feature extractor; note every layer applies ``stride`` again.
    d_model:
        Token width fed to the transformer.  A final grouped 1x1 conv projects
        ``conv_channels -> d_model`` per feature.
    per_feature:
        ``True``  -> grouped conv (``groups=C``): every feature keeps its own
                     private filters, so tokens stay attributable to a feature
                     and feature-axis attention is meaningful (default, and
                     what the architecture in the spec asks for).
        ``False`` -> a single shared conv that mixes features.  Tokens are then
                     *not* per-feature; only ``attn_mode='time'`` makes sense.
    causal:
        If ``True`` pad only on the left so a token never sees the future.
    pad_to_cover:
        Pad the sequence so the **last** time step is inside some patch window.
        Without it a geometry like ``T=24, kernel=5, stride=2`` silently drops
        the final month -- usually the most predictive one.  Leave it on unless
        you deliberately want to truncate.
    padding:
        Extra symmetric padding applied *before* ``pad_to_cover``.
    """

    in_channels: int
    conv_channels: int = 32
    kernel_size: int = 5
    stride: int = 2
    dilation: int = 1
    n_layers: int = 1
    d_model: int = 64
    per_feature: bool = True
    causal: bool = False
    pad_to_cover: bool = True
    padding: int = 0
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0
    norm: Literal["batch", "layer", "none"] = "batch"


# --------------------------------------------------------------------------- #
# Missing-value handling
# --------------------------------------------------------------------------- #
@dataclass
class MissingConfig:
    """How the two kinds of missingness are handled.

    * **Random missing** -> the value is replaced by ``0`` *after* scaling
      (i.e. the training mean when a standard scaler is used) and is *not*
      given an embedding.  Optionally an observed-fraction channel is appended
      so the conv can still tell an imputed zero from a true zero.
    * **Systematic missing** (the time step does not exist for this record --
      e.g. a 5-month-old card in a 24-month window) -> a learned *missing
      embedding* is combined with the patch token.

    Parameters
    ----------
    combine:
        ``"add"``    -> ``token + missing_emb``   (requires ``d_missing == d_model``;
                        it is forced to ``d_model`` automatically).
        ``"concat"`` -> ``Linear([token ; missing_emb]) -> d_model``.
        ``"none"``   -> disable the missing embedding entirely (ablation).
    d_missing:
        Width of the missing embedding.  Ignored (set to ``d_model``) when
        ``combine == "add"``.
    soft:
        If ``True`` the embedding is scaled by the *fraction* of systematically
        missing time steps inside the patch (continuous, differentiable, and it
        handles patches that straddle the boundary).  If ``False`` a hard
        binary indicator (fraction > ``hard_threshold``) is used.
    per_feature_embedding:
        Give every input feature its own missing vector (features can go
        missing for different reasons).  Otherwise one shared vector is used.
    add_observed_channel:
        Append the per-time-step observed mask as extra conv input channels so
        the local extractor also sees *random* missingness.  Doubles the
        tokenizer input channels.
    mask_fully_missing_tokens:
        Exclude tokens whose patch is 100% systematically missing from
        attention (``key_padding_mask``) and from pooling.  The missing
        embedding still marks them for the readout.
    """

    combine: Literal["add", "concat", "none"] = "add"
    d_missing: int = 16
    soft: bool = True
    hard_threshold: float = 0.5
    per_feature_embedding: bool = True
    add_observed_channel: bool = True
    mask_fully_missing_tokens: bool = True


# --------------------------------------------------------------------------- #
# Transformer encoder
# --------------------------------------------------------------------------- #
@dataclass
class TransformerConfig:
    """Encoder over the ``(C, N)`` token grid.

    ``attn_mode``
        ``"feature"``    -- attention **across features** at each patch
                            position (the mode the spec asks for: the Conv1d
                            already models the temporal pattern).  Sequence
                            length ``C``, batched over ``B*N``.
        ``"time"``       -- attention across patches within each feature.
        ``"factorized"`` -- alternate feature-attention and patch-attention
                            blocks (axial attention).
        ``"full"``       -- flatten to ``C*N`` tokens and attend over
                            everything (``O((C*N)^2)`` -- expensive).
    """

    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: Literal["gelu", "relu"] = "gelu"
    pre_norm: bool = True
    attn_mode: Literal["feature", "time", "factorized", "full"] = "feature"
    feature_embedding: bool = True   # learned per-feature identity vector
    patch_embedding: bool = True     # learned positional vector over patches
    pool: Literal["attention", "mean", "max", "flatten_feature"] = "attention"


# --------------------------------------------------------------------------- #
# Snapshot (static) branch
# --------------------------------------------------------------------------- #
@dataclass
class SnapshotConfig:
    """Static / snapshot variables that are constant over time.

    ``cat_cardinalities`` counts must already include the slot reserved for
    "missing/unknown" (``0``); :func:`tsfm.data.prepare_datasets` guarantees
    that encoding.
    """

    cat_cardinalities: List[int] = field(default_factory=list)
    cat_emb_dims: Optional[List[int]] = None      # default: min(16, ceil(card/2))
    n_numeric: int = 0
    numeric_missing_indicator: bool = True
    hidden_dims: Sequence[int] = (64,)
    d_out: int = 32
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"


# --------------------------------------------------------------------------- #
# Fusion head
# --------------------------------------------------------------------------- #
@dataclass
class HeadConfig:
    """Late-fusion MLP on ``[ts_embedding ; snapshot_embedding]``."""

    hidden_dims: Sequence[int] = (128, 64)
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    n_targets: int = 1
    output_activation: Literal["none", "softplus", "sigmoid"] = "none"


@dataclass
class ModelConfig:
    """Full model spec.  ``seq_len`` (T) is needed up front because the number
    of patch tokens -- and therefore the patch positional embedding -- depends
    on it."""

    tokenizer: TokenizerConfig
    seq_len: int
    missing: MissingConfig = field(default_factory=MissingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    head: HeadConfig = field(default_factory=HeadConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation / logging.

    Model selection: lowest **holdout RMSE**; OOT is only ever *reported*.
    """

    epochs: int = 50
    batch_size: int = 256
    eval_batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    loss: Literal["mse", "huber", "mae"] = "mse"
    huber_delta: float = 1.0
    grad_clip: Optional[float] = 1.0
    scheduler: Literal["none", "cosine", "plateau", "onecycle"] = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    amp: bool = True                       # mixed precision (GPU only)
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    early_stopping_patience: Optional[int] = 10
    monitor: str = "rmse"                  # metric on holdout used for selection
    monitor_mode: Literal["min", "max"] = "min"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 42
    device: str = "cuda"
    log_every_epoch: bool = True
    eval_on_train: bool = True             # full-pass train metrics each epoch
    ckpt_path: str = "best_model.pt"
    compile: bool = False                  # torch.compile (PyTorch >= 2.0)
