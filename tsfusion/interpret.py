"""Interpretation tools for :class:`TSFusionTransformer` - built for the early-fusion
model (value-aware tokens); most functions also work for the late / hybrid modes.

Structure
---------
1. Recording         :func:`record_attention`, :func:`run_with_attention`
                     -> per-layer attention maps + intermediate embeddings for one batch
2. Attention         :func:`cls_rollout`         relevance of every (variable, patch) token
                                                 for the final time-series CLS
                     :func:`cls_attention_summary` what the CLS tokens attend to (one layer)
                     :func:`snapshot_cls_attention`, :func:`snapshot_rollout`
3. Missing/special   :func:`patch_state_category`, :func:`relevance_by_category`,
                     :func:`relevance_by_variable`
4. Embeddings        :func:`cell_embedding_map`, :func:`state_similarity`, :func:`pca_2d`
5. Behaviour         :func:`modality_ablation`, :func:`permutation_importance`, :func:`state_probe`
                     (change the input / embedding and measure the prediction - these are
                     about what the model *uses*, attention only shows where it *looks*)
6. Dataset pass      :func:`collect_interpretation` runs 1-4 over a dataset in batches

Token grid of the time-series encoder (R = V+1 rows, C = Np+1 columns)::

            col 0          col 1 .. Np
    row 0   [CLS global]   [CLS time_1] ... [CLS time_Np]
    row v   [CLS var_v]    patch(v,1)   ... patch(v,Np)

``AttentionMaps.var[l]``  (B, C, H, R, R): attention across variables inside column c.
``AttentionMaps.time[l]`` (B, R, H, C, C): attention across time inside row r.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .data import InMemoryLoader, Preprocessor, ProcessedData
from .losses import build_loss
from .metrics import rmse
from .model import SelfAttention, TSFusionTransformer


# =============================================================================
# 1. Recording
# =============================================================================
@dataclass
class AttentionMaps:
    var: List[torch.Tensor] = field(default_factory=list)          # (B, C, H, R, R) per layer
    time: List[torch.Tensor] = field(default_factory=list)         # (B, R, H, C, C) per layer
    snapshot: List[torch.Tensor] = field(default_factory=list)     # (B, H, S+1, S+1) per layer
    special_var: List[torch.Tensor] = field(default_factory=list)  # late / hybrid only
    special_time: List[torch.Tensor] = field(default_factory=list)


@contextmanager
def record_attention(model: TSFusionTransformer) -> Iterator[List[SelfAttention]]:
    """Inside the block every attention layer computes its weights explicitly and keeps
    them in ``.attn_weights``. The model is put in eval mode and restored afterwards."""
    mods = [m for m in model.modules() if isinstance(m, SelfAttention)]
    was_training = model.training
    model.eval()
    for m in mods:
        m.store_attn = True
    try:
        yield mods
    finally:
        for m in mods:
            m.store_attn, m.attn_weights = False, None
        model.train(was_training)


def _grid_maps(encoder, B: int, R: int, C: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    var, time = [], []
    for blk in encoder.blocks:
        H = blk.var_attn.n_heads
        var.append(blk.var_attn.attn_weights.view(B, C, H, R, R).cpu())
        time.append(blk.time_attn.attn_weights.view(B, R, H, C, C).cpu())
    return var, time


@torch.no_grad()
def run_with_attention(model: TSFusionTransformer, x: torch.Tensor, state: torch.Tensor,
                       snapshot: Optional[torch.Tensor] = None):
    """One forward pass that returns ``(link_output, cls_embeddings, details, maps)``.

    ``details`` holds ``state_patches``, ``tokens``, ``ts_grid`` and ``snapshot_tokens``
    (see :meth:`TSFusionTransformer.embed`); ``maps`` is an :class:`AttentionMaps`."""
    with record_attention(model):
        details: Dict[str, torch.Tensor] = {}
        emb = model.embed(x, state, snapshot, details=details)
        z = torch.cat(list(emb.values()), dim=-1)
        out = model.head(z.float())
        B, R, C = x.shape[0], model.cfg.n_vars + 1, model.n_patches + 1
        maps = AttentionMaps()
        maps.var, maps.time = _grid_maps(model.ts_encoder, B, R, C)
        if model.has_special_branch:
            maps.special_var, maps.special_time = _grid_maps(model.special_encoder, B, R, C)
        if model.has_snapshot:
            maps.snapshot = [blk.attn.attn_weights.cpu() for blk in model.snapshot_encoder.blocks]
    return out, emb, details, maps


# =============================================================================
# 2. Attention summaries
# =============================================================================
def cls_rollout(var: Sequence[torch.Tensor], time: Sequence[torch.Tensor],
                residual: float = 0.5) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema, 2020) adapted to factorized attention.

    Each attention step is approximated by ``residual * I + (1 - residual) * A`` (heads
    averaged, residual connection) and composed through all layers, backwards from the
    global CLS at (0, 0). Returns ``(B, R, C)``: how much of the final global CLS traces
    back to each input token. Every record's map sums to 1; ``[:, 1:, 1:]`` is the
    (variable x patch) part, the rest flows to the learned CLS tokens."""
    B, C, _, R, _ = var[0].shape
    r = torch.zeros(B, R, C)
    r[:, 0, 0] = 1.0
    eye_r, eye_c = torch.eye(R), torch.eye(C)
    for a_var, a_time in reversed(list(zip(var, time))):
        at = residual * eye_c + (1 - residual) * a_time.float().mean(2)     # (B, R, C, C)
        r = torch.einsum("brc,brcd->brd", r, at)                         # back through time
        av = residual * eye_r + (1 - residual) * a_var.float().mean(2)      # (B, C, R, R)
        r = torch.einsum("brc,bcrs->bsc", r, av)                         # back through variables
    return r


def cls_attention_summary(maps: AttentionMaps, layer: int = -1) -> Dict[str, torch.Tensor]:
    """Head-averaged attention of the CLS tokens in one layer.

    * ``global_to_var``  (B, V)     global CLS -> variable-CLS tokens (column 0)
    * ``global_to_time`` (B, Np)    global CLS -> time-CLS tokens (row 0)
    * ``timecls_to_var`` (B, V, Np) time-CLS of patch n -> each variable's patch at n
    * ``varcls_to_time`` (B, V, Np) variable-CLS of v -> each of v's patches
    """
    av, at = maps.var[layer].float(), maps.time[layer].float()
    return {
        "global_to_var": av[:, 0, :, 0, 1:].mean(1),
        "global_to_time": at[:, 0, :, 0, 1:].mean(1),
        "timecls_to_var": av[:, 1:, :, 0, 1:].mean(2).transpose(1, 2),
        "varcls_to_time": at[:, 1:, :, 0, 1:].mean(2),
    }


def snapshot_cls_attention(maps: AttentionMaps, layer: int = -1) -> torch.Tensor:
    """(B, H, S): attention of the snapshot CLS to each snapshot-variable token."""
    return maps.snapshot[layer][:, :, 0, 1:].float()


def snapshot_rollout(maps: AttentionMaps, residual: float = 0.5) -> torch.Tensor:
    """(B, S+1): rollout of the snapshot CLS over [CLS, snapshot tokens]; sums to 1."""
    a0 = maps.snapshot[0]
    B, L = a0.shape[0], a0.shape[-1]
    r = torch.zeros(B, L)
    r[:, 0] = 1.0
    eye = torch.eye(L)
    for a in reversed(maps.snapshot):
        r = torch.einsum("bl,blm->bm", r, residual * eye + (1 - residual) * a.float().mean(1))
    return r


# =============================================================================
# 3. Missing / special values
# =============================================================================
def category_names(n_special: int) -> List[str]:
    return ["regular only", "has missing"] + [f"has special {k}" for k in range(1, n_special + 1)]


def patch_state_category(state_patches: torch.Tensor) -> torch.Tensor:
    """(B, V, Np, P) state codes -> (B, V, Np) patch category:
    0 = all cells regular, 1 = contains missing (no special), 1+k = contains special k
    (the highest code if a patch has several)."""
    sp = state_patches.long()
    special = torch.where(sp >= 2, sp, torch.zeros_like(sp)).amax(-1)
    missing = (sp == 1).any(-1)
    return torch.where(special > 0, special, missing.long())


def relevance_by_category(relevance: np.ndarray, category: np.ndarray, n_special: int
                          ) -> pd.DataFrame:
    """Share of the CLS relevance that lands on each kind of patch.

    ``relevance`` (N, V, Np) - patch part of :func:`cls_rollout`; each record is
    renormalised to sum 1 so every record counts equally. ``lift`` = mean relevance of a
    patch in that category / mean relevance of any patch (1 = no preference)."""
    rel = relevance / np.clip(relevance.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    overall = rel.mean()
    rows = []
    for k, name in enumerate(category_names(n_special)):
        m = category == k
        if not m.any():
            continue
        rows.append({"category": name, "code": k, "n_patches": int(m.sum()),
                     "share_of_patches": float(m.mean()),
                     "share_of_relevance": float(rel[m].sum() / rel.sum()),
                     "lift": float(rel[m].mean() / overall)})
    return pd.DataFrame(rows)


def relevance_by_variable(relevance: np.ndarray, category: np.ndarray,
                          var_names: Sequence[str]) -> pd.DataFrame:
    """Per variable: share of patch relevance, and how often its patches are not all regular."""
    rel = relevance / np.clip(relevance.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    return pd.DataFrame({"variable": list(var_names),
                         "share_of_relevance": rel.sum(axis=(0, 2)) / rel.sum(),
                         "share_patches_not_regular": (category > 0).mean(axis=(0, 2))})


# =============================================================================
# 4. Embeddings
# =============================================================================
def state_names(n_special: int) -> List[str]:
    return ["regular", "missing"] + [f"special {k}" for k in range(1, n_special + 1)]


@torch.no_grad()
def cell_embedding_map(model: TSFusionTransformer,
                       x_values: Sequence[float] = tuple(np.linspace(-3, 3, 61))) -> dict:
    """Cell-level vectors of the value-aware tokenizer (early / hybrid):
    ``regular`` (len(x_values), d_s) = w*x + b + E[regular] for each standardised x,
    ``states`` {state_code: E[state]} for missing and every special code."""
    emb = model.value_embed
    if not hasattr(emb, "value_proj"):
        raise ValueError("cell_embedding_map needs missing_fusion='early' or 'hybrid'")
    x = torch.as_tensor(np.asarray(x_values, dtype=np.float32)).view(-1, 1)
    w = emb.state_emb.weight.detach().cpu()
    wv, bv = emb.value_proj.weight.detach().cpu(), emb.value_proj.bias.detach().cpu()
    regular = (torch.nn.functional.linear(x, wv, bv) + w[0]).numpy()
    return {"x_values": np.asarray(x_values), "regular": regular,
            "states": {s: w[s].numpy() for s in range(1, w.shape[0])}}


def state_similarity(model: TSFusionTransformer) -> pd.DataFrame:
    """Cosine similarity between the learned state embeddings."""
    emb = model.value_embed if hasattr(model.value_embed, "state_emb") else model.state_embed
    w = emb.state_emb.weight.detach().cpu().float()
    cos = torch.nn.functional.cosine_similarity(w[:, None], w[None], dim=-1).numpy()
    names = state_names(w.shape[0] - 2)
    return pd.DataFrame(cos, index=names, columns=names)


def pca_2d(X: np.ndarray, fit_on: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project rows of ``X`` onto the first 2 principal components of ``fit_on``
    (default ``X``). Returns ``(coords (n, 2), explained_variance_ratio (2,))``."""
    F = X if fit_on is None else fit_on
    mu = F.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    ratio = (s ** 2) / np.sum(s ** 2)
    return (X - mu) @ vt[:2].T, ratio[:2]


# =============================================================================
# 5. Behaviour: what the model uses
# =============================================================================
def to_original(link: torch.Tensor, prep: Preprocessor, train_cfg: TrainConfig) -> np.ndarray:
    """Network output (link scale) -> predictions on the original target scale."""
    z = build_loss(train_cfg).inverse_link(link.float()).cpu().numpy()
    return prep.target_scaler.inverse_transform(z)


@torch.no_grad()
def _embed_all(model, data: ProcessedData, batch_size: int) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {}
    for b in InMemoryLoader(data, batch_size, device=device):
        for k, v in model.embed(b["x"], b["state"], b.get("snapshot")).items():
            parts.setdefault(k, []).append(v.float())
    return {k: torch.cat(v) for k, v in parts.items()}


@torch.no_grad()
def _head(model, emb: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model.head(torch.cat(list(emb.values()), dim=-1).float())


@torch.no_grad()
def _predict(model, data: ProcessedData, prep, train_cfg, batch_size) -> np.ndarray:
    return to_original(_head(model, _embed_all(model, data, batch_size)), prep, train_cfg)


def _log_change(new: np.ndarray, base: np.ndarray) -> float:
    pos = (new > 0) & (base > 0)
    return float(np.mean(np.abs(np.log(new[pos] / base[pos])))) if pos.any() else float("nan")


@torch.no_grad()
def modality_ablation(model: TSFusionTransformer, data: ProcessedData, prep: Preprocessor,
                      train_cfg: TrainConfig, batch_size: int = 1024) -> pd.DataFrame:
    """Replace one CLS embedding by its mean over ``data`` (i.e. remove the record-specific
    information of that branch) and measure the effect on the predictions."""
    emb = _embed_all(model, data, batch_size)
    base = to_original(_head(model, emb), prep, train_cfg)
    y = data.y_raw
    rows = []
    for key in emb:
        abl = {k: (v.mean(0, keepdim=True).expand_as(v) if k == key else v) for k, v in emb.items()}
        pred = to_original(_head(model, abl), prep, train_cfg)
        row = {"branch": key, "mean_abs_log_change": _log_change(pred, base)}
        if y is not None:
            row["rmse_increase_pct"] = 100 * (rmse(y, pred) / rmse(y, base) - 1)
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def permutation_importance(model: TSFusionTransformer, data: ProcessedData, prep: Preprocessor,
                           train_cfg: TrainConfig, var_names: Sequence[str],
                           snapshot_names: Sequence[str] = (), n_repeats: int = 3,
                           seed: int = 0, batch_size: int = 1024) -> pd.DataFrame:
    """Shuffle one input across records (a time-series variable: its whole 11-step series
    of values AND states; a snapshot variable: its PLE vector) and measure the change in
    predictions and RMSE. Time-series and snapshot variables end up on one scale."""
    rng = np.random.default_rng(seed)
    base = _predict(model, data, prep, train_cfg, batch_size)
    y = data.y_raw
    n = len(data)
    groups = [("time series", v, nm) for v, nm in enumerate(var_names)]
    if data.snapshot is not None:
        groups += [("snapshot", j, nm) for j, nm in enumerate(snapshot_names)]
    rows = []
    for kind, j, name in groups:
        changes, incs = [], []
        for _ in range(n_repeats):
            perm = torch.as_tensor(rng.permutation(n))
            x, st, sn = data.x.clone(), data.state.clone(), data.snapshot
            if kind == "time series":
                x[:, :, j], st[:, :, j] = data.x[perm, :, j], data.state[perm, :, j]
            else:
                sn = data.snapshot.clone()
                sn[:, j] = data.snapshot[perm, j]
            pred = _predict(model, ProcessedData(x, st, sn, data.y, y), prep, train_cfg, batch_size)
            changes.append(_log_change(pred, base))
            if y is not None:
                incs.append(100 * (rmse(y, pred) / rmse(y, base) - 1))
        row = {"input": name, "kind": kind, "mean_abs_log_change": float(np.mean(changes))}
        if incs:
            row["rmse_increase_pct"] = float(np.mean(incs))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_abs_log_change", ascending=False,
                                          ignore_index=True)


@torch.no_grad()
def state_probe(model: TSFusionTransformer, data: ProcessedData, prep: Preprocessor,
                train_cfg: TrainConfig, n_cells: int = 400, seed: int = 0,
                batch_size: int = 1024) -> pd.DataFrame:
    """Switch randomly chosen *regular* cells to each missing / special state and report the
    change of the mean prediction of the affected records (counterfactual probe)."""
    rng = np.random.default_rng(seed)
    reg = np.argwhere(data.state.numpy() == 0)
    pick = reg[rng.choice(len(reg), size=min(n_cells, len(reg)), replace=False)]
    touched = np.unique(pick[:, 0])
    base = _predict(model, data, prep, train_cfg, batch_size)
    n_states = model.cfg.n_states
    rows = []
    for code, name in list(enumerate(state_names(n_states - 2)))[1:]:
        st = data.state.clone()
        st[pick[:, 0], pick[:, 1], pick[:, 2]] = code
        pred = _predict(model, ProcessedData(data.x, st, data.snapshot, data.y, data.y_raw),
                        prep, train_cfg, batch_size)
        rows.append({"state": name, "code": code,
                     "mean_prediction_change_pct": 100 * (pred[touched].mean() / base[touched].mean() - 1),
                     "mean_abs_log_change": _log_change(pred[touched], base[touched])})
    return pd.DataFrame(rows)


# =============================================================================
# 6. Dataset pass
# =============================================================================
@dataclass
class InterpretationResult:
    index: np.ndarray                 # records used (positions in the dataset)
    link: np.ndarray                  # (N,) network output (link scale)
    cls: Dict[str, np.ndarray]        # branch -> (N, d)
    rollout: np.ndarray               # (N, V+1, Np+1) relevance for the global time-series CLS
    patch_category: np.ndarray        # (N, V, Np) see patch_state_category
    global_to_var: np.ndarray         # (N, V)     last layer, heads averaged
    global_to_time: np.ndarray        # (N, Np)
    timecls_to_var: np.ndarray        # (N, V, Np)
    varcls_to_time: np.ndarray        # (N, V, Np)
    snapshot_attn: Optional[np.ndarray] = None      # (N, H, S) last layer CLS -> snapshot tokens
    snapshot_rollout: Optional[np.ndarray] = None   # (N, S+1)
    snapshot_tokens: Optional[np.ndarray] = None    # (N, S, d)

    @property
    def patch_relevance(self) -> np.ndarray:
        return self.rollout[:, 1:, 1:]


def collect_interpretation(model: TSFusionTransformer, data: ProcessedData,
                           batch_size: int = 64, max_records: Optional[int] = None,
                           seed: int = 0) -> InterpretationResult:
    """Attention rollout, CLS attention, patch categories and embeddings for (a random
    subset of) a dataset. Results take O(N * V * Np) memory - use ``max_records`` on large
    data. Recording materialises the attention weights: the variable stage alone needs
    about ``batch_size * (Np+1) * n_heads * (V+1)**2 * 4`` bytes on the device, so lower
    ``batch_size`` when there are hundreds of variables (e.g. 8 for V = 1000)."""
    idx = np.arange(len(data))
    if max_records is not None and max_records < len(data):
        idx = np.sort(np.random.default_rng(seed).choice(len(data), max_records, replace=False))
    device = next(model.parameters()).device
    acc: Dict[str, list] = {}

    def add(k, v):
        acc.setdefault(k, []).append(v.detach().float().cpu().numpy())

    for b in InMemoryLoader(data, batch_size, indices=idx, device=device):
        out, emb, det, maps = run_with_attention(model, b["x"], b["state"], b.get("snapshot"))
        add("link", out)
        for k, v in emb.items():
            add(f"cls:{k}", v)
        add("rollout", cls_rollout(maps.var, maps.time))
        add("patch_category", patch_state_category(det["state_patches"]))
        for k, v in cls_attention_summary(maps).items():
            add(k, v)
        if maps.snapshot:
            add("snapshot_attn", snapshot_cls_attention(maps))
            add("snapshot_rollout", snapshot_rollout(maps))
            add("snapshot_tokens", det["snapshot_tokens"])
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    return InterpretationResult(
        index=idx, link=cat["link"],
        cls={k[4:]: v for k, v in cat.items() if k.startswith("cls:")},
        rollout=cat["rollout"], patch_category=cat["patch_category"].astype(int),
        global_to_var=cat["global_to_var"], global_to_time=cat["global_to_time"],
        timecls_to_var=cat["timecls_to_var"], varcls_to_time=cat["varcls_to_time"],
        snapshot_attn=cat.get("snapshot_attn"), snapshot_rollout=cat.get("snapshot_rollout"),
        snapshot_tokens=cat.get("snapshot_tokens"))
