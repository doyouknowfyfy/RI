"""Static matplotlib figures for :mod:`tsfusion.interpret` results.

Colour roles (from a validated reference palette; one meaning per colour everywhere):
  * cell / patch state - identity:   regular = neutral gray, missing = blue,
    special = orange (the special code is always printed as a label or marker, so colour
    never carries the code on its own)
  * attention / relevance - magnitude: one-hue blue ramp (light = low)
  * similarity - polarity:           blue <-> gray <-> red, centred at 0
  * model-behaviour bars (importance): violet, single series
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
REGULAR, MISSING, SPECIAL = "#898781", "#2a78d6", "#eb6834"
BEHAVIOUR = "#6250d6"
SEQ = LinearSegmentedColormap.from_list("tsf_blue", [
    "#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# for points / lines: the ramp minus its lightest steps, so low values stay visible on the surface
SEQ_MARKS = LinearSegmentedColormap.from_list("tsf_blue_marks", [
    "#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("tsf_div", [
    "#104281", "#3987e5", "#9ec5f4", NEUTRAL, "#f2b0af", "#e34948", "#9e2b2a"])
SPECIAL_MARKERS = ["D", "^", "s", "v", "P", "X"]


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "figure.dpi": 110, "font.family": "sans-serif"})


def state_color(code: int) -> str:
    return REGULAR if code == 0 else MISSING if code == 1 else SPECIAL


def state_glyph(code: int) -> str:
    return "" if code == 0 else "M" if code == 1 else f"S{code - 1}"


def state_legend(n_special: int, regular_color: str = NEUTRAL):
    return [Patch(facecolor=regular_color, edgecolor=BASELINE, label="regular"),
            Patch(facecolor=MISSING, label="missing (M)"),
            Patch(facecolor=SPECIAL, label=f"special (S1..S{n_special})")]


def colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=INK2)
    cb.set_label(label, fontsize=8, color=INK2)
    return cb


# ----------------------------------------------------------------------------- grids
def plot_state_grid(ax, state_tv: np.ndarray, var_names: Sequence[str],
                    time_labels: Sequence, title: str = "cell states") -> None:
    """(T, V) state codes -> variables x time grid; code printed in every non-regular cell."""
    st = np.asarray(state_tv).T                                   # (V, T)
    codes = sorted(set(np.unique(st).tolist()) | {0})
    lut = {c: i for i, c in enumerate(codes)}
    ax.imshow(np.vectorize(lut.get)(st), aspect="auto", interpolation="nearest",
              cmap=ListedColormap([NEUTRAL if c == 0 else state_color(c) for c in codes]),
              vmin=-0.5, vmax=len(codes) - 0.5)
    for v in range(st.shape[0]):
        for t in range(st.shape[1]):
            if st[v, t]:
                ax.text(t, v, state_glyph(int(st[v, t])), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
    _grid_ticks(ax, var_names, time_labels)
    ax.set_xlabel("time step")
    ax.set_title(title, loc="left")


def _grid_ticks(ax, row_labels, col_labels):
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _relative_norm(values: np.ndarray) -> Normalize:
    d = max(float(np.nanmax(np.abs(values - 1.0))), 1e-6)
    return Normalize(vmin=1 - d, vmax=1 + d)


def plot_patch_heatmap(fig, ax, values: np.ndarray, category: Optional[np.ndarray],
                       var_names: Sequence[str], patch_labels: Sequence, title: str,
                       cbar_label: str, vmax: Optional[float] = None,
                       relative: bool = False) -> None:
    """(V, Np) magnitudes (blue ramp); patches holding missing / special cells are marked
    with their glyph (M, S1, ...) so state is readable without colour. With
    ``relative=True`` the values are ratios to a reference (1 = reference), drawn on the
    diverging scale centred at 1."""
    if relative:
        norm, cmap = _relative_norm(values), DIV
        is_dark = lambda val: abs(val - 1) > 0.6 * (norm.vmax - 1)  # noqa: E731
    else:
        vmax = float(np.nanmax(values)) if vmax is None else vmax
        norm, cmap = Normalize(vmin=0, vmax=vmax), SEQ
        is_dark = lambda val: val > 0.55 * vmax  # noqa: E731
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    if category is not None:
        for v in range(values.shape[0]):
            for n in range(values.shape[1]):
                if category[v, n]:
                    ax.text(n, v, state_glyph(int(category[v, n])), ha="center", va="center",
                            fontsize=7, fontweight="bold",
                            color="white" if is_dark(values[v, n]) else INK)
    _grid_ticks(ax, var_names, patch_labels)
    ax.set_xlabel("patch (time window)")
    ax.set_title(title, loc="left")
    colorbar(fig, im, ax, cbar_label)


def plot_attention_matrix(fig, ax, attn: np.ndarray, labels: Sequence[str], title: str,
                          relative: bool = True) -> None:
    """One (query x key) attention matrix, heads averaged (rows sum to 1).

    ``relative=True`` (default) shows weight x number of keys: 1 = the key gets exactly a
    uniform share, >1 (red) more, <1 (blue) less. Near-uniform layers - common in small
    models - are otherwise one flat colour."""
    if relative:
        vals = attn * attn.shape[-1]
        im = ax.imshow(vals, cmap=DIV, norm=_relative_norm(vals), interpolation="nearest")
        cbar = "attention ÷ uniform share"
    else:
        im = ax.imshow(attn, cmap=SEQ, vmin=0, vmax=max(float(attn.max()), 1e-6),
                       interpolation="nearest")
        cbar = "attention weight"
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("key (attended to)")
    ax.set_ylabel("query")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    colorbar(fig, im, ax, cbar)


# ----------------------------------------------------------------------------- bars
def plot_hbar(ax, labels: Sequence[str], values: Sequence[float], colors=BEHAVIOUR,
              title: str = "", xlabel: str = "", fmt: str = "{:.2f}",
              ref: Optional[float] = None, errors: Optional[Sequence[float]] = None) -> None:
    """Horizontal bars, first label on top, value printed at each bar end."""
    y = np.arange(len(labels))[::-1]
    values = np.asarray(values, dtype=float)
    ax.barh(y, values, height=0.62, color=colors, xerr=errors,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    span = np.nanmax(np.abs(values)) if len(values) else 1.0
    for yi, v in zip(y, values):
        ax.text(v + (0.015 * span if v >= 0 else -0.015 * span), yi, fmt.format(v),
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}, zorder=4)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=BASELINE, linewidth=1)
    if ref is not None:
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    lo, hi = min(0.0, float(np.nanmin(values))), max(0.0, float(np.nanmax(values)))
    ax.set_xlim(lo - 0.32 * span * (lo < 0), hi + 0.22 * span)


def plot_category_lift(ax, table: pd.DataFrame, title: str = "") -> None:
    """Bars of relevance lift per patch category (1 = an average patch), coloured by state."""
    colors = [state_color(int(c)) for c in table["code"]]
    labels = [f"{c}  ({100 * s:.1f}% of patches)" for c, s in
              zip(table["category"], table["share_of_patches"])]
    plot_hbar(ax, labels, table["lift"], colors=colors, ref=1.0, fmt="{:.2f}×",
              title=title, xlabel="relevance per patch vs. average patch")


# ----------------------------------------------------------------------------- embeddings
def plot_cell_embedding_map(ax, cmap: dict, title: str = "") -> None:
    """PCA of the cell-level vectors: the line of regular values (x = -3..3) and one point
    per missing / special state, each labelled."""
    from .interpret import pca_2d
    states = cmap["states"]
    pts = np.vstack([cmap["regular"]] + [states[k][None] for k in sorted(states)])
    xy, ratio = pca_2d(pts)
    n = len(cmap["regular"])
    line, xs = xy[:n], cmap["x_values"]
    ax.plot(line[:, 0], line[:, 1], color=INK2, linewidth=2, zorder=1)
    for target in (-2, -1, 0, 1, 2):
        i = int(np.argmin(np.abs(xs - target)))
        ax.scatter(*line[i], s=22, color=SURFACE, edgecolor=INK2, linewidth=1.2, zorder=2)
        ax.annotate(f"x={target:+d}" if target else "x=0 (mean)", line[i], fontsize=7,
                    color=INK2, xytext=(4, -10), textcoords="offset points")
    for j, k in enumerate(sorted(states)):
        p = xy[n + j]
        is_missing = k == 1
        ax.scatter(*p, s=70, color=MISSING if is_missing else SPECIAL,
                   marker="o" if is_missing else SPECIAL_MARKERS[(k - 2) % len(SPECIAL_MARKERS)],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate("missing" if is_missing else f"special {k - 1}", p, fontsize=8,
                    fontweight="bold", color=INK, xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.0f}%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")


def plot_similarity(fig, ax, sim: pd.DataFrame, title: str = "") -> None:
    im = ax.imshow(sim.values, cmap=DIV, vmin=-1, vmax=1)
    names = list(sim.index)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = sim.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    colorbar(fig, im, ax, "cosine similarity")


def plot_embedding_scatter(fig, ax, xy: np.ndarray, values: np.ndarray, title: str,
                           cbar_label: str, ratio: Optional[np.ndarray] = None) -> None:
    """2-D projection coloured by a magnitude (blue ramp). Points are plotted in random
    order so neither end of the scale hides the other."""
    order = np.random.default_rng(0).permutation(len(xy))
    im = ax.scatter(xy[order, 0], xy[order, 1], c=np.asarray(values)[order], cmap=SEQ_MARKS, s=7,
                    alpha=0.8, linewidths=0)
    if ratio is not None:
        ax.set_xlabel(f"PC1 ({100 * ratio[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * ratio[1]:.0f}%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    colorbar(fig, im, ax, cbar_label)
