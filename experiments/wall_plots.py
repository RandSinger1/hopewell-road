"""
wall_plots.py — Figures for wall_lab Results.

    show(res)                          one panel: a stage + detected pairs
    show_stages(res)                   every pipeline stage side by side
    show_sweep(results, "param")       one panel per Result, titled by the swept value
    show_sites_sweep(sites, "param", values)   rows = sites, cols = values
    plot_counts(sites, "param", values)        pair count / top spacing vs value
    show_profile(res, i)                       mean LRM cross-section of pair i

Panels are drawn in pixel space with metre tick labels centred on the site,
so the axes read as distance from the coordinate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from report.report_utils import render_hillshaded_dtm, pair_style
from experiments.wall_lab import Params, Result, run

STAGES = ("dtm", "lrm", "ridge", "orientation", "binary", "closed", "skeleton")


def _stage_image(res: Result, stage: str):
    """(array, imshow kwargs) for one stage."""
    nd = res.raw == res.nodata
    if stage == "dtm":
        return render_hillshaded_dtm(res.smoothed, nodata=res.nodata, z_factor=4.0), {}
    if stage == "lrm":
        v = np.where(nd, np.nan, res.lrm)
        lim = np.nanpercentile(np.abs(v), 98) or 1.0
        return v, {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}
    if stage == "ridge":
        return np.where(nd, np.nan, res.ridge), {"cmap": "magma",
                                                  "vmax": np.nanpercentile(res.ridge, 99.5) or None}
    if stage in ("orientation", "binary", "closed", "skeleton"):
        arr = {"orientation": res.orientation_mask, "binary": res.binary,
               "closed": res.closed, "skeleton": res.skeleton}[stage]
        return arr.astype(float), {"cmap": "gray", "vmin": 0, "vmax": 1}
    raise ValueError(f"stage must be one of {STAGES}")


def _metre_ticks(ax, res: Result, step_m: float = 100):
    h, w = res.raw.shape
    for axis, n in ((ax.xaxis, w), (ax.yaxis, h)):
        half = n * res.m_per_px / 2
        ms = np.arange(-np.floor(half / step_m) * step_m, half + 1e-9, step_m)
        axis.set_ticks(n / 2 + ms / res.m_per_px)
        axis.set_ticklabels([f"{m:+.0f}" if m else "0" for m in (ms if axis is ax.xaxis else -ms)])
    ax.tick_params(labelsize=7)


def draw_pairs(ax, res: Result, raw_pairs: bool = False, labels: bool = True,
               true_positive=None, top: int | None = None):
    """Overlay detected pairs. true_positive=None uses the site's
    true_positive_pair_index only when parameters are default (indices shift
    as parameters change, so a stored index is meaningless otherwise)."""
    pairs = res.pairs_raw if raw_pairs else res.pairs
    pairs = pairs[:top] if top else pairs      # pairs are sorted by score
    if true_positive is None and not res.params.diff():
        true_positive = res.coord.get("true_positive_pair_index")
    for i, p in enumerate(pairs):
        st = pair_style(i, true_positive)
        for line, color in zip((p["line1"], p["line2"]), st["colors"]):
            (x0, y0), (x1, y1) = line
            ax.plot([x0, x1], [y0, y1], color=color, lw=st["linewidth"],
                    alpha=st["alpha"], zorder=st["zorder"])
        if labels:
            (ax0, ay0), (ax1, ay1) = p["line1"]
            (bx0, by0), (bx1, by1) = p["line2"]
            cx, cy = (ax0 + ax1 + bx0 + bx1) / 4, (ay0 + ay1 + by0 + by1) / 4
            ax.text(cx, cy, f"#{i}\n{p['spacing'] * res.m_per_px:.0f} m", color=st["label_color"],
                    fontsize=7, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))


def show(res: Result, stage: str = "dtm", ax=None, title: str | None = None,
         pairs: bool = True, raw_pairs: bool = False, labels: bool = True,
         top: int | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    img, kw = _stage_image(res, stage)
    ax.imshow(img, interpolation="nearest", **kw)
    if pairs:
        draw_pairs(ax, res, raw_pairs=raw_pairs, labels=labels, top=top)
    ax.set_xlim(0, res.raw.shape[1] - 1)
    ax.set_ylim(res.raw.shape[0] - 1, 0)
    _metre_ticks(ax, res)
    if title is None:
        title = f"{res.coord['name']} — {stage}\n{len(res.pairs)} pair(s), target {res.target_angle:.1f}°"
    ax.set_title(title, fontsize=9)
    return ax


def show_stages(res: Result, stages=STAGES, ncols: int = 4):
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, stage in zip(axes.flat, stages):
        show(res, stage, ax=ax, title=stage, pairs=stage in ("dtm", "skeleton"),
             labels=stage == "dtm")
    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    changed = res.params.diff()
    fig.suptitle(f"{res.coord['name']} ({res.mode}, target {res.target_angle:.1f}°) — "
                 f"{len(res.pairs)} pair(s)" + (f"\n{changed}" if changed else ""), fontsize=10)
    fig.tight_layout()
    return fig


def _label(res: Result, param: str | None):
    if param is None:
        return ", ".join(f"{k}={v}" for k, v in res.params.diff().items()) or "defaults"
    return f"{param}={getattr(res.params, param)}"


def show_sweep(results: list[Result], param: str | None = None, stage: str = "dtm",
               ncols: int = 4, size: float = 4, top: int | None = None):
    """One panel per Result. `param` is the swept field (used for titles)."""
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax, res in zip(axes.flat, results):
        show(res, stage, ax=ax, title=f"{_label(res, param)}\n{len(res.pairs)} pair(s)", top=top)
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    fig.suptitle(results[0].coord["name"] if len({r.coord["name"] for r in results}) == 1 else "",
                 fontsize=11)
    fig.tight_layout()
    return fig


def show_sites_sweep(sites, param: str, values, base: Params | None = None,
                     stage: str = "dtm", size: float = 3.6, top: int | None = None):
    """Rows = sites, columns = values of one Params field."""
    base = base or Params()
    fig, axes = plt.subplots(len(sites), len(values),
                             figsize=(size * len(values), size * len(sites)), squeeze=False)
    for r, s in enumerate(sites):
        for c, v in enumerate(values):
            res = run(s, base.with_(**{param: v}))
            show(res, stage, ax=axes[r, c],
                 title=(f"{param}={v}\n" if r == 0 else "") + f"{len(res.pairs)} pair(s)",
                 top=top)
        axes[r, 0].set_ylabel(run(s, base).coord["name"], fontsize=10)
    fig.tight_layout()
    return fig


def plot_counts(sites, param: str, values, base: Params | None = None, ax=None):
    """Pair count and top-pair spacing vs one Params field, one line per site."""
    base = base or Params()
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    xs = [str(v) for v in values]
    for s in sites:
        rs = [run(s, base.with_(**{param: v})) for v in values]
        name = rs[0].coord["name"]
        ax[0].plot(xs, [len(r.pairs) for r in rs], "o-", label=name)
        ax[1].plot(xs, [r.pairs[0]["spacing"] * r.m_per_px if r.pairs else np.nan for r in rs],
                   "o-", label=name)
    ax[0].set_ylabel("pairs detected")
    ax[1].set_ylabel("top-scoring pair spacing (m)")
    ax[1].axhline(base.spacing_meters, color="gray", ls=":", lw=1)
    for a in ax:
        a.set_xlabel(param)
        a.legend(fontsize=8)
    plt.tight_layout()
    return ax


def show_profile(res: Result, pair_index: int = 0, half_width_m: float = 60,
                 along_m: float | None = None, ax=None):
    """Mean LRM cross-section perpendicular to a detected pair, averaged along
    the pair's overlap (or ±along_m around its centre)."""
    from scipy.ndimage import map_coordinates
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    if pair_index >= len(res.pairs):
        ax.set_title(f"no pair #{pair_index}")
        return ax
    p = res.pairs[pair_index]
    pts = np.array([*p["line1"], *p["line2"]], dtype=float)
    centre = pts.mean(axis=0)
    d = pts[1] - pts[0]
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    half_len_px = (along_m / res.m_per_px if along_m else
                   0.5 * min(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[3] - pts[2])))
    ts = np.linspace(-half_len_px, half_len_px, 40)
    ss = np.linspace(-half_width_m, half_width_m, 241) / res.m_per_px
    S, T = np.meshgrid(ss, ts)
    xs = centre[0] + T * d[0] + S * n[0]
    ys = centre[1] + T * d[1] + S * n[1]
    lrm = np.where(res.raw == res.nodata, np.nan, res.lrm)
    prof = map_coordinates(lrm, [ys, xs], order=1, mode="constant", cval=np.nan)
    with np.errstate(all="ignore"):
        mean, std = np.nanmean(prof, axis=0), np.nanstd(prof, axis=0)
    x_m = ss * res.m_per_px
    ax.fill_between(x_m, mean - std, mean + std, alpha=0.25)
    ax.plot(x_m, mean, lw=1.5)
    sp = p["spacing"] * res.m_per_px
    for x in (-sp / 2, sp / 2):
        ax.axvline(x, color="red", ls="--", lw=0.8)
    ax.set_xlabel("distance across walls (m)")
    ax.set_ylabel("LRM (m)")
    ax.set_title(f"{res.coord['name']} pair #{pair_index}: spacing {sp:.1f} m", fontsize=9)
    return ax
