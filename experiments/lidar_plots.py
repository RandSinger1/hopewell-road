"""
lidar_plots.py — Figures for lidar_lab LidarResults.

    show_lidar_sweep(sites, "smrf_window", [24, 60, 100])   rows = sites, cols = values
    plot_lidar_stats(sites, "smrf_window", [24, 60, 100])   ground %, DTM gaps, pairs vs value
    ground_density(lr)                                     ground points per m² (gaps under trees)
    compare_transects(results)                             detrended DTM transect across the corridor
    dtm_difference(a, b)                                   elevation difference between two variants
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.lidar_lab import LidarParams, LidarResult, run_lidar, FT_PER_M
from experiments import wall_plots as wp
from report.report_utils import sample_profile


def _value_label(lr: LidarResult, param: str | None) -> str:
    if param is None:
        return ", ".join(f"{k}={v}" for k, v in lr.lidar.diff().items()) or "defaults"
    return f"{param}={getattr(lr.lidar, param)}"


def _title(lr: LidarResult, head: str = "") -> str:
    s = lr.stats
    return (head + "\n" if head else "") + \
        f"ground {s['ground_pct']}% · gaps {s['dtm_nodata_pct']}% · {s['pairs']} pair(s)"


def show_lidar_sweep(sites, param: str, values, base: LidarParams | None = None,
                     wall=None, size_m: float | None = 120, stage: str = "dtm",
                     size: float = 3.6, top: int | None = 5):
    """Rows = sites, columns = values of one LidarParams field, with detected pairs."""
    base = base or LidarParams()
    fig, axes = plt.subplots(len(sites), len(values),
                             figsize=(size * len(values), size * len(sites) + 0.3), squeeze=False)
    for r, s in enumerate(sites):
        for c, v in enumerate(values):
            lr = run_lidar(s, base.with_(**{param: v}), wall=wall, size_m=size_m)
            wp.show(lr.wall, stage, ax=axes[r, c], top=top,
                    title=_title(lr, f"{param}={v}" if r == 0 else ""))
        axes[r, 0].set_ylabel(s, fontsize=10)
    fig.tight_layout()
    return fig


def show_lidar_variants(site, variants: dict, wall=None, size_m: float | None = 120,
                        stage: str = "dtm", size: float = 3.6, top: int | None = 5):
    """One panel per named LidarParams, e.g. {"production": LidarParams(), ...}."""
    fig, axes = plt.subplots(1, len(variants), figsize=(size * len(variants), size + 0.4),
                             squeeze=False)
    out = {}
    for ax, (label, lp) in zip(axes[0], variants.items()):
        lr = out[label] = run_lidar(site, lp, wall=wall, size_m=size_m)
        wp.show(lr.wall, stage, ax=ax, top=top, title=_title(lr, label))
    fig.suptitle(site, fontsize=11)
    fig.tight_layout()
    return out


def plot_lidar_stats(sites, param: str, values, base: LidarParams | None = None,
                     wall=None, size_m: float | None = 120):
    base = base or LidarParams()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
    xs = [str(v) for v in values]
    for s in sites:
        lrs = [run_lidar(s, base.with_(**{param: v}), wall=wall, size_m=size_m) for v in values]
        ax[0].plot(xs, [lr.stats["ground_pct"] for lr in lrs], "o-", label=s)
        ax[1].plot(xs, [lr.stats["dtm_nodata_pct"] for lr in lrs], "o-", label=s)
        ax[2].plot(xs, [lr.stats["pairs"] for lr in lrs], "o-", label=s)
        ax[0].axhline(lrs[0].stats["vendor_ground_pct"], ls=":", lw=1,
                      color=ax[0].lines[-1].get_color())
    for a, yl in zip(ax, ["ground points (%)  · dotted = vendor class 2",
                          "DTM no-data cells (%)", "wall pairs detected"]):
        a.set_xlabel(param)
        a.set_ylabel(yl, fontsize=8)
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def ground_density(lr: LidarResult, cell_m: float = 2.0, ax=None, vmax: float | None = None):
    """Ground points per m² on a cell_m grid — low values are where the DTM is
    interpolated (canopy, buildings, water)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x0, x1, y0, y1 = lr.bbox_ft
    cell_ft = cell_m * FT_PER_M
    nx, ny = int((x1 - x0) / cell_ft), int((y1 - y0) / cell_ft)
    H, _, _ = np.histogram2d(lr.ground_xy[:, 1], lr.ground_xy[:, 0], bins=[ny, nx],
                             range=[[y0, y1], [x0, x1]])
    dens = H[::-1] / cell_m ** 2
    half = (x1 - x0) / FT_PER_M / 2
    im = ax.imshow(dens, cmap="viridis", vmax=vmax or np.percentile(dens, 99),
                   extent=[-half, half, -half, half])
    plt.colorbar(im, ax=ax, fraction=0.046, label="ground pts / m²")
    ax.set_title(f"{lr.coord['name']} ground density\n{_value_label(lr, None)}", fontsize=9)
    ax.set_xlabel("m from site")
    return ax


def compare_transects(results: dict, bearing_deg: float | None = None,
                      half_width_m: float = 80, ax=None):
    """Detrended elevation transect through the site centre, perpendicular to
    the expected wall bearing, one line per variant ({label: LidarResult})."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.4))
    for label, lr in results.items():
        res = lr.wall
        b = bearing_deg if bearing_deg is not None else res.target_angle
        d, z = sample_profile(res.raw, list(res.extent), lr.coord["lat"], lr.coord["lon"],
                              b, half_width_m, 400, nodata=res.nodata)
        ok = np.isfinite(z)
        if ok.sum() > 2:
            z = z - np.polyval(np.polyfit(d[ok], z[ok], 1), d)
        ax.plot(d, z, lw=1.2, label=label)
    ax.set_xlabel("distance across corridor (m)")
    ax.set_ylabel("detrended elevation (m)")
    ax.legend(fontsize=8)
    ax.set_title(f"{next(iter(results.values())).coord['name']}: transect at {b:.1f}°", fontsize=9)
    return ax


def dtm_difference(a: LidarResult, b: LidarResult, ax=None, lim_m: float | None = None):
    """b − a elevation (m) where both DTMs have data (same window/grid needed)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ra, rb = a.wall, b.wall
    h = min(ra.raw.shape[0], rb.raw.shape[0])
    w = min(ra.raw.shape[1], rb.raw.shape[1])
    za, zb = ra.raw[:h, :w].astype(float), rb.raw[:h, :w].astype(float)
    bad = (za == ra.nodata) | (zb == rb.nodata)
    diff = np.where(bad, np.nan, zb - za)
    lim = lim_m or np.nanpercentile(np.abs(diff), 98) or 0.5
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=ax, fraction=0.046, label="m")
    ax.set_title(f"{_value_label(b, None)}\nminus {_value_label(a, None)}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return ax
