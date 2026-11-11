"""Smooth direct-peak 42-bin state map for alpha2=[0, .35], fs=[.2, .9]."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator


HERE = Path(__file__).resolve().parent
BOUNDARY_CSV = HERE / "mf_42bin_direct_peak_state_changing_boundaries.csv"
GRID_CSV = HERE / "mf_42bin_direct_peak_state_composition_grid.csv"
OUTPUT_PNG = HERE / "mf_42bin_direct_peak_fs0p2_0p9_smooth.png"
OUTPUT_PDF = HERE / "mf_42bin_direct_peak_fs0p2_0p9_smooth.pdf"

ALPHA_MIN, ALPHA_MAX = 0.0, 0.35
FS_MIN, FS_MAX = 0.20, 0.90
LOWER_JUNCTION_FS = 0.70235
TP_ALPHA = 0.1932732499
TP_FS = 0.7957070426

COEXISTENCE = "Coexistence\n(biased unimodal and bimodal)"
BIASED = "Biased unimodal"
BIMODAL = "Bimodal polarized"
NEUTRAL = "Neutral unimodal"

STATE_COLORS = {
    COEXISTENCE: "#ef9a9a",
    BIASED: "#ffe08a",
    BIMODAL: "#b39ddb",
    NEUTRAL: "#9ec9ee",
}
BOUNDARY_COLORS = {
    "Saddle-node": "#00D51A",
    "Supercritical pitchfork": "#FFD000",
    "Subcritical pitchfork": "#E67E22",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_boundary_rows() -> list[dict[str, str]]:
    with BOUNDARY_CSV.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_interpolator(points: list[tuple[float, float]]):
    """Return monotone-shape interpolation alpha2(fs) for one branch."""
    points = sorted(set(points), key=lambda point: point[1])
    fs = np.asarray([point[1] for point in points], dtype=float)
    alpha = np.asarray([point[0] for point in points], dtype=float)
    rounded_fs = np.round(fs, 12)
    unique_fs = np.unique(rounded_fs)
    unique_alpha = np.asarray(
        [np.mean(alpha[rounded_fs == value]) for value in unique_fs],
        dtype=float,
    )
    return {
        "fs": unique_fs,
        "alpha": unique_alpha,
        "interpolator": PchipInterpolator(unique_fs, unique_alpha, extrapolate=False),
    }


def recover_left_entry(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Extend a low-fs branch smoothly to its alpha2=0 intersection."""
    ordered = sorted(points, key=lambda point: point[0])
    alpha = np.asarray([point[0] for point in ordered], dtype=float)
    fs = np.asarray([point[1] for point in ordered], dtype=float)
    local_count = min(12, alpha.size)
    entry_fs = float(np.polyval(np.polyfit(alpha[:local_count], fs[:local_count], 2), 0.0))
    return [(0.0, entry_fs), *points]


def load_curves():
    rows = read_boundary_rows()
    saddle = [
        (float(row["alpha2"]), float(row["fs"]))
        for row in rows
        if row["boundary"] == "Saddle-node"
    ]
    supercritical = [
        (float(row["alpha2"]), float(row["fs"]))
        for row in rows
        if row["boundary"] == "Supercritical pitchfork"
    ]
    subcritical = [
        (float(row["alpha2"]), float(row["fs"]))
        for row in rows
        if row["boundary"] == "Subcritical pitchfork"
    ]

    lower_saddle = [point for point in saddle if point[1] <= LOWER_JUNCTION_FS + 0.002]
    upper_saddle = [point for point in saddle if point[1] >= TP_FS - 0.002]
    lower_saddle = recover_left_entry(lower_saddle)
    supercritical = recover_left_entry(supercritical)

    return {
        "lower_saddle": make_interpolator(lower_saddle),
        "upper_saddle": make_interpolator(upper_saddle),
        "supercritical": make_interpolator(supercritical),
        "subcritical": make_interpolator(subcritical),
    }


def fit_bimodal_neutral_boundary():
    """Smooth the direct-peak P/N classification boundary from the .005 grid."""
    grouped: dict[float, list[tuple[float, str]]] = {}
    with GRID_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(float(row["fs"]), []).append(
                (float(row["alpha2"]), row["composition"])
            )

    fs_points = []
    alpha_points = []
    for fs, row in sorted(grouped.items()):
        bimodal = sorted(a for a, state in row if state == BIMODAL)
        neutral = sorted(a for a, state in row if state == NEUTRAL)
        if not bimodal or not neutral:
            continue
        right_bimodal = max(bimodal)
        right_neutral = [a for a in neutral if a > right_bimodal]
        if not right_neutral:
            continue
        fs_points.append(fs)
        alpha_points.append(0.5 * (right_bimodal + min(right_neutral)))

    fs_points = np.asarray(fs_points, dtype=float)
    alpha_points = np.asarray(alpha_points, dtype=float)
    # A quadratic fit removes the visible .005-grid staircase while retaining
    # the monotone direct-peak classification boundary (RMSE about 0.0013).
    polynomial = np.poly1d(np.polyfit(fs_points, alpha_points, 2))
    return {
        "entry_fs": float(fs_points[0]),
        "polynomial": polynomial,
    }


def curve_value(curve, fs_value: float) -> float:
    return float(curve["interpolator"](fs_value))


def build_state_map(curves, pn_boundary):
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, 2101)
    fs_grid = np.linspace(FS_MIN, FS_MAX, 2101)
    # Codes: 0 coexistence, 1 biased, 2 bimodal, 3 neutral.
    states = np.full((fs_grid.size, alpha_grid.size), 3, dtype=np.int8)

    super_curve = curves["supercritical"]
    pn_entry = pn_boundary["entry_fs"]
    for row, fs_value in enumerate(fs_grid):
        if super_curve["fs"][0] <= fs_value <= TP_FS:
            super_alpha = curve_value(super_curve, fs_value)
            states[row, alpha_grid < super_alpha] = 1

            if fs_value >= pn_entry:
                pn_alpha = float(pn_boundary["polynomial"](fs_value))
                states[row, (alpha_grid >= super_alpha) & (alpha_grid < pn_alpha)] = 2

        if fs_value < LOWER_JUNCTION_FS:
            lower = curves["lower_saddle"]
            if lower["fs"][0] <= fs_value <= lower["fs"][-1]:
                fold_alpha = curve_value(lower, fs_value)
                states[row, alpha_grid < fold_alpha] = 0
        elif fs_value < TP_FS:
            sub = curves["subcritical"]
            if sub["fs"][0] <= fs_value <= sub["fs"][-1]:
                fold_alpha = curve_value(sub, fs_value)
                states[row, alpha_grid < fold_alpha] = 0
        else:
            upper = curves["upper_saddle"]
            if upper["fs"][0] <= fs_value <= upper["fs"][-1]:
                fold_alpha = curve_value(upper, fs_value)
                states[row, alpha_grid < fold_alpha] = 0
            pn_alpha = float(pn_boundary["polynomial"](fs_value))
            states[row, (alpha_grid >= fold_alpha) & (alpha_grid < pn_alpha)] = 2

    return alpha_grid, fs_grid, states


def draw_curve(ax, curve, name: str) -> None:
    lower = max(FS_MIN, float(curve["fs"][0]))
    upper = min(FS_MAX, float(curve["fs"][-1]))
    fs_dense = np.linspace(lower, upper, 900)
    alpha_dense = curve["interpolator"](fs_dense)
    ax.plot(
        alpha_dense,
        fs_dense,
        color=BOUNDARY_COLORS[name],
        linewidth=3.2,
        linestyle="--",
        dashes=(8, 5),
        dash_capstyle="round",
        path_effects=[
            path_effects.Stroke(linewidth=5.2, foreground="black", alpha=0.50),
            path_effects.Normal(),
        ],
        zorder=7,
    )


def main() -> None:
    curves = load_curves()
    pn_boundary = fit_bimodal_neutral_boundary()
    alpha_grid, fs_grid, states = build_state_map(curves, pn_boundary)

    fig, ax = plt.subplots(figsize=(8.3, 6.6), dpi=180)
    cmap = ListedColormap(
        [
            STATE_COLORS[COEXISTENCE],
            STATE_COLORS[BIASED],
            STATE_COLORS[BIMODAL],
            STATE_COLORS[NEUTRAL],
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.pcolormesh(
        alpha_grid,
        fs_grid,
        states,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )

    draw_curve(ax, curves["lower_saddle"], "Saddle-node")
    draw_curve(ax, curves["upper_saddle"], "Saddle-node")
    draw_curve(ax, curves["supercritical"], "Supercritical pitchfork")
    draw_curve(ax, curves["subcritical"], "Subcritical pitchfork")

    ax.plot(
        TP_ALPHA,
        TP_FS,
        marker="o",
        markersize=8.0,
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=1.5,
        zorder=12,
    )
    ax.text(
        TP_ALPHA + 0.008,
        TP_FS + 0.014,
        "TP",
        fontsize=18,
        fontweight="bold",
        zorder=12,
    )

    boundary_handles = [
        Line2D([0], [0], color=BOUNDARY_COLORS["Saddle-node"], lw=3.0,
               ls="--", dashes=(8, 5), label="Saddle-node"),
        Line2D([0], [0], color=BOUNDARY_COLORS["Supercritical pitchfork"], lw=3.0,
               ls="--", dashes=(8, 5), label="Supercritical pitchfork"),
        Line2D([0], [0], color=BOUNDARY_COLORS["Subcritical pitchfork"], lw=3.0,
               ls="--", dashes=(8, 5), label="Subcritical pitchfork"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="white",
               markeredgecolor="0.15", lw=0, label="Tricritical point"),
    ]
    boundary_legend = fig.legend(
        handles=boundary_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.992),
        ncol=4,
        fontsize=11.4,
        framealpha=0.96,
        borderpad=0.45,
        columnspacing=1.05,
        labelspacing=0.30,
        handlelength=2.05,
        handletextpad=0.55,
    )

    state_handles = [
        Patch(facecolor=STATE_COLORS[label], edgecolor="black", linewidth=1.0, label=label)
        for label in (COEXISTENCE, BIASED, BIMODAL, NEUTRAL)
    ]
    ax.legend(
        handles=state_handles,
        loc="lower right",
        fontsize=13.2,
        framealpha=0.96,
        borderpad=0.50,
        labelspacing=0.35,
    )

    ax.set_xlim(ALPHA_MIN, ALPHA_MAX)
    ax.set_ylim(FS_MIN, FS_MAX)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.35], ["0", "0.1", "0.2", "0.3", "0.35"])
    ax.set_yticks(
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    )
    ax.set_xlabel(r"$\alpha_2$", fontsize=37)
    ax.set_ylabel(r"$f_s$", fontsize=36)
    ax.tick_params(axis="both", labelsize=28, width=1.4, length=7)
    ax.grid(linestyle=":", linewidth=0.8, alpha=0.20, color="black", zorder=1)

    fig.text(0.012, 0.958, "(b)", fontsize=21, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.87)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
