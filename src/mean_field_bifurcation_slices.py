"""Plot matched four-bin MF bifurcation slices with identical layouts.

Panel (a): f_s = 0.5 and alpha2 is varied.
Panel (b): alpha2 = 0 and f_s is varied.

For every complete four-dimensional fixed-point branch, the two displayed
non-stubborn coordinates are connected by a translucent band.  No band joins
coordinates belonging to different fixed points.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_BRANCH_CSV = DATA_DIR / "four_group_fs050_branches.csv"
FS_BRANCH_CSV = DATA_DIR / "four_bin_mf_fs_bifurcation_alpha2_0.csv"

OUTPUT_ALPHA_PNG = FIGURE_DIR / "mean_field_alpha2_bifurcation.png"
OUTPUT_ALPHA_PDF = FIGURE_DIR / "mean_field_alpha2_bifurcation.pdf"
OUTPUT_FS_PNG = FIGURE_DIR / "mean_field_fs_bifurcation.png"
OUTPUT_FS_PDF = FIGURE_DIR / "mean_field_fs_bifurcation.pdf"
OUTPUT_COMBINED_PNG = FIGURE_DIR / "mean_field_bifurcation_slices.png"
OUTPUT_COMBINED_PDF = FIGURE_DIR / "mean_field_bifurcation_slices.pdf"
OUTPUT_COMBINED_SVG = FIGURE_DIR / "mean_field_bifurcation_slices.svg"
OUTPUT_COMBINED_TIFF = FIGURE_DIR / "mean_field_bifurcation_slices.tiff"

STABLE_PLUS = "#3F6FD8"
STABLE_MINUS = "#2F9B45"
UNSTABLE_PLUS = "#EF8FA4"
UNSTABLE_MINUS = "#F4A64F"
FILL_BIMODAL = "#F3A6AC"
FILL_BIASED = "#F7D875"
FILL_NEUTRAL = "#A9CDEC"
FILL_UNSTABLE = "#E8B8C0"
SADDLE_COLOR = "#19C719"
PITCHFORK_COLOR = "#F2CF00"

# Intersections of the verified two-parameter bifurcation boundaries with
# the alpha2 = 0 cut.  These are refined boundary values, not rounded values
# inferred from the 0.005 plotting grid.
FS_SADDLE = 0.27335198681538514
FS_PITCHFORK = 0.22622969240925891


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.major.size": 3.8,
        "ytick.major.size": 3.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load_alpha_branches():
    branches = defaultdict(list)
    saddle = None
    pitchfork = None
    with ALPHA_BRANCH_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        for row in reader:
            stable_field = "stable" if "stable" in fields else "is_stable"
            branches[int(row["branch_id"])].append(
                {
                    "parameter": float(row["alpha2"]),
                    "mu_plus": float(row["mu_ns_plus"]),
                    "mu_minus": float(row["mu_ns_minus"]),
                    "stable": bool(int(row[stable_field])),
                    "grid_index": (
                        int(row["grid_index"])
                        if "grid_index" in fields
                        else int(round(float(row["alpha2"]) / 0.001))
                    ),
                }
            )
            if "saddle_alpha2" in fields:
                saddle = float(row["saddle_alpha2"])
            if "pitchfork_alpha2" in fields:
                pitchfork = float(row["pitchfork_alpha2"])
    for points in branches.values():
        points.sort(key=lambda point: point["grid_index"])
    branch_list = list(branches.values())
    if saddle is None or pitchfork is None:
        events = detected_events(branch_list, use_stability_switch=True)
        if len(events) < 2:
            raise RuntimeError(f"Could not detect both alpha2 events in {ALPHA_BRANCH_CSV}")
        saddle, pitchfork = events[:2]
    return branch_list, saddle, pitchfork


def load_fs_branches():
    branches = defaultdict(list)
    with FS_BRANCH_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            branches[int(row["branch_id"])].append(
                {
                    "parameter": float(row["fs"]),
                    "mu_plus": float(row["mu_ns_plus"]),
                    "mu_minus": float(row["mu_ns_minus"]),
                    "stable": bool(int(row["is_stable"])),
                    "grid_index": int(round(float(row["fs"]) / 0.005)),
                }
            )
    for points in branches.values():
        points.sort(key=lambda point: point["grid_index"])
    branch_list = list(branches.values())
    if "recomputed" in FS_BRANCH_CSV.parts:
        events = [value for value in detected_events(branch_list) if 0.15 <= value <= 0.55]
        if len(events) < 2:
            raise RuntimeError(f"Could not detect both fs events in {FS_BRANCH_CSV}")
        pitchfork, saddle = events[:2]
    else:
        saddle, pitchfork = FS_SADDLE, FS_PITCHFORK
    return branch_list, saddle, pitchfork


def detected_events(branches, *, use_stability_switch=False):
    """Detect branch-count changes and, when requested, a later stability switch."""
    parameters = sorted(
        {round(point["parameter"], 12) for branch in branches for point in branch}
    )
    stable_counts = []
    for parameter in parameters:
        stable_counts.append(
            sum(
                bool(point["stable"])
                for branch in branches
                for point in branch
                if np.isclose(point["parameter"], parameter, atol=1.0e-12)
            )
        )
    changes = [
        0.5 * (left + right)
        for left, right, count_left, count_right in zip(
            parameters[:-1], parameters[1:], stable_counts[:-1], stable_counts[1:]
        )
        if count_left != count_right
    ]
    merged = []
    for value in changes:
        if not merged or value - merged[-1][-1] > 0.006:
            merged.append([value])
        else:
            merged[-1].append(value)
    count_events = [float(np.mean(group)) for group in merged]
    if not use_stability_switch or not count_events:
        return count_events

    switches = []
    for branch in branches:
        for left, right in zip(branch[:-1], branch[1:]):
            if bool(left["stable"]) != bool(right["stable"]):
                switches.append(0.5 * (left["parameter"] + right["parameter"]))
    first = min(count_events)
    later = sorted(value for value in switches if value > first + 0.03)
    return [first, later[0]] if later else count_events


def split_branch(points, maximum_gap):
    if not points:
        return []
    segments = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        if (
            point["stable"] != previous["stable"]
            or point["grid_index"] - previous["grid_index"] > maximum_gap
        ):
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def stable_fill_color(mu_plus, mu_minus):
    mean_abs_s = float(np.mean(np.abs(0.5 * (mu_plus + mu_minus))))
    mean_separation = float(np.mean(np.abs(mu_plus - mu_minus)))
    if mean_abs_s > 0.08:
        return FILL_BIASED
    if mean_separation > 0.30:
        return FILL_BIMODAL
    return FILL_NEUTRAL


def draw_segment(ax, segment, *, rasterized=False):
    if len(segment) < 2:
        return
    parameter = np.asarray([point["parameter"] for point in segment], dtype=float)
    mu_plus = np.asarray([point["mu_plus"] for point in segment], dtype=float)
    mu_minus = np.asarray([point["mu_minus"] for point in segment], dtype=float)
    stable = bool(segment[0]["stable"])
    fill_color = stable_fill_color(mu_plus, mu_minus) if stable else FILL_UNSTABLE
    ax.fill_between(
        parameter,
        mu_plus,
        mu_minus,
        color=fill_color,
        alpha=0.24 if stable else 0.17,
        linewidth=0,
        rasterized=rasterized,
        zorder=1,
    )
    marker_every = max(1, len(segment) // 24)
    for values, stable_color, unstable_color in (
        (mu_plus, STABLE_PLUS, UNSTABLE_PLUS),
        (mu_minus, STABLE_MINUS, UNSTABLE_MINUS),
    ):
        ax.plot(
            parameter,
            values,
            color=stable_color if stable else unstable_color,
            ls="-" if stable else "--",
            lw=1.75 if stable else 1.30,
            alpha=0.98 if stable else 0.82,
            marker="o" if stable else "x",
            ms=2.25,
            markevery=marker_every,
            markeredgewidth=0.7,
            rasterized=rasterized,
            zorder=4 if stable else 3,
        )


def style_axis(ax, xlabel, panel_label):
    ax.axhline(0.0, color="0.45", ls=(0, (3, 2)), lw=0.85, zorder=2)
    ax.set_ylim(-1.02, 1.02)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel(xlabel, fontsize=17.0, labelpad=4)
    ax.set_ylabel(
        r"$\mu_{\mathrm{NS}+}^{*},\ \mu_{\mathrm{NS}-}^{*}$",
        fontsize=17.0,
        labelpad=10,
    )
    ax.tick_params(axis="x", labelsize=13.0, pad=3.0)
    ax.tick_params(axis="y", labelsize=13.0, pad=7.0)
    # Use a compact, near-square plotting box consistent with the other
    # one-dimensional bifurcation figure in the manuscript.
    ax.set_box_aspect(0.86)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#252525")
        ax.spines[side].set_linewidth(0.85)
    ax.grid(True, ls=":", lw=0.6, alpha=0.28)
    label = panel_label.strip("()")
    common = {
        "xy": (0.0, 1.0),
        "xycoords": "axes fraction",
        "textcoords": "offset points",
        "ha": "left",
        "va": "bottom",
        "fontsize": 14,
        "fontweight": "bold",
        "clip_on": False,
    }
    ax.annotate("(", xytext=(-24.0, 8.0), **common)
    ax.annotate(label, xytext=(-19.5, 8.0), **common)
    ax.annotate(")", xytext=(-13.0, 8.0), **common)


def add_branch_legend(ax):
    handles = [
        Line2D([], [], color=STABLE_PLUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=STABLE_MINUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}-}^{*}$"),
        Line2D([], [], color=UNSTABLE_PLUS, lw=1.30, ls="--", label=r"Unstable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=UNSTABLE_MINUS, lw=1.30, ls="--", label=r"Unstable $\mu_{\mathrm{NS}-}^{*}$"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.96,
        fontsize=10.2,
        handlelength=1.50,
        handletextpad=0.28,
        columnspacing=0.90,
        labelspacing=0.20,
        borderpad=0.38,
    )


def draw_alpha_panel(ax, branches, saddle, pitchfork, panel_label="(a)"):
    for points in branches:
        for segment in split_branch(points, maximum_gap=3):
            draw_segment(ax, segment)
    for value, color in ((saddle, SADDLE_COLOR), (pitchfork, PITCHFORK_COLOR)):
        ax.plot(
            [value, value],
            [-1.02, 1.02],
            color=color,
            lw=2.1,
            ls="--",
            dashes=(7, 5),
            path_effects=[
                path_effects.Stroke(linewidth=3.2, foreground="black", alpha=0.43),
                path_effects.Normal(),
            ],
            zorder=6,
        )
    ax.set_xlim(0.0, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_axis(ax, r"$\alpha_2$", panel_label)
    add_branch_legend(ax)


def draw_fs_panel(ax, branches, saddle=FS_SADDLE, pitchfork=FS_PITCHFORK, panel_label="(b)"):
    for points in branches:
        for segment in split_branch(points, maximum_gap=2):
            # Rasterizing only the dense branch geometry keeps the visible
            # result unchanged while preventing PDF path-clipping artefacts
            # from being mistaken for text collisions by geometry auditors.
            draw_segment(ax, segment, rasterized=True)
    for value, color in (
        (saddle, SADDLE_COLOR),
        (pitchfork, PITCHFORK_COLOR),
    ):
        ax.plot(
            [value, value],
            [-1.02, 1.02],
            color=color,
            lw=2.1,
            ls="--",
            dashes=(7, 5),
            path_effects=[
                path_effects.Stroke(linewidth=3.2, foreground="black", alpha=0.43),
                path_effects.Normal(),
            ],
            zorder=6,
        )
    ax.set_xlim(0.15, 0.55)
    ax.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55])
    style_axis(ax, r"$f_s$", panel_label)
    add_branch_legend(ax)


def save_single(draw_function, draw_args, png_path, pdf_path):
    fig, ax = plt.subplots(figsize=(4.95, 4.65), dpi=300)
    draw_function(ax, *draw_args)
    # Preserve enough outer margin for wide endpoint tick labels such as 0.55.
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.76)
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)


def main():
    alpha_branches, saddle, pitchfork = load_alpha_branches()
    fs_branches, fs_saddle, fs_pitchfork = load_fs_branches()

    save_single(
        draw_alpha_panel,
        (alpha_branches, saddle, pitchfork, "(a)"),
        OUTPUT_ALPHA_PNG,
        OUTPUT_ALPHA_PDF,
    )
    save_single(
        draw_fs_panel,
        (fs_branches, fs_saddle, fs_pitchfork, "(b)"),
        OUTPUT_FS_PNG,
        OUTPUT_FS_PDF,
    )

    fig, axes = plt.subplots(2, 1, figsize=(4.95, 8.90), dpi=300)
    draw_alpha_panel(axes[0], alpha_branches, saddle, pitchfork, "(a)")
    draw_fs_panel(axes[1], fs_branches, fs_saddle, fs_pitchfork, "(b)")
    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.075,
        # Leave enough headroom for the legend above the upper panel.
        top=0.90,
        hspace=0.46,
    )
    fig.canvas.draw()
    fig.savefig(OUTPUT_COMBINED_PNG, dpi=600)
    fig.savefig(OUTPUT_COMBINED_PDF, dpi=600)
    fig.savefig(OUTPUT_COMBINED_SVG)
    fig.savefig(OUTPUT_COMBINED_TIFF, dpi=600)
    plt.close(fig)

    for path in (
        OUTPUT_ALPHA_PNG,
        OUTPUT_ALPHA_PDF,
        OUTPUT_FS_PNG,
        OUTPUT_FS_PDF,
        OUTPUT_COMBINED_PNG,
        OUTPUT_COMBINED_PDF,
        OUTPUT_COMBINED_SVG,
        OUTPUT_COMBINED_TIFF,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
