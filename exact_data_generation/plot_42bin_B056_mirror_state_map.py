"""Reclassify the 42-bin stable-state map using B=0.56 and mirror pairs.

Rules requested by the manuscript author:

* B > 0.56: bimodal polarized;
* B <= 0.56 and a distinct stable sign-reflected partner exists:
  biased unimodal;
* B <= 0.56 and no distinct stable sign-reflected partner exists:
  neutral unimodal.

The mirror partner is verified directly among the stable roots at the same
(alpha2, fs); it is not inferred solely from a nonzero signed mean.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

import plot_42bin_direct_peak_fs0p2_0p9_smooth as smooth


HERE = Path(__file__).resolve().parent
SOLUTION_CACHE = HERE / "mf_42bin_full_stability_step0p005.pkl"
B_CUTOFF = float(os.environ.get("MF_B_CUTOFF", "0.56"))
B_TAG = f"{B_CUTOFF:.3f}".replace("0.", "").rstrip("0")
OUTPUT_GRID = HERE / f"mf_42bin_B{B_TAG}_mirror_state_composition_grid.csv"
OUTPUT_SUMMARY = HERE / f"mf_42bin_B{B_TAG}_mirror_state_composition_summary.json"
OUTPUT_PNG = HERE / f"mf_42bin_B{B_TAG}_mirror_fs0p2_0p9_smooth.png"
OUTPUT_PDF = HERE / f"mf_42bin_B{B_TAG}_mirror_fs0p2_0p9_smooth.pdf"

MIRROR_TOL = 2.0e-5

P = "Bimodal polarized"
R = "Biased unimodal"
N = "Neutral unimodal"
STATE_ORDER = (P, R, N)

PR = f"{P} + {R}"
RN = f"{R} + {N}"
PLOT_CLASSES = (PR, R, P, N)
PLOT_COLORS = {
    PR: "#ef9a9a",
    R: "#ffe08a",
    P: "#b39ddb",
    N: "#9ec9ee",
}
PLOT_LABELS = {
    PR: "Coexistence\n(biased unimodal and bimodal)",
    R: "Biased unimodal",
    P: "Bimodal polarized",
    N: "Neutral unimodal",
}


def bimodality_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    m2 = float(np.mean(centered**2))
    if m2 <= 1.0e-18:
        return 0.0
    skewness = float(np.mean(centered**3) / m2**1.5)
    kurtosis = float(np.mean(centered**4) / m2**2)
    return (skewness**2 + 1.0) / kurtosis


def sign_reflection(solution: np.ndarray) -> np.ndarray:
    """Return the exact eta-reflected, opinion-sign-reversed binned state."""
    x = np.asarray(solution, dtype=float)
    if x.size < 4 or (x.size - 2) % 2:
        raise ValueError("Expected two stubborn entries and two equal eta-bin blocks.")
    n = (x.size - 2) // 2
    reflected = np.empty_like(x)
    reflected[0] = -x[1]
    reflected[1] = -x[0]
    reflected[2 : 2 + n] = -x[2 + n : 2 + 2 * n][::-1]
    reflected[2 + n : 2 + 2 * n] = -x[2 : 2 + n][::-1]
    return reflected


def has_distinct_stable_mirror(solution: np.ndarray, roots: list[np.ndarray]) -> bool:
    reflected = sign_reflection(solution)
    if float(np.max(np.abs(reflected - solution))) <= MIRROR_TOL:
        return False
    return any(
        float(np.max(np.abs(reflected - candidate))) <= MIRROR_TOL
        for candidate in roots
    )


def classify_solution(solution: np.ndarray, roots: list[np.ndarray]) -> str:
    b_value = bimodality_coefficient(solution[2:])
    if b_value > B_CUTOFF:
        return P
    return R if has_distinct_stable_mirror(solution, roots) else N


def classify_and_write_grid() -> dict:
    with SOLUTION_CACHE.open("rb") as stream:
        scan = pickle.load(stream)

    alpha = np.asarray(scan["alpha_values"], dtype=float)
    fs = np.asarray(scan["fs_values"], dtype=float)
    solutions = scan["solutions"]

    composition_counts: Counter[tuple[str, ...]] = Counter()
    root_type_counts: Counter[str] = Counter()
    rows = []
    missing_distinct_mirrors = 0

    for fs_index, fs_value in enumerate(fs):
        for alpha_index, alpha_value in enumerate(alpha):
            roots = [
                np.asarray(root, dtype=float)
                for root in solutions[fs_index][alpha_index]
            ]
            labels = []
            for root in roots:
                b_value = bimodality_coefficient(root[2:])
                label = classify_solution(root, roots)
                labels.append(label)
                root_type_counts[label] += 1
                if b_value <= B_CUTOFF:
                    reflected = sign_reflection(root)
                    distinct = np.max(np.abs(reflected - root)) > MIRROR_TOL
                    if distinct and label == N:
                        missing_distinct_mirrors += 1

            present = set(labels)
            composition = tuple(state for state in STATE_ORDER if state in present)
            composition_counts[composition] += 1
            rows.append(
                {
                    "alpha2": float(alpha_value),
                    "fs": float(fs_value),
                    "composition": " + ".join(composition),
                }
            )

    with OUTPUT_GRID.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["alpha2", "fs", "composition"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "B_cutoff": B_CUTOFF,
        "mirror_tolerance": MIRROR_TOL,
        "root_type_counts": dict(root_type_counts),
        "composition_counts": {
            " + ".join(key): value for key, value in composition_counts.items()
        },
        "distinct_low_B_roots_without_stable_mirror": missing_distinct_mirrors,
        "grid_shape": [int(fs.size), int(alpha.size)],
        "alpha_range": [float(alpha.min()), float(alpha.max())],
        "fs_range": [float(fs.min()), float(fs.max())],
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def read_classification_grid():
    rows = []
    with OUTPUT_GRID.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    alpha = np.asarray(sorted({float(row["alpha2"]) for row in rows}))
    fs = np.asarray(sorted({float(row["fs"]) for row in rows}))
    lookup = {
        (float(row["alpha2"]), float(row["fs"])): row["composition"]
        for row in rows
    }
    code_for = {label: index for index, label in enumerate(PLOT_CLASSES)}
    codes = np.full((fs.size, alpha.size), -1, dtype=np.int8)
    for i, fs_value in enumerate(fs):
        for j, alpha_value in enumerate(alpha):
            composition = lookup[(float(alpha_value), float(fs_value))]
            if composition in code_for:
                codes[i, j] = code_for[composition]

    # R+N occurs only at isolated grid points lying numerically on the
    # pitchfork boundary.  It is a zero-area boundary composition, not a
    # fifth macroscopic region.  Assign those pixels to their nearest open
    # region before constructing smooth signed-distance boundaries.
    unresolved = codes < 0
    if np.any(unresolved):
        nearest = distance_transform_edt(
            unresolved,
            return_distances=False,
            return_indices=True,
        )
        codes[unresolved] = codes[tuple(nearest[:, unresolved])]
    return alpha, fs, codes


def smooth_categorical_grid(alpha, fs, codes):
    alpha_dense = np.linspace(smooth.ALPHA_MIN, smooth.ALPHA_MAX, 1801)
    fs_dense = np.linspace(smooth.FS_MIN, smooth.FS_MAX, 1801)
    aa, ff = np.meshgrid(alpha_dense, fs_dense)
    points = np.column_stack((ff.ravel(), aa.ravel()))

    scores = []
    for code in range(len(PLOT_CLASSES)):
        mask = codes == code
        signed_distance = distance_transform_edt(mask) - distance_transform_edt(~mask)
        # Smooth only the signed-distance representation, not the raw class
        # labels.  This removes the visible 0.005-grid staircase while keeping
        # every macroscopic region and junction found by the classification.
        signed_distance = gaussian_filter(signed_distance, sigma=0.9, mode="nearest")
        interpolator = RegularGridInterpolator(
            (fs, alpha),
            signed_distance,
            method="linear",
            bounds_error=False,
            fill_value=-1.0e6,
        )
        scores.append(interpolator(points).reshape(ff.shape))
    dense_codes = np.argmax(np.stack(scores, axis=0), axis=0).astype(np.int8)
    return alpha_dense, fs_dense, dense_codes


def plot_new_map():
    alpha, fs, codes = read_classification_grid()
    alpha_dense, fs_dense, dense_codes = smooth_categorical_grid(alpha, fs, codes)
    curves = smooth.load_curves()

    cmap = ListedColormap([PLOT_COLORS[label] for label in PLOT_CLASSES])
    norm = BoundaryNorm(np.arange(len(PLOT_CLASSES) + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(8.3, 6.6), dpi=180)
    ax.pcolormesh(
        alpha_dense,
        fs_dense,
        dense_codes,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )

    smooth.draw_curve(ax, curves["lower_saddle"], "Saddle-node")
    smooth.draw_curve(ax, curves["upper_saddle"], "Saddle-node")
    smooth.draw_curve(ax, curves["supercritical"], "Supercritical pitchfork")
    smooth.draw_curve(ax, curves["subcritical"], "Subcritical pitchfork")
    ax.plot(
        smooth.TP_ALPHA,
        smooth.TP_FS,
        marker="o",
        markersize=8.0,
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=1.5,
        zorder=12,
    )
    ax.text(
        smooth.TP_ALPHA + 0.008,
        smooth.TP_FS + 0.014,
        "TP",
        fontsize=18,
        fontweight="bold",
        zorder=12,
    )

    boundary_handles = [
        Line2D([0], [0], color=smooth.BOUNDARY_COLORS["Saddle-node"], lw=3.0,
               ls="--", dashes=(8, 5), label="Saddle-node"),
        Line2D([0], [0], color=smooth.BOUNDARY_COLORS["Supercritical pitchfork"], lw=3.0,
               ls="--", dashes=(8, 5), label="Supercritical pitchfork"),
        Line2D([0], [0], color=smooth.BOUNDARY_COLORS["Subcritical pitchfork"], lw=3.0,
               ls="--", dashes=(8, 5), label="Subcritical pitchfork"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="white",
               markeredgecolor="0.15", lw=0, label="Tricritical point"),
    ]
    fig.legend(
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
        Patch(
            facecolor=PLOT_COLORS[label],
            edgecolor="black",
            linewidth=1.0,
            label=PLOT_LABELS[label],
        )
        for label in PLOT_CLASSES
    ]
    ax.legend(
        handles=state_handles,
        loc="lower right",
        fontsize=13.2,
        framealpha=0.96,
        borderpad=0.50,
        labelspacing=0.35,
    )

    ax.set_xlim(smooth.ALPHA_MIN, smooth.ALPHA_MAX)
    ax.set_ylim(smooth.FS_MIN, smooth.FS_MAX)
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


def main() -> None:
    summary = classify_and_write_grid()

    # Reuse the verified continuous bifurcation curves, but derive every
    # colored state boundary from the new classification grid itself.
    plot_new_map()

    print(f"Saved {OUTPUT_GRID}")
    print(f"Saved {OUTPUT_SUMMARY}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
