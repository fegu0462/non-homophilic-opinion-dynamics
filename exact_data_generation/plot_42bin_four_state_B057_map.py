"""Four-state 42-bin map separating symmetric/asymmetric bimodality.

Classification of every stable fixed point:

* B > 0.57 and self-reflected: symmetric bimodal;
* B > 0.57 and a distinct reflected stable partner exists: asymmetric bimodal;
* B <= 0.57 and a distinct reflected stable partner exists: biased unimodal;
* B <= 0.57 and self-reflected: neutral unimodal.

The plotted background records the major stable-state compositions.  Two
zero-area compositions occurring only on the sampled pitchfork boundary are
assigned to their nearest open region before smoothing.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

import plot_42bin_B056_mirror_state_map as helper


HERE = Path(__file__).resolve().parent
CACHE = HERE / "mf_42bin_full_stability_step0p005.pkl"
GRID_CSV = HERE / "mf_42bin_four_state_B57_composition_grid.csv"
SUMMARY_JSON = HERE / "mf_42bin_four_state_B57_summary.json"
OUTPUT_PNG = HERE / "mf_42bin_four_state_B57_fs0p2_0p9_smooth.png"
OUTPUT_PDF = HERE / "mf_42bin_four_state_B57_fs0p2_0p9_smooth.pdf"

B_CUTOFF = 0.57
MIRROR_TOL = 2.0e-5

SB = "Symmetric bimodal"
AB = "Asymmetric bimodal"
BU = "Biased unimodal"
NU = "Neutral unimodal"
ROOT_ORDER = (SB, AB, BU, NU)

TRIPLE = "Biased unimodal + Symmetric bimodal + Asymmetric bimodal"
SB_AB = "Symmetric bimodal + Asymmetric bimodal"
PLOT_CLASSES = (TRIPLE, BU, SB_AB, SB, AB, NU)
COLORS = {
    TRIPLE: "#ef9a9a",
    BU: "#ffe08a",
    SB_AB: "#8475c1",
    SB: "#b39ddb",
    AB: "#58bdb8",
    NU: "#9ec9ee",
}
LABELS = {
    TRIPLE: "Coexistence\n(biased unimodal + both bimodal types)",
    BU: "Biased unimodal",
    SB_AB: "Coexistence\n(symmetric + asymmetric bimodal)",
    SB: "Symmetric bimodal",
    AB: "Asymmetric bimodal",
    NU: "Neutral unimodal",
}


def classify_root(root: np.ndarray, roots: list[np.ndarray]) -> str:
    b_value = helper.bimodality_coefficient(root[2:])
    reflected = helper.sign_reflection(root)
    distinct = float(np.max(np.abs(reflected - root))) > MIRROR_TOL
    has_partner = distinct and any(
        float(np.max(np.abs(reflected - candidate))) <= MIRROR_TOL
        for candidate in roots
    )
    if b_value > B_CUTOFF:
        return AB if has_partner else SB
    return BU if has_partner else NU


def compute_grid():
    with CACHE.open("rb") as stream:
        scan = pickle.load(stream)
    alpha = np.asarray(scan["alpha_values"], dtype=float)
    fs = np.asarray(scan["fs_values"], dtype=float)
    solutions = scan["solutions"]

    root_counts: Counter[str] = Counter()
    composition_counts: Counter[tuple[str, ...]] = Counter()
    composition_grid: list[list[tuple[str, ...]]] = []
    rows = []
    for i, fs_value in enumerate(fs):
        grid_row = []
        for j, alpha_value in enumerate(alpha):
            roots = [np.asarray(root, dtype=float) for root in solutions[i][j]]
            labels = [classify_root(root, roots) for root in roots]
            root_counts.update(labels)
            present = set(labels)
            composition = tuple(label for label in ROOT_ORDER if label in present)
            composition_counts[composition] += 1
            grid_row.append(composition)
            rows.append(
                {
                    "alpha2": float(alpha_value),
                    "fs": float(fs_value),
                    "composition": " + ".join(composition),
                }
            )
        composition_grid.append(grid_row)

    with GRID_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["alpha2", "fs", "composition"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "B_cutoff": B_CUTOFF,
        "mirror_tolerance": MIRROR_TOL,
        "root_type_counts": dict(root_counts),
        "composition_counts": {
            " + ".join(key): value for key, value in composition_counts.items()
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    return alpha, fs, composition_grid, summary


def major_composition_codes(compositions):
    code_for = {label: index for index, label in enumerate(PLOT_CLASSES)}
    codes = np.full((len(compositions), len(compositions[0])), -1, dtype=np.int8)
    for i, row in enumerate(compositions):
        for j, composition in enumerate(row):
            present = set(composition)
            if present == {SB, AB, BU}:
                label = TRIPLE
            elif present == {BU}:
                label = BU
            elif present == {SB, AB}:
                label = SB_AB
            elif present == {SB}:
                label = SB
            elif present == {AB}:
                label = AB
            elif present == {NU}:
                label = NU
            else:
                continue
            codes[i, j] = code_for[label]

    unresolved = codes < 0
    if np.any(unresolved):
        nearest = distance_transform_edt(
            unresolved,
            return_distances=False,
            return_indices=True,
        )
        codes[unresolved] = codes[tuple(nearest[:, unresolved])]
    return codes


def smooth_codes(alpha, fs, codes):
    alpha_dense = np.linspace(helper.smooth.ALPHA_MIN, helper.smooth.ALPHA_MAX, 1801)
    fs_dense = np.linspace(helper.smooth.FS_MIN, helper.smooth.FS_MAX, 1801)
    aa, ff = np.meshgrid(alpha_dense, fs_dense)
    query = np.column_stack((ff.ravel(), aa.ravel()))
    scores = []
    for code in range(len(PLOT_CLASSES)):
        mask = codes == code
        signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
        # The classification scan has spacing 0.005 in both parameters.
        # Smooth the signed-distance fields over roughly two grid cells so
        # threshold boundaries are displayed as continuous curves rather than
        # as staircase artifacts of the scan resolution.
        signed = gaussian_filter(signed, sigma=1.3, mode="nearest")
        interp = RegularGridInterpolator(
            (fs, alpha), signed, method="linear", bounds_error=False, fill_value=-1e6
        )
        scores.append(interp(query).reshape(ff.shape))
    dense = np.argmax(np.stack(scores), axis=0).astype(np.int8)
    return alpha_dense, fs_dense, dense


def plot_map(alpha, fs, compositions):
    codes = major_composition_codes(compositions)
    alpha_dense, fs_dense, dense = smooth_codes(alpha, fs, codes)
    curves = helper.smooth.load_curves()

    fig, ax = plt.subplots(figsize=(8.7, 6.6), dpi=180)
    cmap = ListedColormap([COLORS[label] for label in PLOT_CLASSES])
    norm = BoundaryNorm(np.arange(len(PLOT_CLASSES) + 1) - 0.5, cmap.N)
    ax.pcolormesh(
        alpha_dense,
        fs_dense,
        dense,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )

    curves_to_draw = (
        ("lower_saddle", "Saddle-node"),
        ("upper_saddle", "Saddle-node"),
        ("supercritical", "Supercritical pitchfork"),
        ("subcritical", "Subcritical pitchfork"),
    )
    for curve_key, label in curves_to_draw:
        helper.smooth.draw_curve(ax, curves[curve_key], label)
    ax.plot(
        helper.smooth.TP_ALPHA,
        helper.smooth.TP_FS,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=1.5,
        zorder=12,
    )
    ax.text(
        helper.smooth.TP_ALPHA + 0.008,
        helper.smooth.TP_FS + 0.014,
        "TP",
        fontsize=18,
        fontweight="bold",
        zorder=12,
    )

    boundary_handles = [
        Line2D([0], [0], color=helper.smooth.BOUNDARY_COLORS["Saddle-node"], lw=3,
               ls="--", dashes=(8, 5), label="Saddle-node"),
        Line2D([0], [0], color=helper.smooth.BOUNDARY_COLORS["Supercritical pitchfork"], lw=3,
               ls="--", dashes=(8, 5), label="Supercritical pitchfork"),
        Line2D([0], [0], color=helper.smooth.BOUNDARY_COLORS["Subcritical pitchfork"], lw=3,
               ls="--", dashes=(8, 5), label="Subcritical pitchfork"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="white",
               markeredgecolor="0.15", lw=0, label="Tricritical point"),
    ]
    fig.legend(
        handles=boundary_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.992),
        ncol=4,
        fontsize=11.0,
        framealpha=0.96,
        columnspacing=1.0,
        handlelength=2.0,
    )
    state_handles = [
        Patch(facecolor=COLORS[label], edgecolor="black", linewidth=1, label=LABELS[label])
        for label in PLOT_CLASSES
    ]
    ax.legend(
        handles=state_handles,
        loc="lower right",
        fontsize=9.5,
        framealpha=0.96,
        borderpad=0.45,
        labelspacing=0.30,
    )

    ax.set_xlim(helper.smooth.ALPHA_MIN, helper.smooth.ALPHA_MAX)
    ax.set_ylim(helper.smooth.FS_MIN, helper.smooth.FS_MAX)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.35], ["0", "0.1", "0.2", "0.3", "0.35"])
    ax.set_yticks(
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    )
    ax.set_xlabel(r"$\alpha_2$", fontsize=37)
    ax.set_ylabel(r"$f_s$", fontsize=36)
    ax.tick_params(axis="both", labelsize=28, width=1.4, length=7)
    ax.grid(ls=":", lw=0.8, alpha=0.20, color="black", zorder=1)
    fig.text(0.012, 0.958, "(b)", fontsize=21, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.87)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    alpha, fs, compositions, summary = compute_grid()
    plot_map(alpha, fs, compositions)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {GRID_CSV}")
    print(f"Saved {SUMMARY_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
