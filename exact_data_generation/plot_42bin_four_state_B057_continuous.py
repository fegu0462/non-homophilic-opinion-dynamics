"""Continuous-boundary four-state map for the 42-bin mean-field model.

This revision distinguishes dynamical bifurcations from the configurable
bimodality-coefficient classification threshold.  In particular, the previously unmarked
pink--dark-purple boundary above the tricritical point is recomputed as the
saddle-node at which the biased-unimodal stable pair disappears.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, root
from scipy.stats import kurtosis, skew


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model
import plot_42bin_four_state_B057_map as base
import trace_42bin_all_detected_folds as fold_trace
from plot_42bin_B056_mirror_state_map import sign_reflection


FS_TP = base.helper.smooth.TP_FS
ALPHA_TP = base.helper.smooth.TP_ALPHA
FS_HIGH = np.round(np.arange(0.80, 0.9001, 0.005), 12)
FS_B_LOW = np.round(np.arange(FS_TP - 0.0025, 0.4999, -0.0025), 12)
B_CUTOFF = float(os.environ.get("MF_B_CUTOFF", "0.57"))
B_TAG = f"{B_CUTOFF:.2f}".replace(".", "")
BIMODALITY_MODE = os.environ.get("BIMODALITY_MODE", "population").strip().lower()
if BIMODALITY_MODE not in {"population", "sample_corrected"}:
    raise ValueError(f"Unsupported BIMODALITY_MODE={BIMODALITY_MODE!r}")
MODE_SUFFIX = "_sample_corrected" if BIMODALITY_MODE == "sample_corrected" else ""
FINITE_SAMPLE_N = 40
MERGE_BIMODAL = os.environ.get("MF_MERGE_BIMODAL", "0") == "1"
STABILITY_MARGIN = 2.0e-6

CACHE = HERE / "mf_42bin_full_stability_step0p005.pkl"
ALL_FOLDS = HERE / "bifurcation_boundaries_42bin_eta_resolved_folds.csv"
PSEUDOARCLENGTH = HERE / "bifurcation_boundaries_42D_pseudoarclength.csv"
if MERGE_BIMODAL:
    OUTPUT_BOUNDARIES = HERE / f"mf_42bin_three_state_B{B_TAG}{MODE_SUFFIX}_continuous_boundaries.csv"
    OUTPUT_SUMMARY = HERE / f"mf_42bin_three_state_B{B_TAG}{MODE_SUFFIX}_continuous_summary.json"
    OUTPUT_PNG = HERE / f"mf_42bin_three_state_B{B_TAG}{MODE_SUFFIX}_continuous.png"
    OUTPUT_PDF = HERE / f"mf_42bin_three_state_B{B_TAG}{MODE_SUFFIX}_continuous.pdf"
else:
    OUTPUT_BOUNDARIES = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_continuous_boundaries.csv"
    OUTPUT_SUMMARY = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_continuous_summary.json"
    OUTPUT_PNG = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_continuous.png"
    OUTPUT_PDF = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_continuous.pdf"

SADDLE_COLOR = base.helper.smooth.BOUNDARY_COLORS["Saddle-node"]
SUPER_COLOR = base.helper.smooth.BOUNDARY_COLORS["Supercritical pitchfork"]
SUB_COLOR = base.helper.smooth.BOUNDARY_COLORS["Subcritical pitchfork"]
CLASSIFICATION_COLOR = "#555555"
# The B contour and the supercritical pitchfork meet exactly.  Starting the
# rendered gray stroke a few fs units past that mathematical intersection
# prevents their finite line widths from visually lying on top of each other.
CLASSIFICATION_VISUAL_GAP_FS = 0.003


def finite_sample_corrected_bimodality(values: np.ndarray) -> float:
    """Pfister/Sarle BC with bias-corrected moments for 40 NS bins."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n != FINITE_SAMPLE_N:
        raise ValueError(f"Expected {FINITE_SAMPLE_N} non-stubborn bins, received {n}.")
    if float(np.std(values)) <= 1.0e-12:
        return 0.0
    sample_skewness = float(skew(values, bias=False))
    excess_kurtosis = float(kurtosis(values, fisher=True, bias=False))
    finite_n_term = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = excess_kurtosis + finite_n_term
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float((sample_skewness**2 + 1.0) / denominator)


def corrected_observables(x: np.ndarray, pop) -> tuple[float, float]:
    values = np.asarray(x, dtype=float)[2:]
    weights = np.asarray(pop.weights[2:], dtype=float)
    weights = weights / np.sum(weights)
    mean = float(np.sum(weights * values))
    return finite_sample_corrected_bimodality(values), mean


if BIMODALITY_MODE == "sample_corrected":
    base.helper.bimodality_coefficient = finite_sample_corrected_bimodality
    model.observables = corrected_observables

MERGED_COEXISTENCE = "Coexistence: biased unimodal and bimodal"
MERGED_BIMODAL = "Bimodal"
MERGED_PLOT_CLASSES = (MERGED_COEXISTENCE, base.BU, MERGED_BIMODAL, base.NU)
MERGED_COLORS = {
    MERGED_COEXISTENCE: base.COLORS[base.TRIPLE],
    base.BU: base.COLORS[base.BU],
    MERGED_BIMODAL: base.COLORS[base.SB],
    base.NU: base.COLORS[base.NU],
}
MERGED_LABELS = {
    MERGED_COEXISTENCE: "Coexistence\n(biased unimodal and bimodal)",
    base.BU: "Biased unimodal",
    MERGED_BIMODAL: "Bimodal",
    base.NU: "Neutral unimodal",
}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def unique_pchip(points):
    points = sorted(points, key=lambda item: item[1])
    fs = np.asarray([item[1] for item in points], dtype=float)
    alpha = np.asarray([item[0] for item in points], dtype=float)
    rounded = np.round(fs, 11)
    unique_fs = np.unique(rounded)
    unique_alpha = np.asarray([np.mean(alpha[rounded == value]) for value in unique_fs])
    return {
        "fs": unique_fs,
        "alpha": unique_alpha,
        "interpolator": PchipInterpolator(unique_fs, unique_alpha, extrapolate=False),
    }


def alpha_zero_entry(rows):
    """Interpolate a continuation curve's entry through alpha2=0."""
    negative = [row for row in rows if float(row["alpha2"]) <= 0.0]
    positive = [row for row in rows if float(row["alpha2"]) >= 0.0]
    if not negative or not positive:
        return None
    left = min(negative, key=lambda row: abs(float(row["alpha2"])))
    right = min(positive, key=lambda row: abs(float(row["alpha2"])))
    a0, f0 = float(left["alpha2"]), float(left["fs"])
    a1, f1 = float(right["alpha2"]), float(right["fs"])
    if abs(a1 - a0) < 1.0e-14:
        return 0.0, 0.5 * (f0 + f1)
    return 0.0, f0 + (0.0 - a0) * (f1 - f0) / (a1 - a0)


def recomputed_yellow_curves():
    """Read the just-rerun pseudo-arclength solution for all yellow edges."""
    rows = read_csv(PSEUDOARCLENGTH)
    saddle_rows = [row for row in rows if row["event"] == "Saddle-node"]
    pitch_rows = [row for row in rows if row["event"] == "Pitchfork"]
    apex = int(np.argmax([float(row["fs"]) for row in pitch_rows]))
    super_rows = pitch_rows[: apex + 1]
    sub_rows = pitch_rows[apex:]

    def stable(row):
        return float(row["largest_noncritical_multiplier_modulus"]) < 1.0 - STABILITY_MARGIN

    lower_selected = [
        row
        for row in saddle_rows
        if stable(row)
        and 0.0 <= float(row["alpha2"]) <= 0.4
        and base.helper.smooth.FS_MIN <= float(row["fs"]) <= base.helper.smooth.LOWER_JUNCTION_FS + 0.004
    ]
    super_selected = [
        row
        for row in super_rows
        if stable(row)
        and 0.0 <= float(row["alpha2"]) <= 0.4
        and base.helper.smooth.FS_MIN <= float(row["fs"]) <= FS_TP + 0.002
    ]
    sub_selected = [
        row
        for row in sub_rows
        if stable(row)
        and 0.0 <= float(row["alpha2"]) <= 0.4
        and base.helper.smooth.LOWER_JUNCTION_FS - 0.004 <= float(row["fs"]) <= FS_TP + 0.002
    ]

    lower_points = [(float(row["alpha2"]), float(row["fs"])) for row in lower_selected]
    super_points = [(float(row["alpha2"]), float(row["fs"])) for row in super_selected]
    sub_points = [(float(row["alpha2"]), float(row["fs"])) for row in sub_selected]

    lower_entry = alpha_zero_entry([row for row in saddle_rows if stable(row)])
    super_entry = alpha_zero_entry([row for row in super_rows if stable(row)])
    if lower_entry is not None and lower_entry[1] >= base.helper.smooth.FS_MIN - 0.01:
        lower_points.append(lower_entry)
    if super_entry is not None and super_entry[1] >= base.helper.smooth.FS_MIN - 0.01:
        super_points.append(super_entry)

    # Both pitchfork pieces terminate at the independently refined TP.
    super_points.append((ALPHA_TP, FS_TP))
    sub_points.append((ALPHA_TP, FS_TP))

    junction_fs = base.helper.smooth.LOWER_JUNCTION_FS
    # The two rerun traces stop on opposite sides of the codimension-two
    # junction by only ~1e-4 in fs.  Join their nearest continued points at
    # the independently refined junction fs, rather than extrapolating either
    # interpolator outside its solved interval.
    lower_tip = max(lower_points, key=lambda point: point[1])
    sub_tip = min(sub_points, key=lambda point: abs(point[1] - junction_fs))
    junction_alpha = 0.5 * (float(lower_tip[0]) + float(sub_tip[0]))
    lower_points.append((junction_alpha, junction_fs))
    sub_points.append((junction_alpha, junction_fs))

    curves = {
        "lower_saddle": unique_pchip(lower_points),
        "supercritical": unique_pchip(super_points),
        "subcritical": unique_pchip(sub_points),
    }

    output_rows = []
    for boundary, selected in (
        ("Saddle-node: yellow left boundary", lower_selected),
        ("Supercritical pitchfork: yellow right boundary", super_selected),
        ("Subcritical pitchfork: yellow upper-left boundary", sub_selected),
    ):
        for row in selected:
            output_rows.append(
                {
                    "boundary": boundary,
                    "alpha2": float(row["alpha2"]),
                    "fs": float(row["fs"]),
                    "B": float(row["B"]),
                    "S": float(row["S"]),
                    "residual": float(row["fixed_residual"]),
                    "critical_multiplier": 1.0 + float(row["sector_zero_eigenvalue"]),
                }
            )
    return curves, output_rows


def load_cache():
    with CACHE.open("rb") as stream:
        return pickle.load(stream)


def positive_bu_seed(scan, fs_value):
    fs_index = int(np.argmin(np.abs(np.asarray(scan["fs_values"]) - fs_value)))
    alpha_index = int(np.argmin(np.abs(np.asarray(scan["alpha_values"]) - 0.0)))
    candidates = []
    pop = model.make_population(float(scan["fs_values"][fs_index]))
    for candidate in scan["solutions"][fs_index][alpha_index]:
        x = np.asarray(candidate, dtype=float)
        b_value, s_value = model.observables(x, pop)
        if s_value > 0.10 and b_value < B_CUTOFF:
            candidates.append((s_value, x))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one positive BU seed at fs={fs_value:.3f}; found {len(candidates)}."
        )
    return candidates[0][1]


def recompute_upper_bu_fold(scan):
    """Independently recompute the pink--purple BU saddle-node."""
    points = [(ALPHA_TP, FS_TP)]
    rows = [
        {
            "boundary": "Saddle-node: biased-unimodal pair disappears",
            "alpha2": ALPHA_TP,
            "fs": FS_TP,
            "B": "",
            "S": "",
            "residual": "",
            "critical_multiplier": "",
        }
    ]
    for fs_value in FS_HIGH:
        pop = model.make_population(float(fs_value))
        seed = positive_bu_seed(scan, float(fs_value))
        near_x, near_alpha, _ = fold_trace.track_to_fold_seed(seed, pop)
        fold_x, fold_alpha, residual = fold_trace.refine_fixed_fs_fold(
            near_x, near_alpha, float(fs_value)
        )
        b_value, s_value = model.observables(fold_x, pop)
        multiplier = fold_trace.multiplier_radius(fold_x, fold_alpha, pop)
        if residual > 2.0e-7 or abs(multiplier - 1.0) > 2.0e-5:
            raise RuntimeError(
                f"BU fold verification failed at fs={fs_value:.3f}: "
                f"residual={residual:.3e}, multiplier={multiplier:.9f}."
            )
        points.append((float(fold_alpha), float(fs_value)))
        rows.append(
            {
                "boundary": "Saddle-node: biased-unimodal pair disappears",
                "alpha2": float(fold_alpha),
                "fs": float(fs_value),
                "B": float(b_value),
                "S": float(s_value),
                "residual": residual,
                "critical_multiplier": multiplier,
            }
        )
    return unique_pchip(points), rows


def continued_last_asymmetric_fold():
    """Load the continued fold at which the last asymmetric pair vanishes."""
    rows = [row for row in read_csv(ALL_FOLDS) if int(row["curve_id"]) == 2]
    rows.sort(key=lambda row: int(row["curve_order"]))
    fs_all = np.asarray([float(row["fs"]) for row in rows])
    turn = int(np.argmin(fs_all))
    selected = rows[: turn + 1]
    points = [(ALPHA_TP, FS_TP)]
    output = []
    for row in selected:
        alpha2 = float(row["alpha2"])
        fs = float(row["fs"])
        noncritical = float(row["largest_noncritical_multiplier_modulus"])
        if fs < FS_TP - 5.0e-4 or fs > 0.900001 or alpha2 < 0.0:
            continue
        if noncritical >= 1.0 - STABILITY_MARGIN:
            continue
        points.append((alpha2, fs))
        output.append(
            {
                "boundary": "Saddle-node: last asymmetric-bimodal pair disappears",
                "alpha2": alpha2,
                "fs": fs,
                "B": float(row["B"]),
                "S": float(row["S"]),
                "residual": float(row["fixed_residual"]),
                "critical_multiplier": float(row["critical_multiplier_real"]),
            }
        )
    return unique_pchip(points), output


def symmetric_seed_near_threshold(scan, fs_value):
    fs_index = int(np.argmin(np.abs(np.asarray(scan["fs_values"]) - fs_value)))
    pop = model.make_population(float(scan["fs_values"][fs_index]))
    best = None
    for alpha_index, alpha2 in enumerate(scan["alpha_values"]):
        for candidate in scan["solutions"][fs_index][alpha_index]:
            x = np.asarray(candidate, dtype=float)
            if float(np.max(np.abs(sign_reflection(x) - x))) > 2.0e-5:
                continue
            b_value, _ = model.observables(x, pop)
            score = abs(b_value - B_CUTOFF)
            if best is None or score < best[0]:
                best = (score, float(alpha2), x)
    if best is None:
        raise RuntimeError(f"No symmetric threshold seed at fs={fs_value:.3f}.")
    return best[1], best[2]


def solve_symmetric_B_threshold(fs_value, u0, alpha0):
    pop = model.make_population(float(fs_value))
    initial = np.concatenate((np.asarray(u0, dtype=float), [float(alpha0)]))

    def equations(z):
        u = z[:-1]
        alpha2 = float(z[-1])
        fixed = model.symmetric_system(u, alpha2, pop)[0]
        x = model.full_from_symmetric(u, pop)
        b_value, _ = model.observables(x, pop)
        return np.concatenate((fixed, [b_value - B_CUTOFF]))

    answer = root(
        equations,
        initial,
        method="hybr",
        options={"xtol": 2.0e-10, "maxfev": 9000},
    )
    residual = float(np.max(np.abs(equations(answer.x))))
    if residual > 2.0e-7:
        raise RuntimeError(
            f"B={B_CUTOFF} threshold failed at fs={fs_value:.6f}; "
            f"residual={residual:.3e}."
        )
    return answer.x[:-1], float(answer.x[-1]), residual


def recompute_symmetric_B_boundary(scan):
    points = []
    rows = []
    solutions = {}
    for fs_value in FS_HIGH:
        alpha0, x0 = symmetric_seed_near_threshold(scan, float(fs_value))
        pop = model.make_population(float(fs_value))
        u, alpha2, residual = solve_symmetric_B_threshold(
            float(fs_value), x0[pop.select_symmetric], alpha0
        )
        solutions[float(fs_value)] = (u, alpha2)
        points.append((alpha2, float(fs_value)))
        rows.append(
            {
                "boundary": f"Classification only: symmetric B={B_CUTOFF:.2f}",
                "alpha2": alpha2,
                "fs": float(fs_value),
                "B": B_CUTOFF,
                "S": 0.0,
                "residual": residual,
                "critical_multiplier": "",
            }
        )

    # Continue the .800 solution a short distance down to the precise TP fs.
    u0, alpha0 = solutions[0.8]
    u_tp, alpha_tp, residual_tp = solve_symmetric_B_threshold(FS_TP, u0, alpha0)
    points.append((alpha_tp, FS_TP))
    rows.append(
        {
            "boundary": f"Classification only: symmetric B={B_CUTOFF:.2f}",
            "alpha2": alpha_tp,
            "fs": FS_TP,
            "B": B_CUTOFF,
            "S": 0.0,
            "residual": residual_tp,
            "critical_multiplier": "",
        }
    )

    # Continue the same B-threshold solution downward as well.  Previously the
    # curve stopped at the TP, which left the lower, visible part of the
    # bimodal--neutral classification boundary uncomputed.
    u_previous = u_tp
    alpha_previous = alpha_tp
    for fs_value in FS_B_LOW:
        u_previous, alpha_previous, residual = solve_symmetric_B_threshold(
            float(fs_value), u_previous, alpha_previous
        )
        points.append((alpha_previous, float(fs_value)))
        rows.append(
            {
                "boundary": f"Classification only: symmetric B={B_CUTOFF:.2f}",
                "alpha2": alpha_previous,
                "fs": float(fs_value),
                "B": B_CUTOFF,
                "S": 0.0,
                "residual": residual,
                "critical_multiplier": "",
            }
        )
    return unique_pchip(points), rows


def first_curve_intersection(curve_a, curve_b):
    """Return the lowest-fs intersection of two alpha(fs) curves."""
    lower = max(float(curve_a["fs"][0]), float(curve_b["fs"][0]))
    upper = min(float(curve_a["fs"][-1]), float(curve_b["fs"][-1]))
    fs_probe = np.linspace(lower, upper, 3000)
    difference = curve_a["interpolator"](fs_probe) - curve_b["interpolator"](fs_probe)
    for index in range(len(fs_probe) - 1):
        if not (np.isfinite(difference[index]) and np.isfinite(difference[index + 1])):
            continue
        if difference[index] == 0.0:
            return float(fs_probe[index])
        if difference[index] * difference[index + 1] < 0.0:
            return float(
                brentq(
                    lambda value: float(curve_a["interpolator"](value))
                    - float(curve_b["interpolator"](value)),
                    float(fs_probe[index]),
                    float(fs_probe[index + 1]),
                )
            )
    raise RuntimeError("The two continued curves do not intersect in their shared domain.")


def visible_B_boundary_interval(b_boundary, supercritical):
    """Visible B-boundary: yellow junction to its exit through alpha2=max."""
    fs_start = first_curve_intersection(b_boundary, supercritical)
    fs_upper = min(float(b_boundary["fs"][-1]), base.helper.smooth.FS_MAX)
    alpha_max = base.helper.smooth.ALPHA_MAX
    if float(b_boundary["interpolator"](fs_upper)) <= alpha_max:
        fs_end = fs_upper
    else:
        fs_end = float(
            brentq(
                lambda value: float(b_boundary["interpolator"](value)) - alpha_max,
                fs_start,
                fs_upper,
            )
        )
    return fs_start, fs_end


def draw_outlined(ax, curve, color, dashes, linewidth=3.0, zorder=8):
    fs_dense = np.linspace(max(FS_TP, curve["fs"][0]), min(0.9, curve["fs"][-1]), 700)
    alpha_dense = curve["interpolator"](fs_dense)
    line, = ax.plot(
        alpha_dense,
        fs_dense,
        color=color,
        lw=linewidth,
        ls="--",
        zorder=zorder,
    )
    line.set_dashes(dashes)
    line.set_dash_capstyle("round")
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=linewidth + 1.8, foreground="0.2", alpha=0.42),
            path_effects.Normal(),
        ]
    )


def override_continuous_regions(
    alpha_dense,
    fs_dense,
    states,
    standard,
    bu_fold,
    ab_fold,
    b_boundary,
):
    """Replace grid-derived major boundaries by continuation curves.

    Below the TP, the yellow biased-unimodal region is reconstructed directly
    between its left fold boundary (symmetric saddle-node or subcritical
    pitchfork) and its right supercritical-pitchfork boundary.  Above the TP,
    the three upper transitions use the two continued folds and the solved
    bimodality-threshold classification curve.
    """
    code = {label: index for index, label in enumerate(base.PLOT_CLASSES)}
    for row, fs_value in enumerate(fs_dense):
        if fs_value < FS_TP:
            super_curve = standard["supercritical"]
            if not (super_curve["fs"][0] <= fs_value <= super_curve["fs"][-1]):
                continue
            right_alpha = float(super_curve["interpolator"](fs_value))

            if fs_value <= base.helper.smooth.LOWER_JUNCTION_FS:
                left_curve = standard["lower_saddle"]
            else:
                left_curve = standard["subcritical"]

            # Before the coexistence fold enters through alpha2=0, the yellow
            # region begins at the left plotting boundary.
            if left_curve["fs"][0] <= fs_value <= left_curve["fs"][-1]:
                left_alpha = float(left_curve["interpolator"](fs_value))
                states[row, alpha_dense < left_alpha] = code[base.TRIPLE]
            else:
                left_alpha = base.helper.smooth.ALPHA_MIN

            states[
                row,
                (alpha_dense >= left_alpha) & (alpha_dense < right_alpha),
            ] = code[base.BU]

            # Outside the yellow region, use the continuously solved B
            # threshold instead of retaining the jagged coarse-grid classes.
            # The threshold becomes an exposed state boundary only after it
            # has crossed the supercritical-pitchfork curve.
            outside_yellow = alpha_dense >= right_alpha
            if b_boundary["fs"][0] <= fs_value <= b_boundary["fs"][-1]:
                b_alpha = float(b_boundary["interpolator"](fs_value))
                if b_alpha > right_alpha:
                    states[
                        row,
                        outside_yellow & (alpha_dense < b_alpha),
                    ] = code[base.SB]
                    states[row, alpha_dense >= b_alpha] = code[base.NU]
                else:
                    states[row, outside_yellow] = code[base.NU]
            else:
                states[row, outside_yellow] = code[base.NU]
            continue

        bu_alpha = float(bu_fold["interpolator"](fs_value))
        ab_alpha = float(ab_fold["interpolator"](fs_value))
        b_alpha = float(b_boundary["interpolator"](fs_value))
        states[row, alpha_dense < bu_alpha] = code[base.TRIPLE]
        states[row, (alpha_dense >= bu_alpha) & (alpha_dense < ab_alpha)] = code[base.SB_AB]
        states[row, (alpha_dense >= ab_alpha) & (alpha_dense < b_alpha)] = code[base.SB]
        states[row, alpha_dense >= b_alpha] = code[base.NU]
    return states


def grid_transition_rmse(alpha, fs, compositions, bu_fold):
    estimates = []
    errors = []
    for i, fs_value in enumerate(fs):
        if fs_value < 0.8 - 1e-12:
            continue
        present_bu = []
        for j, composition in enumerate(compositions[i]):
            if base.BU in composition:
                present_bu.append(j)
        if not present_bu or max(present_bu) >= len(alpha) - 1:
            continue
        j = max(present_bu)
        grid_alpha = 0.5 * (alpha[j] + alpha[j + 1])
        continued_alpha = float(bu_fold["interpolator"](fs_value))
        estimates.append((float(fs_value), float(grid_alpha), continued_alpha))
        errors.append(grid_alpha - continued_alpha)
    return estimates, float(np.sqrt(np.mean(np.asarray(errors) ** 2)))


def plot_map(alpha, fs, compositions, standard, bu_fold, ab_fold, b_boundary):
    codes = base.major_composition_codes(compositions)
    alpha_dense, fs_dense, dense = base.smooth_codes(alpha, fs, codes)
    dense = override_continuous_regions(
        alpha_dense,
        fs_dense,
        dense,
        standard,
        bu_fold,
        ab_fold,
        b_boundary,
    )

    if MERGE_BIMODAL:
        old_code = {
            label: index
            for index, label in enumerate(base.PLOT_CLASSES)
        }
        new_code = {
            label: index
            for index, label in enumerate(MERGED_PLOT_CLASSES)
        }
        merged_dense = np.full_like(dense, -1)
        merged_dense[dense == old_code[base.TRIPLE]] = new_code[MERGED_COEXISTENCE]
        merged_dense[dense == old_code[base.BU]] = new_code[base.BU]
        for label in (base.SB_AB, base.SB, base.AB):
            merged_dense[dense == old_code[label]] = new_code[MERGED_BIMODAL]
        merged_dense[dense == old_code[base.NU]] = new_code[base.NU]
        if np.any(merged_dense < 0):
            raise RuntimeError("Unmapped state code in merged-bimodal plot.")
        dense = merged_dense
        plot_classes = MERGED_PLOT_CLASSES
        plot_colors = MERGED_COLORS
        plot_labels = MERGED_LABELS
    else:
        plot_classes = base.PLOT_CLASSES
        plot_colors = base.COLORS
        plot_labels = base.LABELS

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=220)
    cmap = ListedColormap([plot_colors[label] for label in plot_classes])
    norm = BoundaryNorm(np.arange(len(plot_classes) + 1) - 0.5, cmap.N)
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

    # Existing lower continuous bifurcation curves.
    for key, label in (
        ("lower_saddle", "Saddle-node"),
        ("supercritical", "Supercritical pitchfork"),
        ("subcritical", "Subcritical pitchfork"),
    ):
        base.helper.smooth.draw_curve(ax, standard[key], label)

    # Two different stable-pair saddle-nodes meet the TP from above.
    draw_outlined(ax, bu_fold, SADDLE_COLOR, (8, 5), linewidth=3.2, zorder=9)
    draw_outlined(ax, ab_fold, SADDLE_COLOR, (8, 5), linewidth=3.2, zorder=9)

    # The B threshold changes only the label of the surviving symmetric branch.
    # Draw its complete *exposed* boundary: from the point where it leaves the
    # yellow region to the point where it exits the plotting window.
    fs_class_intersection, fs_class_end = visible_B_boundary_interval(
        b_boundary, standard["supercritical"]
    )
    fs_class_start = min(
        fs_class_intersection + CLASSIFICATION_VISUAL_GAP_FS,
        fs_class_end,
    )
    fs_class = np.linspace(fs_class_start, fs_class_end, 700)
    ax.plot(
        b_boundary["interpolator"](fs_class),
        fs_class,
        color=CLASSIFICATION_COLOR,
        lw=2.1,
        ls=(0, (2.2, 2.2)),
        zorder=10,
    )

    ax.plot(
        ALPHA_TP,
        FS_TP,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=1.5,
        zorder=12,
    )
    ax.text(ALPHA_TP + 0.008, FS_TP + 0.014, "TP", fontsize=18, fontweight="bold", zorder=12)

    boundary_handles = [
        Line2D([0], [0], color=SADDLE_COLOR, lw=3, ls="--", dashes=(8, 5), label="Saddle-node"),
        Line2D([0], [0], color=SUPER_COLOR, lw=3, ls="--", dashes=(8, 5), label="Supercritical pitchfork"),
        Line2D([0], [0], color=SUB_COLOR, lw=3, ls="--", dashes=(8, 5), label="Subcritical pitchfork"),
        Line2D([0], [0], color=CLASSIFICATION_COLOR, lw=2.1, ls=(0, (2.2, 2.2)), label=rf"$B={B_CUTOFF:.2f}$ (classification only)"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="white", markeredgecolor="0.15", lw=0, label="Tricritical point"),
    ]
    fig.legend(
        handles=boundary_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        ncol=3,
        fontsize=9.4,
        framealpha=0.97,
        columnspacing=0.85,
        handlelength=2.15,
        borderpad=0.42,
    )

    state_handles = [
        Patch(facecolor=plot_colors[label], edgecolor="black", linewidth=1, label=plot_labels[label])
        for label in plot_classes
    ]
    ax.legend(
        handles=state_handles,
        loc="lower right",
        fontsize=9.2,
        framealpha=0.96,
        borderpad=0.45,
        labelspacing=0.28,
    )

    ax.set_xlim(base.helper.smooth.ALPHA_MIN, 0.30)
    ax.set_ylim(base.helper.smooth.FS_MIN, base.helper.smooth.FS_MAX)
    ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    ax.set_yticks(
        [0.2, 0.4, 0.6, 0.9],
        ["0.2", "0.4", "0.6", "0.9"],
    )
    ax.set_xlabel(r"$\alpha_2$", fontsize=37)
    ax.set_ylabel(r"$f_s$", fontsize=36)
    ax.tick_params(axis="both", labelsize=28, width=1.4, length=7)
    ax.set_box_aspect(1)
    ax.grid(ls=":", lw=0.8, alpha=0.20, color="black", zorder=1)
    fig.text(0.012, 0.958, "(b)", fontsize=21, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.84)
    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    scan = load_cache()
    # Keep the coarse-grid root classification consistent with the continuous
    # threshold solve without overwriting files produced for another cutoff.
    base.B_CUTOFF = B_CUTOFF
    base.GRID_CSV = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_composition_grid.csv"
    base.SUMMARY_JSON = HERE / f"mf_42bin_four_state_B{B_TAG}{MODE_SUFFIX}_summary.json"
    alpha, fs, compositions, _ = base.compute_grid()

    print("Loading the freshly rerun pseudo-arclength curves for the yellow region...", flush=True)
    yellow_curves, yellow_rows = recomputed_yellow_curves()
    print("Recomputing the upper biased-unimodal saddle-node...", flush=True)
    bu_fold, bu_rows = recompute_upper_bu_fold(scan)
    print("Loading the continuously traced last asymmetric-bimodal saddle-node...", flush=True)
    ab_fold, ab_rows = continued_last_asymmetric_fold()
    print(
        f"Solving the continuous symmetric B={B_CUTOFF:.2f} classification boundary...",
        flush=True,
    )
    b_boundary, b_rows = recompute_symmetric_B_boundary(scan)

    all_rows = yellow_rows + bu_rows + ab_rows + b_rows
    with OUTPUT_BOUNDARIES.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    estimates, rmse = grid_transition_rmse(alpha, fs, compositions, bu_fold)
    alpha_bu_085 = float(bu_fold["interpolator"](0.85))
    alpha_ab_085 = float(ab_fold["interpolator"](0.85))
    alpha_b_085 = float(b_boundary["interpolator"](0.85))
    summary = {
        "state_grouping": (
            "symmetric and asymmetric bimodal merged as Bimodal"
            if MERGE_BIMODAL
            else "symmetric and asymmetric bimodal kept separate"
        ),
        "bimodality_mode": BIMODALITY_MODE,
        "pink_to_dark_purple_is_bifurcation": True,
        "pink_to_dark_purple_event": "Saddle-node of the biased-unimodal stable pair",
        "yellow_region_boundaries": {
            "left_lower": "fresh pseudo-arclength saddle-node continuation",
            "left_upper": "fresh pseudo-arclength subcritical-pitchfork continuation",
            "right": "fresh pseudo-arclength supercritical-pitchfork continuation",
        },
        "fs_0.85": {
            "biased_unimodal_saddle_node_alpha2": alpha_bu_085,
            "last_asymmetric_bimodal_saddle_node_alpha2": alpha_ab_085,
            "symmetric_B_classification_alpha2": alpha_b_085,
            "B_cutoff": B_CUTOFF,
        },
        "coarse_grid_vs_continuation_BU_boundary_RMSE": rmse,
        "coarse_grid_transition_comparisons": [
            {"fs": f, "grid_midpoint": a_grid, "continued_fold": a_cont}
            for f, a_grid, a_cont in estimates
        ],
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    data_only = os.environ.get("MF_DATA_ONLY", "0") == "1"
    if not data_only:
        plot_map(alpha, fs, compositions, yellow_curves, bu_fold, ab_fold, b_boundary)
    print(json.dumps(summary, indent=2), flush=True)
    paths = [OUTPUT_BOUNDARIES, OUTPUT_SUMMARY]
    if not data_only:
        paths.extend([OUTPUT_PNG, OUTPUT_PDF])
    for path in paths:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
