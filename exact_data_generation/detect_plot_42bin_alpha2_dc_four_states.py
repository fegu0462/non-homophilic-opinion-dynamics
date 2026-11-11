"""Recompute and plot the 42-bin (alpha2, dc) four-state diagram.

This script deliberately does not reuse the two old manuscript branches.  It
starts from the exhaustive stable-root cache, refines every nearby loss of
stability with the 43-equation fixed-point/critical-multiplier system, and
classifies every stable root as one of four macroscopic states:

    symmetric bimodal, asymmetric bimodal, biased unimodal, neutral unimodal.

The coloured background records which of those state types coexist.  Thin
green curves show every detected stable-involving bifurcation; the two bold
green/yellow curves are the boundaries that change the macroscopic state
composition.  Dotted grey curves are B=0.57 classification boundaries and
are explicitly not labelled as bifurcations.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model
import plot_42bin_B056_mirror_state_map as helper


CACHE = HERE / "alpha2_dc_42bin_full_stability_step0p01.pkl"
CRITICAL_CSV = HERE / "alpha2_dc_42bin_redetected_stable_bifurcations.csv"
CURVE_CSV = HERE / "alpha2_dc_42bin_redetected_bifurcation_curves.csv"
GRID_CSV = HERE / "alpha2_dc_42bin_four_state_compositions.csv"
SUMMARY_JSON = HERE / "alpha2_dc_42bin_four_state_summary.json"
MACRO_CSV = HERE / "alpha2_dc_42bin_macroscopic_boundaries.csv"
TRUE_CONTINUATION_CSV = HERE / "alpha2_dc_42bin_true_pseudoarclength_curves.csv"
PITCHFORK_CRITICALITY_CSV = HERE / "alpha2_dc_42bin_pitchfork_criticality_fine.csv"
TRICRITICAL_CSV = HERE / "alpha2_dc_42bin_tricritical_points.csv"
OUTPUT_PNG = HERE / "alpha2_dc_42bin_four_state_redetected.png"
OUTPUT_PDF = HERE / "alpha2_dc_42bin_four_state_redetected.pdf"

FS = 0.50
B_CUTOFF = 0.57
MIRROR_TOL = 2.0e-5
CANDIDATE_RHO = 0.93
X_SCALE = np.sqrt(model.STATE_DIMENSION)

SB = "Symmetric bimodal"
AB = "Asymmetric bimodal"
BU = "Biased unimodal"
NU = "Neutral unimodal"
ROOT_ORDER = (SB, AB, BU, NU)

COMP_TRIPLE_BU = "BU + SB + AB"
COMP_TRIPLE_NU = "NU + SB + AB"
COMP_BU = "BU only"
COMP_NU = "NU only"
COMP_MISC = "Other coexistence"
PLOT_CLASSES = (COMP_TRIPLE_BU, COMP_TRIPLE_NU, COMP_BU, COMP_NU, COMP_MISC)
COLORS = {
    COMP_TRIPLE_BU: "#ef999c",
    COMP_TRIPLE_NU: "#b7a2d8",
    COMP_BU: "#ffe08a",
    COMP_NU: "#9ec9ee",
    COMP_MISC: "#b8d8b8",
}
LABELS = {
    COMP_TRIPLE_BU: "Coexistence: biased unimodal +\n"
                    "symmetric & asymmetric bimodal",
    COMP_TRIPLE_NU: "Coexistence: neutral unimodal +\n"
                    "symmetric & asymmetric bimodal",
    COMP_BU: "Biased unimodal",
    COMP_NU: "Neutral unimodal",
    COMP_MISC: "Other coexistence (grid boundary only)",
}


def sign_reflection(x: np.ndarray) -> np.ndarray:
    return helper.sign_reflection(np.asarray(x, dtype=float))


def classify_root(root: np.ndarray, roots: list[np.ndarray]) -> str:
    b_value = helper.bimodality_coefficient(np.asarray(root)[2:])
    reflected = sign_reflection(root)
    distinct = float(np.max(np.abs(reflected - root))) > MIRROR_TOL
    has_partner = distinct and any(
        float(np.max(np.abs(reflected - candidate))) <= MIRROR_TOL
        for candidate in roots
    )
    if b_value > B_CUTOFF:
        return AB if has_partner else SB
    return BU if has_partner else NU


def composition_label(composition: tuple[str, ...]) -> str:
    present = set(composition)
    if present == {SB, AB, BU}:
        return COMP_TRIPLE_BU
    if present == {SB, AB, NU}:
        return COMP_TRIPLE_NU
    if present == {BU}:
        return COMP_BU
    if present == {NU}:
        return COMP_NU
    return COMP_MISC


def compute_composition_grid(scan):
    alpha = np.asarray(scan["alpha_values"], dtype=float)
    dc = np.asarray(scan["dc_values"], dtype=float)
    solutions = scan["solutions"]
    codes = np.zeros((dc.size, alpha.size), dtype=np.int8)
    compositions: list[list[tuple[str, ...]]] = []
    root_counts: Counter[str] = Counter()
    composition_counts: Counter[tuple[str, ...]] = Counter()
    rows = []
    for i, dc_value in enumerate(dc):
        row_compositions = []
        for j, alpha_value in enumerate(alpha):
            roots = [np.asarray(value, dtype=float) for value in solutions[i][j]]
            labels = [classify_root(value, roots) for value in roots]
            root_counts.update(labels)
            present = set(labels)
            composition = tuple(label for label in ROOT_ORDER if label in present)
            composition_counts[composition] += 1
            row_compositions.append(composition)
            plot_label = composition_label(composition)
            codes[i, j] = PLOT_CLASSES.index(plot_label)
            rows.append(
                {
                    "alpha2": float(alpha_value),
                    "dc": float(dc_value),
                    "stable_root_count": len(roots),
                    "composition": " + ".join(composition),
                    "plot_composition": plot_label,
                }
            )
        compositions.append(row_compositions)

    with GRID_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return alpha, dc, solutions, compositions, codes, root_counts, composition_counts


def smooth_codes(alpha: np.ndarray, dc: np.ndarray, codes: np.ndarray):
    alpha_dense = np.linspace(float(alpha.min()), float(alpha.max()), 1501)
    dc_dense = np.linspace(float(dc.min()), float(dc.max()), 1801)
    aa, dd = np.meshgrid(alpha_dense, dc_dense)
    query = np.column_stack((dd.ravel(), aa.ravel()))
    fields = []
    for code in range(len(PLOT_CLASSES)):
        mask = codes == code
        signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
        signed = gaussian_filter(signed, sigma=1.55, mode="nearest")
        interpolator = RegularGridInterpolator(
            (dc, alpha), signed, method="linear", bounds_error=False, fill_value=-1e6
        )
        fields.append(interpolator(query).reshape(dd.shape))
    dense = np.argmax(np.stack(fields), axis=0).astype(np.int8)
    return alpha_dense, dc_dense, dense


def fold_residual(z: np.ndarray, dc_value: float, pop: model.Population):
    model.DC = float(dc_value)
    x = X_SCALE * np.asarray(z[:-1], dtype=float)
    alpha2 = float(z[-1])
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    eig = np.linalg.eigvals(jacobian - np.eye(pop.dimension))
    critical = eig[int(np.argmin(np.abs(eig)))]
    return np.concatenate((update - x, [float(np.real(critical))]))


def critical_diagnostics(x: np.ndarray, alpha2: float, dc_value: float, pop):
    model.DC = float(dc_value)
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    multipliers, eigenvectors = np.linalg.eig(jacobian)
    index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, index)
    reflected = sign_reflection(x)
    symmetry_error = float(np.max(np.abs(reflected - x)))

    if symmetry_error <= 2e-5:
        even_matrix = pop.q_even.T @ jacobian @ pop.q_even
        odd_matrix = pop.q_odd.T @ jacobian @ pop.q_odd
        even_distance = float(np.min(np.abs(np.linalg.eigvals(even_matrix) - 1.0)))
        odd_distance = float(np.min(np.abs(np.linalg.eigvals(odd_matrix) - 1.0)))
        event = "Pitchfork" if odd_distance < even_distance else "Symmetric saddle-node"
    else:
        even_distance = np.nan
        odd_distance = np.nan
        event = "Asymmetric saddle-node"

    b_value = helper.bimodality_coefficient(x[2:])
    _, s_value = model.observables(x, pop)
    return {
        "alpha2": float(alpha2),
        "dc": float(dc_value),
        "event": event,
        "B": float(b_value),
        "S": float(s_value),
        "abs_S": abs(float(s_value)),
        "symmetry_error": symmetry_error,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_multiplier_real": float(np.real(multipliers[index])),
        "critical_multiplier_imag": float(np.imag(multipliers[index])),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
        "even_distance_to_one": even_distance,
        "odd_distance_to_one": odd_distance,
        "solution": x,
    }


def append_distinct_critical(records: list[dict], candidate: dict):
    x = candidate["solution"]
    reflected = sign_reflection(x)
    for old in records:
        if abs(candidate["alpha2"] - old["alpha2"]) > 4e-4:
            continue
        direct = float(np.sqrt(np.mean((x - old["solution"]) ** 2)))
        mirror = float(np.sqrt(np.mean((reflected - old["solution"]) ** 2)))
        if min(direct, mirror) < 8e-4:
            return
    records.append(candidate)


def refine_dc_row(index: int, dc_value: float, alpha, row_solutions):
    pop = model.make_population(FS)
    model.DC = float(dc_value)
    candidates = []
    for j, alpha2 in enumerate(alpha):
        for x in row_solutions[j]:
            _, jacobian, _ = model.update_jacobian_alpha(x, float(alpha2), pop)
            rho = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
            _, s_value = model.observables(x, pop)
            # Only one member of each exact mirror pair is required.
            if rho >= CANDIDATE_RHO and s_value >= -2e-7:
                candidates.append((rho, float(alpha2), np.asarray(x, dtype=float)))

    # Highest-rho candidates converge most reliably.  Lower-rho candidates
    # remain useful where the 0.01 discovery grid lies farther from the fold.
    candidates.sort(key=lambda item: item[0], reverse=True)
    refined: list[dict] = []
    for rho, alpha2, x in candidates:
        initial = np.concatenate((x / X_SCALE, [alpha2 + 0.0015]))
        answer = least_squares(
            lambda z: fold_residual(z, dc_value, pop),
            initial,
            xtol=2e-11,
            ftol=2e-11,
            gtol=2e-11,
            max_nfev=500,
            x_scale="jac",
        )
        error = float(np.max(np.abs(fold_residual(answer.x, dc_value, pop))))
        critical_alpha = float(answer.x[-1])
        if error > 2e-7 or critical_alpha < -0.004 or critical_alpha > 0.304:
            continue
        critical_x = X_SCALE * answer.x[:-1]
        record = critical_diagnostics(critical_x, critical_alpha, dc_value, pop)
        # A critical point involves a stable sheet only if all remaining
        # multipliers lie within the unit circle (allowing numerical slack).
        if record["largest_noncritical_multiplier_modulus"] > 1.002:
            continue
        if abs(record["critical_multiplier_real"] - 1.0) > 3e-5:
            continue
        append_distinct_critical(refined, record)

    refined.sort(key=lambda item: (item["alpha2"], item["abs_S"]))
    return index, refined


def detect_critical_points(alpha, dc, solutions, workers: int):
    rows: list[list[dict]] = [[] for _ in dc]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                refine_dc_row, index, float(dc_value), alpha, solutions[index]
            ): index
            for index, dc_value in enumerate(dc)
        }
        completed = 0
        for future in as_completed(futures):
            index, records = future.result()
            rows[index] = records
            completed += 1
            if completed % 10 == 0 or completed == len(dc):
                print(
                    f"critical refinement {completed}/{len(dc)}; "
                    f"row dc={dc[index]:.2f}, events={len(records)}",
                    flush=True,
                )
    return rows


def connect_curves(rows: list[list[dict]], dc: np.ndarray):
    """Connect adjacent independently refined sections without inventing gaps."""
    curves: list[list[dict]] = []
    active: list[int] = []
    for row_index, records in enumerate(rows):
        for record in records:
            record["_row_index"] = row_index
        unused = set(range(len(records)))
        new_active = []
        # Greedy nearest-neighbour matching is robust here because the eta-bin
        # cascade curves remain ordered and the dc spacing is only 0.01.
        for curve_index in active:
            previous = curves[curve_index][-1]
            gap = row_index - int(previous["_row_index"])
            if gap > 3:
                continue
            best = None
            best_cost = np.inf
            for candidate_index in unused:
                candidate = records[candidate_index]
                da = abs(candidate["alpha2"] - previous["alpha2"])
                db = abs(candidate["B"] - previous["B"])
                ds = abs(candidate["abs_S"] - previous["abs_S"])
                if da > 0.035 * gap or db > 0.22 or ds > 0.20:
                    continue
                event_penalty = 0.0 if candidate["event"] == previous["event"] else 0.45
                cost = da / (0.012 * gap) + db / 0.06 + ds / 0.07 + event_penalty
                if cost < best_cost:
                    best_cost = cost
                    best = candidate_index
            if best is not None and best_cost < 4.0:
                curves[curve_index].append(records[best])
                unused.remove(best)
                new_active.append(curve_index)
            else:
                new_active.append(curve_index)
        for candidate_index in sorted(unused):
            curves.append([records[candidate_index]])
            new_active.append(len(curves) - 1)
        active = new_active

    # Remove isolated solver coincidences; real curves must be present on at
    # least three adjacent dc sections.
    curves = [curve for curve in curves if len(curve) >= 3]
    curves.sort(
        key=lambda curve: (
            curve[0]["event"],
            float(np.mean([point["alpha2"] for point in curve])),
        )
    )
    return curves


def macro_transition_targets(compositions, solutions, alpha: np.ndarray):
    """Locate only composition changes accompanied by a stable-count change.

    This keeps B=0.57 relabelling out of the bifurcation data.  Two Boolean
    properties are followed independently: presence of any bimodal root, and
    presence of a biased-unimodal mirror pair.
    """
    targets = []
    for i, row in enumerate(compositions):
        row_targets = {"bimodal_loss": [], "biased_loss": []}
        for j in range(len(alpha) - 1):
            left, right = set(row[j]), set(row[j + 1])
            if len(solutions[i][j]) == len(solutions[i][j + 1]):
                continue
            middle = float(0.5 * (alpha[j] + alpha[j + 1]))
            left_bimodal = bool(left & {SB, AB})
            right_bimodal = bool(right & {SB, AB})
            left_biased = BU in left
            right_biased = BU in right
            if left_bimodal != right_bimodal:
                row_targets["bimodal_loss"].append(middle)
            if left_biased != right_biased:
                row_targets["biased_loss"].append(middle)
        targets.append(row_targets)
    return targets


def select_macro_points(rows, targets):
    selected = {"bimodal_loss": [], "biased_loss": []}
    for row, row_targets in zip(rows, targets):
        for kind, values in row_targets.items():
            if not values:
                continue
            target = values[-1]
            if kind == "biased_loss":
                preferred = [point for point in row if point["event"] == "Pitchfork"]
            else:
                preferred = [point for point in row if "saddle-node" in point["event"]]
            candidates = preferred if preferred else row
            if not candidates:
                continue
            point = min(candidates, key=lambda item: abs(item["alpha2"] - target))
            if abs(point["alpha2"] - target) <= 0.018:
                selected[kind].append(point)
                point["macroscopic_boundary"] = True
                point["macroscopic_boundary_type"] = kind

    with MACRO_CSV.open("w", newline="", encoding="utf-8") as stream:
        fields = ["boundary", "alpha2", "dc", "event", "B", "S", "abs_S"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for kind, points in selected.items():
            for point in points:
                writer.writerow({"boundary": kind, **{key: point[key] for key in fields[1:]}})
    return selected


def save_critical(rows, curves):
    curve_lookup = {}
    for curve_id, curve in enumerate(curves, start=1):
        for point in curve:
            curve_lookup[(round(point["dc"], 8), round(point["alpha2"], 8), point["event"])] = curve_id

    fields = [
        "curve_id", "alpha2", "dc", "event", "B", "S", "abs_S",
        "symmetry_error", "fixed_residual", "critical_multiplier_real",
        "critical_multiplier_imag", "largest_noncritical_multiplier_modulus",
        "even_distance_to_one", "odd_distance_to_one", "macroscopic_boundary",
    ]
    all_records = []
    for row in rows:
        for point in row:
            record = {key: point.get(key, False if key == "macroscopic_boundary" else "") for key in fields}
            record["curve_id"] = curve_lookup.get(
                (round(point["dc"], 8), round(point["alpha2"], 8), point["event"]), ""
            )
            all_records.append(record)
    with CRITICAL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    curve_rows = []
    for curve_id, curve in enumerate(curves, start=1):
        for order, point in enumerate(curve):
            row = {key: point.get(key, "") for key in fields if key != "curve_id"}
            row.update({"curve_id": curve_id, "curve_order": order})
            curve_rows.append(row)
    curve_fields = ["curve_id", "curve_order"] + [key for key in fields if key != "curve_id"]
    with CURVE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)


def draw_curve(ax, curve):
    alpha = np.asarray([point["alpha2"] for point in curve])
    dc = np.asarray([point["dc"] for point in curve])
    event = curve[0]["event"]
    order = np.argsort(dc)
    alpha = alpha[order]
    dc = dc[order]
    style = {
        "Asymmetric saddle-node": ("#07883b", 1.05),
        "Symmetric saddle-node": ("#005a27", 1.15),
        "Pitchfork": ("#d99200", 1.05),
    }
    color, linewidth = style.get(event, ("#315f45", 0.95))
    ax.plot(alpha, dc, color=color, lw=linewidth, alpha=0.86, zorder=6)


def draw_all_critical_points(ax, critical_rows):
    styles = {
        "Asymmetric saddle-node": ("#07883b", "o", 11),
        "Symmetric saddle-node": ("#005a27", "D", 13),
        "Pitchfork": ("#d99200", "s", 10),
    }
    for event, (color, marker, size) in styles.items():
        points = [point for row in critical_rows for point in row if point["event"] == event]
        if not points:
            continue
        ax.scatter(
            [point["alpha2"] for point in points],
            [point["dc"] for point in points],
            s=size,
            marker=marker,
            c=color,
            edgecolors="none",
            alpha=0.78,
            zorder=7,
        )


def load_true_continuation_curves():
    curves = {}
    if not TRUE_CONTINUATION_CSV.exists():
        return curves
    with TRUE_CONTINUATION_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            curve_id = int(row["curve_id"])
            curves.setdefault(curve_id, []).append(
                {
                    "order": int(row["curve_order"]),
                    "alpha2": float(row["alpha2"]),
                    "dc": float(row["dc"]),
                    "event": row["event"],
                    "stable_involving": row["stable_involving"].lower() == "true",
                }
            )
    for points in curves.values():
        points.sort(key=lambda point: point["order"])
    return curves


def draw_true_continuation_curves(ax, curves):
    for points in curves.values():
        # Split at domain exits or at any part that does not involve a stable
        # sheet; never connect across a rejected interval.
        segments = []
        current = []
        for point in points:
            inside = (
                point["stable_involving"]
                and 0.0 <= point["alpha2"] <= 0.30
                and 0.0 <= point["dc"] <= 1.0
            )
            if inside:
                current.append(point)
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) < 2:
                continue
            alpha = [point["alpha2"] for point in segment]
            dc = [point["dc"] for point in segment]
            event = max(
                (point["event"] for point in segment),
                key=lambda name: sum(point["event"] == name for point in segment),
            )
            if event == "Pitchfork":
                draw_pitchfork_by_criticality(ax, alpha, dc, linewidth=1.55, zorder=7)
                continue
            elif event == "Symmetric saddle-node":
                color, linewidth = "#005a27", 1.35
            else:
                color, linewidth = "#07913d", 1.15
            ax.plot(alpha, dc, color=color, lw=linewidth, alpha=0.93, zorder=7)
            ax.plot(alpha[::10], dc[::10], ls="none", marker="o", markersize=1.9,
                    color=color, alpha=0.82, zorder=7.1)


def load_pitchfork_criticality():
    with PITCHFORK_CRITICALITY_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    dc = np.asarray([float(row["dc"]) for row in rows])
    cubic = np.asarray([float(row["normal_form_cubic"]) for row in rows])
    return dc, cubic


def draw_pitchfork_by_criticality(ax, alpha, dc, *, linewidth, zorder, linestyle="-"):
    alpha = np.asarray(alpha, dtype=float)
    dc = np.asarray(dc, dtype=float)
    reference_dc, reference_cubic = load_pitchfork_criticality()
    cubic = np.interp(dc, reference_dc, reference_cubic)
    subcritical = cubic > 0.0
    start = 0
    for index in range(1, len(alpha) + 1):
        if index == len(alpha) or subcritical[index] != subcritical[start]:
            stop = min(index + 1, len(alpha))
            color = "#e67e22" if subcritical[start] else "#f2bd00"
            ax.plot(
                alpha[start:stop], dc[start:stop], color=color, lw=linewidth,
                ls=linestyle, solid_capstyle="round", zorder=zorder,
            )
            start = index


def draw_tricritical_points(ax):
    if not TRICRITICAL_CSV.exists():
        return
    with TRICRITICAL_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, start=1):
        alpha2, dc = float(row["alpha2"]), float(row["dc"])
        ax.plot(
            alpha2, dc, marker="o", markersize=6.0, markerfacecolor="white",
            markeredgecolor="0.15", markeredgewidth=1.2, zorder=12,
        )
        ax.text(
            alpha2 + 0.004, dc + (0.018 if index == 1 else -0.032),
            rf"TCP$_{{{index}}}$", fontsize=8.5, fontweight="bold", zorder=12,
        )


def smoothed_macro_boundary(points, kind):
    if len(points) < 3:
        return np.asarray([]), np.asarray([])
    points = sorted(points, key=lambda point: point["dc"])
    dc = np.asarray([point["dc"] for point in points], dtype=float)
    alpha = np.asarray([point["alpha2"] for point in points], dtype=float)
    unique_dc, unique_indices = np.unique(dc, return_index=True)
    alpha = alpha[unique_indices]
    dc = unique_dc
    # The critical points themselves are solved continuously.  A light
    # Savitzky-Golay pass only removes row-to-row branch-selection jitter; it
    # never invents a boundary where no critical solution was found.
    if kind == "bimodal_loss" and alpha.size >= 7:
        # This macroscopic edge is the envelope of several adjacent eta-bin
        # folds.  A smoothing spline represents that envelope without drawing
        # the artificial staircase created when the outermost microscopic
        # fold switches from one bin branch to the next.
        alpha = UnivariateSpline(dc, alpha, k=3, s=4.0e-3)(dc)
        alpha = np.clip(alpha, 0.0, 0.30)
    elif alpha.size >= 7:
        window = min(11, alpha.size if alpha.size % 2 else alpha.size - 1)
        alpha = savgol_filter(alpha, window_length=window, polyorder=3, mode="interp")
    color = "#13ad38" if kind == "bimodal_loss" else "#f2bd00"
    dense_dc = np.linspace(dc.min(), dc.max(), max(300, dc.size * 8))
    dense_alpha = np.interp(dense_dc, dc, alpha)
    return dense_alpha, dense_dc


def draw_macro_boundary(ax, points, kind):
    dense_alpha, dense_dc = smoothed_macro_boundary(points, kind)
    if dense_alpha.size == 0:
        return
    color = "#13ad38" if kind == "bimodal_loss" else "#f2bd00"
    if kind == "biased_loss":
        draw_pitchfork_by_criticality(
            ax, dense_alpha, dense_dc, linewidth=2.9, zorder=8.1,
            linestyle=(0, (8, 5)),
        )
        return
    ax.plot(
        dense_alpha, dense_dc, color=color, lw=2.9, ls=(0, (8, 5)),
        solid_capstyle="round", zorder=8.1,
    )


def smooth_macro_background(macro_points):
    """Construct a dense four-composition field from the smooth boundaries."""
    alpha_dense = np.linspace(0.0, 0.30, 1501)
    dc_dense = np.linspace(0.0, 1.0, 1801)
    aa, dd = np.meshgrid(alpha_dense, dc_dense)

    green_alpha, green_dc = smoothed_macro_boundary(
        macro_points["bimodal_loss"], "bimodal_loss"
    )
    yellow_alpha, yellow_dc = smoothed_macro_boundary(
        macro_points["biased_loss"], "biased_loss"
    )
    green_at_dc = np.interp(dc_dense, green_dc, green_alpha, left=green_alpha[0], right=0.0)
    yellow_at_dc = np.interp(dc_dense, yellow_dc, yellow_alpha, left=yellow_alpha[0], right=0.0)

    has_bimodal = aa < green_at_dc[:, None]
    has_biased = aa < yellow_at_dc[:, None]
    dense = np.full(aa.shape, PLOT_CLASSES.index(COMP_NU), dtype=np.int8)
    dense[has_biased & ~has_bimodal] = PLOT_CLASSES.index(COMP_BU)
    dense[has_bimodal & has_biased] = PLOT_CLASSES.index(COMP_TRIPLE_BU)
    dense[has_bimodal & ~has_biased] = PLOT_CLASSES.index(COMP_TRIPLE_NU)
    return alpha_dense, dc_dense, dense


def plot(alpha, dc, codes, curves, macro_points, critical_rows, true_curves=None):
    # The raw 0.01 classification grid is retained in the CSV, but the plotted
    # regions are reconstructed from the continuously refined macroscopic
    # boundaries.  Consequently the fills and the overlaid curves coincide.
    alpha_dense, dc_dense, dense = smooth_macro_background(macro_points)
    cmap = ListedColormap([COLORS[label] for label in PLOT_CLASSES])
    norm = BoundaryNorm(np.arange(len(PLOT_CLASSES) + 1) - 0.5, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=220, sharex=True, sharey=True)
    panel_titles = (
        "(a) Four-state compositions and macroscopic boundaries",
        "(b) All re-detected stable-involving bifurcations",
    )
    for axis_index, (axis, title, all_curves) in enumerate(
        zip(axes, panel_titles, (False, True))
    ):
        axis.pcolormesh(
            alpha_dense, dc_dense, dense, cmap=cmap, norm=norm,
            shading="nearest", rasterized=True, zorder=0,
            alpha=1.0 if axis_index == 0 else 0.43,
        )
        if all_curves:
            if true_curves:
                draw_true_continuation_curves(axis, true_curves)
            else:
                for curve in curves:
                    draw_curve(axis, curve)
                draw_all_critical_points(axis, critical_rows)
        else:
            for kind, points in macro_points.items():
                draw_macro_boundary(axis, points, kind)
        draw_tricritical_points(axis)
        axis.set_xlim(0.0, 0.30)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.0, 0.1, 0.2, 0.3])
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.set_xlabel(r"$\alpha_2$", fontsize=18)
        axis.set_title(title, fontsize=13.2, fontweight="bold", pad=10)
        axis.grid(True, ls=":", lw=0.65, alpha=0.38)
        axis.tick_params(labelsize=12, width=1.0, length=4)
        for spine in axis.spines.values():
            spine.set_linewidth(1.05)
    axes[0].set_ylabel(r"$d_c$", fontsize=18)

    boundary_handles = [
        Line2D([0], [0], color="#13ad38", lw=3, ls=(0, (8, 5)),
               label="Envelope of state-changing saddle-nodes"),
        Line2D([0], [0], color="#07883b", marker="o", markersize=3.8, lw=1.0,
               label="Continued asymmetric saddle-nodes (panel b)"),
        Line2D([0], [0], color="#005a27", marker="D", markersize=3.6, lw=1.0,
               label="Continued symmetric saddle-node (panel b)"),
        Line2D([0], [0], color="#f2bd00", lw=2.3, ls=(0, (8, 5)),
               label="Supercritical pitchfork"),
        Line2D([0], [0], color="#e67e22", lw=2.3, ls=(0, (8, 5)),
               label="Subcritical pitchfork"),
        Line2D([0], [0], marker="o", markersize=5, markerfacecolor="white",
               markeredgecolor="0.15", lw=0, label="Tricritical point"),
    ]
    fig.legend(
        handles=boundary_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.935),
        ncol=6,
        fontsize=7.8,
        framealpha=0.97,
        columnspacing=1.3,
        handlelength=3.2,
    )

    state_handles = [
        Patch(facecolor=COLORS[label], edgecolor="0.35", linewidth=0.8, label=LABELS[label])
        for label in PLOT_CLASSES[:-1]
    ]
    axes[0].legend(
        handles=state_handles,
        loc="lower right",
        fontsize=8.8,
        framealpha=0.96,
        borderpad=0.65,
        labelspacing=0.45,
    )
    axes[1].text(
        0.012, 0.035,
        r"Four root types: SB ($B>0.57$, self-mirror), AB ($B>0.57$, mirror pair),"
        "\n" r"BU ($B\leq0.57$, mirror pair), NU ($B\leq0.57$, self-mirror)",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.72", alpha=0.92),
        zorder=20,
    )
    fig.suptitle(
        r"42-bin mean-field map at $f_s=0.50$: full stable-root rescan",
        fontsize=16.5,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.995, 0.88), w_pad=2.2)
    fig.savefig(OUTPUT_PNG, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-critical", action="store_true")
    args = parser.parse_args()

    with CACHE.open("rb") as stream:
        scan = pickle.load(stream)
    alpha, dc, solutions, compositions, codes, root_counts, composition_counts = (
        compute_composition_grid(scan)
    )

    if args.reuse_critical and CRITICAL_CSV.exists():
        raise RuntimeError("Reuse mode is intentionally disabled for this from-scratch rerun.")
    rows = detect_critical_points(alpha, dc, solutions, args.workers)
    curves = connect_curves(rows, dc)
    targets = macro_transition_targets(compositions, solutions, alpha)
    macro_points = select_macro_points(rows, targets)
    save_critical(rows, curves)
    data_only = os.environ.get("MF_DATA_ONLY", "0") == "1"
    if not data_only:
        plot(
            alpha, dc, codes, curves, macro_points, rows,
            true_curves=load_true_continuation_curves(),
        )

    event_counts = Counter(point["event"] for row in rows for point in row)
    summary = {
        "fs": FS,
        "B_cutoff": B_CUTOFF,
        "mirror_tolerance": MIRROR_TOL,
        "candidate_spectral_radius_threshold": CANDIDATE_RHO,
        "root_type_counts": dict(root_counts),
        "composition_counts": {
            " + ".join(key): value for key, value in composition_counts.items()
        },
        "critical_point_counts": dict(event_counts),
        "connected_curve_count": len(curves),
        "macroscopic_boundary_point_counts": {
            kind: len(points) for kind, points in macro_points.items()
        },
        "note": (
            f"Background boundaries caused only by B={B_CUTOFF:.2f} are classification changes, "
            "not dynamical bifurcations. Thin curves are retained only when all "
            "noncritical multipliers are stable."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not data_only:
        print(f"saved {OUTPUT_PNG}", flush=True)
        print(f"saved {OUTPUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
