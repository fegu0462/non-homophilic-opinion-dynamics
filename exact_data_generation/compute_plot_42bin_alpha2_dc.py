"""Compute the 42-bin mean-field bifurcation map in (alpha2, dc).

The receiver-extremity term is disabled (alpha3=0), alpha1=1-alpha2,
and fs=0.5 is fixed.  Canonical stable fixed points are found by multistart
iteration and checked with the full 42-D analytic Jacobian.  The symmetric
saddle-node and pitchfork curves are continued independently.  Their crossing
creates two distinct bimodal coexistence regions, while a narrow additional
biased--neutral coexistence strip is detected directly from the stable states
near the high-dc change in pitchfork criticality.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import compute_42d_bifurcation_boundaries as model
import compute_plot_42bin_alpha3_fs as shared


FS = 0.50
ALPHA2_VALUES = np.round(np.arange(0.0, 0.30 + 1e-12, 0.01), 12)
DC_VALUES = np.round(np.arange(0.0, 1.00 + 1e-12, 0.01), 12)
ALPHA2_STEP = 0.01
DC_STEP = 0.01
MAX_ITERATIONS = 120_000
ITERATION_TOL = 1e-9
SOLUTION_TOL = 1e-5
OBSERVABLE_TOL = 2e-3
STABILITY_TOL = 1e-6

HERE = Path(__file__).resolve().parent
ATTRACTOR_CSV = HERE / "alpha2_dc_42bin_stable_attractors.csv"
BOUNDARY_CSV = HERE / "alpha2_dc_42bin_boundaries.csv"
BOUNDARY_LONG_CSV = HERE / "alpha2_dc_42bin_boundaries_long.csv"
COMBINED_PNG = HERE / "alpha2_dc_42bin_bifurcation_maps.png"
COMBINED_PDF = HERE / "alpha2_dc_42bin_bifurcation_maps.pdf"
OBSERVABLE_PNG = HERE / "alpha2_dc_42bin_observable_map.png"
OBSERVABLE_PDF = HERE / "alpha2_dc_42bin_observable_map.pdf"
STATE_PNG = HERE / "alpha2_dc_42bin_state_map.png"
STATE_PDF = HERE / "alpha2_dc_42bin_state_map.pdf"

BOUNDARY_COLORS = {
    "Saddle-node": "#00E000",
    "Supercritical pitchfork": "#FFD700",
    "Subcritical pitchfork": "#FF3B30",
}
STATE_LABELS = (
    "Coexistence\n(biased unimodal and bimodal)",
    "Coexistence\n(neutral unimodal and bimodal)",
    "Coexistence\n(biased and neutral unimodal)",
    "Biased unimodal",
    "Neutral unimodal",
)
STATE_COLORS = {
    STATE_LABELS[0]: "#ef9a9a",
    STATE_LABELS[1]: "#c8b6e8",
    STATE_LABELS[2]: "#b8ddb8",
    STATE_LABELS[3]: "#ffe08a",
    STATE_LABELS[4]: "#9ec9ee",
}


def append_distinct(vectors, candidate, tolerance=SOLUTION_TOL):
    if not any(np.max(np.abs(candidate - old)) <= tolerance for old in vectors):
        vectors.append(candidate)


def iterate(seed, alpha2, population):
    x = np.asarray(seed, dtype=float).copy()
    for iteration in range(1, MAX_ITERATIONS + 1):
        x_next = model.update_jacobian_alpha(x, alpha2, population)[0]
        residual = float(np.max(np.abs(x_next - x)))
        x = x_next
        if residual < ITERATION_TOL:
            return x, iteration, residual
    raise RuntimeError(
        f"No convergence at dc={model.DC:.3f}, alpha2={alpha2:.3f}."
    )


def stable_attractors(alpha2, population, continuation_seeds):
    seeds = []
    for seed in list(continuation_seeds) + list(shared.BASE_SEEDS):
        append_distinct(seeds, np.asarray(seed, dtype=float), 1e-10)

    candidates = []
    metadata = {}
    for seed in seeds:
        try:
            solution, iterations, residual = iterate(seed, alpha2, population)
        except RuntimeError:
            continue
        old_count = len(candidates)
        append_distinct(candidates, solution)
        if len(candidates) > old_count:
            metadata[id(candidates[-1])] = (iterations, residual)

    stable = []
    for candidate in candidates:
        jacobian = model.update_jacobian_alpha(
            candidate, alpha2, population
        )[1]
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
        if spectral_radius < 1.0 - STABILITY_TOL:
            b_value, s_value = model.observables(candidate, population)
            iterations, residual = metadata[id(candidate)]
            stable.append(
                {
                    "solution": candidate,
                    "B": b_value,
                    "S": s_value,
                    "abs_S": abs(s_value),
                    "spectral_radius": spectral_radius,
                    "iterations": iterations,
                    "residual": residual,
                }
            )
    stable.sort(key=lambda item: (item["abs_S"], item["B"], item["S"]))
    return stable


def observable_classes(attractors):
    classes = []
    for attractor in attractors:
        match = next(
            (
                item for item in classes
                if abs(float(attractor["B"]) - item["B"]) <= OBSERVABLE_TOL
                and abs(float(attractor["abs_S"]) - item["abs_S"])
                <= OBSERVABLE_TOL
            ),
            None,
        )
        if match is None:
            classes.append(
                {"B": float(attractor["B"]),
                 "abs_S": float(attractor["abs_S"])}
            )
    return classes


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_real(vector):
    vector = np.real_if_close(vector, tol=1000).real
    vector = vector / np.linalg.norm(vector)
    index = int(np.argmax(np.abs(vector)))
    return vector if vector[index] >= 0.0 else -vector


def pitchfork_cubic_coefficient(dc, alpha2, u, population):
    """Lyapunov--Schmidt cubic coefficient of the 42-D pitchfork.

    The residual convention is F=G-x.  Positive d is subcritical and
    negative d is supercritical, matching the manuscript's normal-form check.
    """
    model.DC = float(dc)
    x = model.full_from_symmetric(u, population)
    dimension = x.size

    def residual(state, parameter):
        return model.update_jacobian_alpha(state, parameter, population)[0] - state

    def jacobian(state, parameter):
        return (
            model.update_jacobian_alpha(state, parameter, population)[1]
            - np.eye(dimension)
        )

    j_fixed = jacobian(x, alpha2)
    left_vectors, _, right_vectors = np.linalg.svd(j_fixed)
    q = _normalize_real(right_vectors[-1])
    p = _normalize_real(left_vectors[:, -1])
    p = p / np.dot(p, q)

    def directional_b(first, direction, state_step):
        h = state_step / np.max(np.abs(direction))
        return (
            (jacobian(x + h * direction, alpha2)
             - jacobian(x - h * direction, alpha2)) @ first
        ) / (2.0 * h)

    def directional_c(state_step):
        h = state_step / np.max(np.abs(q))
        return (
            (jacobian(x + h * q, alpha2)
             - 2.0 * jacobian(x, alpha2)
             + jacobian(x - h * q, alpha2)) @ q
        ) / h**2

    def coefficient_at_steps(state_step, parameter_step):
        f_eta = (
            residual(x, alpha2 + parameter_step)
            - residual(x, alpha2 - parameter_step)
        ) / (2.0 * parameter_step)
        b_qq = directional_b(q, q, state_step)
        bordered = np.block(
            [[j_fixed, q[:, None]], [p[None, :], np.zeros((1, 1))]]
        )

        def bordered_solve(rhs):
            return np.linalg.solve(
                bordered, np.concatenate((rhs, [0.0]))
            )[:-1]

        w_20 = bordered_solve(-b_qq)
        return (
            np.dot(p, directional_c(state_step)) / 6.0
            + 0.5 * np.dot(p, directional_b(q, w_20, state_step))
        )

    coarse = coefficient_at_steps(5.0e-4, 2.0e-4)
    fine = coefficient_at_steps(2.5e-4, 1.0e-4)
    return float((4.0 * fine - coarse) / 3.0)


def scan_one_dc(dc):
    model.DC = float(dc)
    population = model.make_population(FS)
    previous_solutions = []
    rows = []
    for alpha2 in ALPHA2_VALUES:
        attractors = stable_attractors(
            float(alpha2), population, previous_solutions
        )
        previous_solutions = [np.asarray(item["solution"]) for item in attractors]
        classes = observable_classes(attractors)
        for attractor_index, attractor in enumerate(attractors, start=1):
            class_index = next(
                index for index, item in enumerate(classes, start=1)
                if abs(float(attractor["B"]) - item["B"]) <= OBSERVABLE_TOL
                and abs(float(attractor["abs_S"]) - item["abs_S"])
                <= OBSERVABLE_TOL
            )
            rows.append(
                {
                    "alpha2": float(alpha2), "dc": float(dc), "fs": FS,
                    "attractor": attractor_index,
                    "observable_class": class_index,
                    "B": float(attractor["B"]), "S": float(attractor["S"]),
                    "abs_S": float(attractor["abs_S"]),
                    "spectral_radius": float(attractor["spectral_radius"]),
                    "iterations": int(attractor["iterations"]),
                    "residual": float(attractor["residual"]),
                }
            )
    return rows


def run_scan(workers):
    rows_by_dc = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_dc, float(dc)): float(dc) for dc in DC_VALUES
        }
        for future in as_completed(futures):
            dc = futures[future]
            rows_by_dc[dc] = future.result()
            print(f"Completed stable-attractor scan at dc={dc:.2f}", flush=True)
    rows = [row for dc in sorted(rows_by_dc) for row in rows_by_dc[dc]]
    write_rows(ATTRACTOR_CSV, rows)
    print(f"Saved {ATTRACTOR_CSV}")


def seed_at_dc_half():
    model.DC = 0.5
    seeds = model.initial_critical_points()
    saddle = model.continue_boundary(
        "Saddle-node", "even", *seeds["Saddle-node"]
    )[FS]
    pitchfork = model.continue_boundary(
        "Pitchfork", "odd", *seeds["Pitchfork"]
    )[FS]
    return {"Saddle-node": saddle, "Pitchfork": pitchfork}


def continue_in_dc(event, sector, alpha0, u0):
    population = model.make_population(FS)
    results = {0.5: (float(alpha0), u0.copy())}
    for sequence in (
        DC_VALUES[DC_VALUES > 0.5], DC_VALUES[DC_VALUES < 0.5][::-1]
    ):
        alpha2, u = float(alpha0), u0.copy()
        for dc in sequence:
            model.DC = float(dc)
            try:
                alpha2, u = model.refine_critical_point(
                    u, alpha2, population, sector
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{event} continuation stopped at dc={dc:.3f}."
                ) from error
            results[float(dc)] = (float(alpha2), u.copy())
    return results


def run_boundaries():
    seeds = seed_at_dc_half()
    curves = {
        "Saddle-node": continue_in_dc(
            "Saddle-node", "even", *seeds["Saddle-node"]
        ),
        "Pitchfork": continue_in_dc(
            "Pitchfork", "odd", *seeds["Pitchfork"]
        ),
    }
    population = model.make_population(FS)
    wide_rows, long_rows = [], []
    for dc in DC_VALUES:
        dc_value = float(dc)
        saddle_alpha, saddle_u = curves["Saddle-node"][dc_value]
        pitch_alpha, pitch_u = curves["Pitchfork"][dc_value]
        pitch_cubic = pitchfork_cubic_coefficient(
            dc_value, pitch_alpha, pitch_u, population
        )
        pitch_type = (
            "Subcritical" if pitch_cubic > 0.0 else "Supercritical"
        )
        wide_rows.append(
            {"dc": dc_value, "fs": FS,
             "saddle_node_alpha2": saddle_alpha,
             "pitchfork_alpha2": pitch_alpha,
             "pitchfork_cubic": pitch_cubic,
             "pitchfork_type": pitch_type}
        )
        for event, sector, alpha2, u in (
            ("Saddle-node", "T-even", saddle_alpha, saddle_u),
            ("Pitchfork", "T-odd", pitch_alpha, pitch_u),
        ):
            model.DC = dc_value
            x = model.full_from_symmetric(u, population)
            residual = float(np.max(np.abs(
                model.symmetric_system(u, alpha2, population)[0]
            )))
            jacobian = model.update_jacobian_alpha(x, alpha2, population)[1]
            multipliers = np.linalg.eigvals(jacobian)
            closest = int(np.argmin(np.abs(multipliers - 1.0)))
            noncritical = np.delete(multipliers, closest)
            b_value, s_value = model.observables(x, population)
            long_rows.append(
                {
                    "dc": dc_value, "fs": FS, "event": event,
                    "alpha2": alpha2, "critical_sector": sector,
                    "fixed_point_residual": residual,
                    "critical_multiplier_real": float(np.real(multipliers[closest])),
                    "critical_multiplier_imag": float(np.imag(multipliers[closest])),
                    "largest_noncritical_modulus": float(np.max(np.abs(noncritical))),
                    "B": b_value, "S": s_value,
                    "normal_form_cubic": (
                        pitch_cubic if event == "Pitchfork" else np.nan
                    ),
                    "pitchfork_type": (
                        pitch_type if event == "Pitchfork" else ""
                    ),
                }
            )
    write_rows(BOUNDARY_CSV, wide_rows)
    write_rows(BOUNDARY_LONG_CSV, long_rows)
    print(f"Saved {BOUNDARY_CSV}")
    print(f"Saved {BOUNDARY_LONG_CSV}")


def load_classes():
    grouped = defaultdict(dict)
    with ATTRACTOR_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (float(row["alpha2"]), float(row["dc"]))
            grouped[key].setdefault(
                int(row["observable_class"]),
                {"B": float(row["B"]), "abs_S": float(row["abs_S"])},
            )
    return {
        key: sorted(values.values(), key=lambda item: (item["abs_S"], item["B"]))
        for key, values in grouped.items()
    }


def load_boundaries():
    rows = list(csv.DictReader(BOUNDARY_CSV.open("r", encoding="utf-8")))
    dc = np.asarray([float(row["dc"]) for row in rows])
    return {
        "Saddle-node": (
            np.asarray([float(row["saddle_node_alpha2"]) for row in rows]), dc
        ),
        "Pitchfork": (
            np.asarray([float(row["pitchfork_alpha2"]) for row in rows]),
            dc,
            np.asarray([float(row["pitchfork_cubic"]) for row in rows]),
        ),
    }


def split_cells(classes):
    polygons, colors = [], []
    half_a, half_d = ALPHA2_STEP / 2.0, DC_STEP / 2.0
    for dc in DC_VALUES:
        for alpha2 in ALPHA2_VALUES:
            values = classes.get((float(alpha2), float(dc)), [])
            if not values:
                continue
            left, right = alpha2 - half_a, alpha2 + half_a
            bottom, top = dc - half_d, dc + half_d
            rgba = shared.rgba_from_observables(
                np.asarray([item["B"] for item in values]),
                np.asarray([item["abs_S"] for item in values]),
            )
            if len(values) == 1:
                polygons.append([(left, bottom), (right, bottom),
                                 (right, top), (left, top)])
                colors.append(rgba[0])
            else:
                polygons.extend([
                    [(left, bottom), (right, bottom), (right, top)],
                    [(left, bottom), (right, top), (left, top)],
                ])
                colors.extend([rgba[0], rgba[-1]])
    return PolyCollection(
        polygons, facecolors=np.asarray(colors),
        edgecolors=(1.0, 1.0, 1.0, 0.55), linewidths=0.24,
        antialiased=True, rasterized=True, zorder=1,
    )


def draw_boundary(axis, curve, event):
    def plot_one(alpha_values, dc_values, color):
        axis.plot(
            alpha_values, dc_values, color=color, linewidth=3.0,
            linestyle="--", dashes=(8, 5), dash_capstyle="round",
            path_effects=[
                path_effects.Stroke(
                    linewidth=5.2, foreground="black", alpha=0.55
                ),
                path_effects.Normal(),
            ], zorder=6,
        )

    if event == "Saddle-node":
        alpha2, dc = curve
        plot_one(alpha2, dc, BOUNDARY_COLORS[event])
        return

    alpha2, dc, cubic = curve
    crossings = np.where(cubic[:-1] * cubic[1:] < 0.0)[0]
    start = 0
    prefix_alpha = []
    prefix_dc = []
    for crossing in list(crossings) + [len(dc) - 1]:
        stop = crossing
        segment_alpha = prefix_alpha + list(alpha2[start:stop + 1])
        segment_dc = prefix_dc + list(dc[start:stop + 1])
        sign_value = cubic[start]
        if crossing < len(dc) - 1:
            fraction = -cubic[crossing] / (cubic[crossing + 1] - cubic[crossing])
            cross_alpha = alpha2[crossing] + fraction * (
                alpha2[crossing + 1] - alpha2[crossing]
            )
            cross_dc = dc[crossing] + fraction * (dc[crossing + 1] - dc[crossing])
            segment_alpha.append(cross_alpha)
            segment_dc.append(cross_dc)
        color_key = (
            "Subcritical pitchfork" if sign_value > 0.0
            else "Supercritical pitchfork"
        )
        plot_one(segment_alpha, segment_dc, BOUNDARY_COLORS[color_key])
        if crossing < len(dc) - 1:
            prefix_alpha = [cross_alpha]
            prefix_dc = [cross_dc]
            start = crossing + 1


def format_axis(axis):
    axis.set_xlim(0.0, 0.30)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0.0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    axis.set_xlabel(r"$\alpha_2$", fontsize=24)
    axis.set_ylabel(r"$d_c$", fontsize=24)
    axis.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    axis.grid(linestyle=":", color="black", alpha=0.16,
              linewidth=0.8, zorder=0)


def add_color_key(axis):
    key_axis = axis.inset_axes([0.68, 0.105, 0.285, 0.25])
    b_axis = np.linspace(shared.B_MIN, shared.B_MAX, 121)
    s_axis = np.linspace(shared.S_MIN, shared.S_MAX, 121)
    b_grid, s_grid = np.meshgrid(b_axis, s_axis)
    key_axis.imshow(
        shared.rgba_from_observables(b_grid, s_grid), origin="lower",
        aspect="auto", extent=(shared.B_MIN, shared.B_MAX,
                               shared.S_MIN, shared.S_MAX),
        interpolation="nearest",
    )
    key_axis.set_xlabel(r"$B$", fontsize=16, labelpad=1)
    key_axis.set_ylabel(r"$|S|$", fontsize=16, labelpad=1)
    key_axis.set_xticks([0.3, 0.5, 1.0], ["0.3", "0.5", "1"])
    key_axis.set_yticks([0.0, 0.6], ["0", "0.6"])
    key_axis.tick_params(axis="both", labelsize=12, pad=1)


def boundary_handles():
    return [
        Line2D(
            [0], [0], color=BOUNDARY_COLORS[event], linewidth=3.0,
            linestyle="--", dashes=(8, 5),
            label=event,
        ) for event in (
            "Saddle-node", "Supercritical pitchfork", "Subcritical pitchfork"
        )
    ]


def draw_observable(axis, classes, curves):
    axis.add_collection(split_cells(classes))
    for event in ("Saddle-node", "Pitchfork"):
        draw_boundary(axis, curves[event], event)
    handles = boundary_handles()
    handles.append(Patch(facecolor="white", edgecolor="black",
                         label="Split cells: coexistence"))
    axis.legend(handles=handles, loc="lower left", fontsize=10.5,
                framealpha=0.95, borderpad=0.4, labelspacing=0.3,
                handlelength=2.5)
    add_color_key(axis)
    format_axis(axis)


def state_grid(classes, curves):
    """Smooth primary regions plus scan-resolved subcritical coexistence."""
    alpha2 = np.linspace(0.0, 0.30, 1201)
    dc = np.linspace(0.0, 1.0, 1201)
    states = np.full((dc.size, alpha2.size), 4, dtype=int)
    saddle_alpha, saddle_dc = curves["Saddle-node"]
    pitch_alpha, pitch_dc, _ = curves["Pitchfork"]
    saddle_at_dc = np.interp(dc, saddle_dc, saddle_alpha)
    pitch_at_dc = np.interp(dc, pitch_dc, pitch_alpha)
    for row in range(dc.size):
        bimodal = alpha2 < saddle_at_dc[row]
        biased = alpha2 < pitch_at_dc[row]
        states[row, biased & ~bimodal] = 3
        states[row, bimodal & biased] = 0
        states[row, bimodal & ~biased] = 1

    # A subcritical high-dc pitchfork produces a very narrow coexistence band
    # that is not bounded by the two symmetric critical curves above.  Mark
    # only cells where both stable biased and neutral solutions were found.
    for dc_value in DC_VALUES:
        for alpha_value in ALPHA2_VALUES:
            values = classes.get((float(alpha_value), float(dc_value)), [])
            bimodal = any(item["B"] > 0.80 for item in values)
            biased = any(item["abs_S"] > 1e-4 for item in values)
            neutral = any(
                item["abs_S"] <= 1e-4 and item["B"] <= 0.80
                for item in values
            )
            if biased and neutral and not bimodal:
                row_mask = np.abs(dc - dc_value) <= DC_STEP / 2.0
                column_mask = np.abs(alpha2 - alpha_value) <= ALPHA2_STEP / 2.0
                states[np.ix_(row_mask, column_mask)] = 2
    return alpha2, dc, states


def draw_state(axis, classes, curves):
    alpha2, dc, states = state_grid(classes, curves)
    cmap = ListedColormap([STATE_COLORS[label] for label in STATE_LABELS])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    axis.pcolormesh(alpha2, dc, states, cmap=cmap, norm=norm,
                    shading="nearest", rasterized=True, zorder=0)
    for event in ("Saddle-node", "Pitchfork"):
        draw_boundary(axis, curves[event], event)
    axis.legend(
        handles=[Patch(facecolor=STATE_COLORS[label], edgecolor="black",
                       linewidth=0.9, label=label) for label in STATE_LABELS],
        loc="lower right", fontsize=8.2, framealpha=0.95,
        borderpad=0.45, labelspacing=0.30,
    )
    format_axis(axis)


def plot_all():
    classes, curves = load_classes(), load_boundaries()
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=180)
    draw_observable(axes[0], classes, curves)
    draw_state(axes[1], classes, curves)
    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.15,
                           top=0.95, wspace=0.22)
    figure.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(COMBINED_PDF, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.3, 5.8), dpi=180)
    draw_observable(axis, classes, curves)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    figure.savefig(OBSERVABLE_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OBSERVABLE_PDF, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.3, 5.8), dpi=180)
    draw_state(axis, classes, curves)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    figure.savefig(STATE_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(STATE_PDF, bbox_inches="tight")
    plt.close(figure)

    for path in (COMBINED_PNG, COMBINED_PDF, OBSERVABLE_PNG,
                 OBSERVABLE_PDF, STATE_PNG, STATE_PDF):
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-boundaries", action="store_true")
    args = parser.parse_args()
    if not args.skip_scan:
        run_scan(max(1, args.workers))
    elif not ATTRACTOR_CSV.exists():
        raise FileNotFoundError(ATTRACTOR_CSV)
    if not args.skip_boundaries:
        run_boundaries()
    elif not BOUNDARY_CSV.exists():
        raise FileNotFoundError(BOUNDARY_CSV)
    plot_all()


if __name__ == "__main__":
    main()
