"""Scatter map of all stable distributional-MF observable classes.

The population has two stubborn types and 20 intrinsic-bias bins on each
non-stubborn side, hence 42 dynamical variables in total.  At every
(alpha2, fs) point, many initial conditions are evolved, duplicate fixed
points are removed, and stability is checked in the full 42-dimensional map.

Each parameter-grid cell is filled with a small regular lattice of dots.  If
several distinct stable (B, |S|) classes coexist, their bivariate colors are
interleaved with equal dot counts.  Dot frequency therefore records existence,
not an estimated basin probability.
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import distributional_mf_alpha2_fs_background as model


# ---------------------------------------------------------------------------
# Parameter grid and 42-dimensional distributional MF
# ---------------------------------------------------------------------------
ALPHA2_VALUES = np.round(np.arange(0.0, 0.30 + 1e-12, 0.01), 12)
FS_VALUES = np.round(np.arange(0.15, 0.55 + 1e-12, 0.01), 12)
N_ETA_BINS_PER_SIDE = 20
N_STATE_VARIABLES = 2 + 2 * N_ETA_BINS_PER_SIDE

SOLUTION_TOL = 1e-4
OBSERVABLE_TOL = 2e-3
STABILITY_EIGEN_TOL = 1e-6
JACOBIAN_STEP = 1e-6

# A 6 x 6 lattice is divisible by both two and three, so the common two- and
# three-class coexistence cases receive exactly equal dot counts.
DOT_GRID_SIDE = 6
DOT_SIZE = 11

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "distributional_mf_multistability_scatter_42d.csv"
OUTPUT_PNG = SCRIPT_DIR / "distributional_mf_multistability_scatter_42d.png"
OUTPUT_PDF = SCRIPT_DIR / "distributional_mf_multistability_scatter_42d.pdf"

# Reuse the original expected-link dynamics, bivariate color scale, and
# precomputed four-group-MF boundary reader.  These functions refer to the
# module-level bin count, so set it before constructing any populations.
model.N_ETA_BINS_PER_SIDE = N_ETA_BINS_PER_SIDE


def make_seed(kind, amplitude=0.5, shift=0.0, noise=None):
    """Construct one deterministic initial condition for the 42-D map."""
    n = N_ETA_BINS_PER_SIDE
    if kind == "split":
        ns = np.concatenate(
            (
                np.full(n, amplitude + shift),
                np.full(n, -amplitude + shift),
            )
        )
    elif kind == "consensus":
        ns = np.full(2 * n, amplitude)
    elif kind == "eta":
        eta_positive = (np.arange(n) + 0.5) / n
        eta_negative = -1.0 + (np.arange(n) + 0.5) / n
        ns = amplitude * np.concatenate((eta_positive, eta_negative)) + shift
    else:
        raise ValueError(f"Unknown seed kind: {kind}")

    if noise is not None:
        ns = ns + noise
    return np.concatenate(([1.0, -1.0], np.clip(ns, -1.0, 1.0)))


def base_initial_conditions():
    """Initial-condition bank used independently at every parameter point."""
    seeds = []

    # Exact symmetric states locate symmetric bimodal/neutral fixed points.
    for amplitude in (0.02, 0.20, 0.50, 0.70, 0.85, 0.95):
        seeds.append(make_seed("split", amplitude=amplitude))

    # Small shifts in both directions locate symmetry-broken biased branches.
    for amplitude in (0.02, 0.20, 0.50, 0.70, 0.85, 0.95):
        seeds.append(make_seed("split", amplitude=amplitude, shift=0.01))
        seeds.append(make_seed("split", amplitude=amplitude, shift=-0.01))

    # Consensus-like and eta-resolved distributions enlarge basin coverage.
    for opinion in (-0.8, -0.5, 0.0, 0.5, 0.8):
        seeds.append(make_seed("consensus", amplitude=opinion))
    for scale in (0.5, 1.0):
        seeds.append(make_seed("eta", amplitude=scale))

    # Reproducible nonsymmetric seeds prevent the search from being restricted
    # to the exact symmetric invariant subspace.
    rng = np.random.default_rng(20260907)
    for amplitude in (0.20, 0.50, 0.80, 0.95):
        noise = 0.08 * rng.normal(size=2 * N_ETA_BINS_PER_SIDE)
        seeds.append(make_seed("split", amplitude=amplitude, noise=noise))

    return seeds


BASE_SEEDS = base_initial_conditions()


def append_distinct(vectors, candidate, tolerance=SOLUTION_TOL):
    """Append a vector only if it is not already represented in the list."""
    if not any(np.max(np.abs(candidate - old)) <= tolerance for old in vectors):
        vectors.append(candidate)


def solve(seed, alpha2, eta, sigma, weights):
    """Iterate a seed to a fixed point; return None on nonconvergence."""
    try:
        solution, iterations, residual = model.iterate_to_stationarity(
            seed, float(alpha2), eta, sigma, weights
        )
    except RuntimeError:
        return None
    return solution, iterations, residual


def map_jacobian(solution, alpha2, eta, sigma, weights):
    """Central-difference Jacobian of the full expected-update map."""
    dimension = solution.size
    jacobian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        step = np.zeros(dimension)
        step[column] = JACOBIAN_STEP
        plus = model.expected_update(
            solution + step, alpha2, eta, sigma, weights
        )
        minus = model.expected_update(
            solution - step, alpha2, eta, sigma, weights
        )
        jacobian[:, column] = (plus - minus) / (2.0 * JACOBIAN_STEP)
    return jacobian


def spectral_radius(solution, alpha2, eta, sigma, weights):
    eigenvalues = np.linalg.eigvals(
        map_jacobian(solution, alpha2, eta, sigma, weights)
    )
    return float(np.max(np.abs(eigenvalues)))


def stable_attractors(alpha2, fs, continuation_seeds=()):
    """Find all stable fixed points reached from the multi-start seed bank."""
    eta, sigma, weights = model.make_population(float(fs))
    seeds = list(continuation_seeds) + BASE_SEEDS

    # Avoid resolving identical continuation/base seeds.
    distinct_seeds = []
    for seed in seeds:
        append_distinct(distinct_seeds, np.asarray(seed, dtype=float), 1e-10)

    candidates = []
    metadata = {}
    for seed in distinct_seeds:
        result = solve(seed, alpha2, eta, sigma, weights)
        if result is None:
            continue
        solution, iterations, residual = result
        before = len(candidates)
        append_distinct(candidates, solution)
        if len(candidates) > before:
            metadata[id(candidates[-1])] = (iterations, residual)

    stable = []
    for candidate in candidates:
        radius = spectral_radius(candidate, alpha2, eta, sigma, weights)
        if radius < 1.0 - STABILITY_EIGEN_TOL:
            b_value, s_value = model.distribution_observables(candidate, weights)
            iterations, residual = metadata[id(candidate)]
            stable.append(
                {
                    "solution": candidate,
                    "B": b_value,
                    "S": s_value,
                    "abs_S": abs(s_value),
                    "spectral_radius": radius,
                    "iterations": iterations,
                    "residual": residual,
                }
            )

    stable.sort(key=lambda item: (item["abs_S"], item["B"], item["S"]))
    return stable


def observable_classes(attractors):
    """Collapse +/- mirror fixed points that have the same (B, |S|) color."""
    classes = []
    for attractor in attractors:
        match = None
        for item in classes:
            if (
                abs(attractor["B"] - item["B"]) <= OBSERVABLE_TOL
                and abs(attractor["abs_S"] - item["abs_S"]) <= OBSERVABLE_TOL
            ):
                match = item
                break
        if match is None:
            classes.append(
                {
                    "B": attractor["B"],
                    "abs_S": attractor["abs_S"],
                    "multiplicity": 1,
                }
            )
        else:
            match["multiplicity"] += 1
    return classes


def scan_parameter_grid():
    """Scan the grid with multi-start searches plus alpha continuation."""
    rows = []
    classes_by_point = {}

    for fs in FS_VALUES:
        previous_alpha_solutions = []
        for alpha2 in ALPHA2_VALUES:
            attractors = stable_attractors(
                float(alpha2),
                float(fs),
                continuation_seeds=previous_alpha_solutions,
            )
            previous_alpha_solutions = [item["solution"] for item in attractors]
            classes = observable_classes(attractors)
            classes_by_point[(float(alpha2), float(fs))] = classes

            class_indices = []
            for attractor in attractors:
                class_index = next(
                    index
                    for index, item in enumerate(classes, start=1)
                    if abs(attractor["B"] - item["B"]) <= OBSERVABLE_TOL
                    and abs(attractor["abs_S"] - item["abs_S"]) <= OBSERVABLE_TOL
                )
                class_indices.append(class_index)

            for attractor_index, (attractor, class_index) in enumerate(
                zip(attractors, class_indices), start=1
            ):
                rows.append(
                    {
                        "alpha2": float(alpha2),
                        "fs": float(fs),
                        "attractor": attractor_index,
                        "observable_class": class_index,
                        "B": attractor["B"],
                        "S": attractor["S"],
                        "abs_S": attractor["abs_S"],
                        "spectral_radius": attractor["spectral_radius"],
                        "iterations": attractor["iterations"],
                        "residual": attractor["residual"],
                    }
                )

        coexistence_points = sum(
            len(classes_by_point[(float(alpha2), float(fs))]) > 1
            for alpha2 in ALPHA2_VALUES
        )
        print(
            f"Completed fs={fs:.2f}; "
            f"coexistence at {coexistence_points}/{ALPHA2_VALUES.size} alpha points"
        )

    return classes_by_point, rows


def save_rows(rows):
    fields = [
        "alpha2",
        "fs",
        "attractor",
        "observable_class",
        "B",
        "S",
        "abs_S",
        "spectral_radius",
        "iterations",
        "residual",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dot_cloud(classes_by_point):
    """Create stratified-random, equally represented colored dots per cell."""
    alpha_step = float(ALPHA2_VALUES[1] - ALPHA2_VALUES[0])
    fs_step = float(FS_VALUES[1] - FS_VALUES[0])
    local_column, local_row = np.meshgrid(
        np.arange(DOT_GRID_SIDE), np.arange(DOT_GRID_SIDE)
    )
    local_column = local_column.ravel()
    local_row = local_row.ravel()
    dots_per_cell = local_column.size
    rng = np.random.default_rng(20260907)

    x_all = []
    y_all = []
    colors_all = []
    for fs_index, fs in enumerate(FS_VALUES):
        for alpha_index, alpha2 in enumerate(ALPHA2_VALUES):
            classes = classes_by_point[(float(alpha2), float(fs))]
            if not classes:
                continue
            class_colors = model.bivariate_rgba(
                np.asarray([item["B"] for item in classes]),
                np.asarray([item["abs_S"] for item in classes]),
            )

            # One randomized point per micro-cell keeps local coverage uniform
            # while removing the distracting global lattice/stripe pattern.
            offset_x = (
                (local_column + rng.random(dots_per_cell)) / DOT_GRID_SIDE - 0.5
            ) * 0.90 * alpha_step
            offset_y = (
                (local_row + rng.random(dots_per_cell)) / DOT_GRID_SIDE - 0.5
            ) * 0.90 * fs_step

            # Repeat every class as evenly as possible, then randomize its dot
            # locations.  Counts show existence only, never basin size.
            assignment = np.resize(np.arange(len(classes)), dots_per_cell)
            rng.shuffle(assignment)

            x_all.extend(float(alpha2) + offset_x)
            y_all.extend(float(fs) + offset_y)
            colors_all.extend(class_colors[assignment])

    return np.asarray(x_all), np.asarray(y_all), np.asarray(colors_all)


def add_symbol_only_color_key(ax):
    """Add the B/|S| color key using symbol-only, presentation-size labels."""
    key_ax = ax.inset_axes([0.68, 0.105, 0.285, 0.27])
    b_axis = np.linspace(model.B_COLOR_MIN, model.B_COLOR_MAX, 121)
    s_axis = np.linspace(model.ABS_S_COLOR_MIN, model.ABS_S_COLOR_MAX, 121)
    b_grid, s_grid = np.meshgrid(b_axis, s_axis)
    key_ax.imshow(
        model.bivariate_rgba(b_grid, s_grid),
        origin="lower",
        aspect="auto",
        extent=(
            model.B_COLOR_MIN,
            model.B_COLOR_MAX,
            model.ABS_S_COLOR_MIN,
            model.ABS_S_COLOR_MAX,
        ),
        interpolation="nearest",
    )
    key_ax.set_xlabel(r"$B$", fontsize=22, labelpad=1)
    key_ax.set_ylabel(r"$|S|$", fontsize=22, labelpad=1)
    key_ax.tick_params(axis="both", labelsize=17)


def plot_result(classes_by_point):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=180)
    x, y, colors = dot_cloud(classes_by_point)
    ax.scatter(
        x,
        y,
        c=colors,
        s=DOT_SIZE,
        marker="o",
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    events = model.load_bifurcation_events()
    present_types = model.plot_boundaries(ax, events)
    boundary_handles = [
        Line2D(
            [0],
            [0],
            color=model.BOUNDARY_COLORS[event_type],
            linewidth=model.BOUNDARY_LINE_WIDTH,
            linestyle="--",
            dashes=model.BOUNDARY_DASHES,
            label=event_type,
        )
        for event_type in present_types
    ]
    coexistence_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#7651a4",
        markeredgecolor="#e88937",
        markersize=8,
        label="Coexisting stable states",
    )
    ax.legend(
        handles=boundary_handles + [coexistence_handle],
        loc="upper right",
        ncol=1,
        fontsize=14,
        framealpha=0.96,
    )
    add_symbol_only_color_key(ax)

    ax.set_xlim(ALPHA2_VALUES[0], ALPHA2_VALUES[-1])
    ax.set_ylim(FS_VALUES[0], FS_VALUES[-1])
    ax.set_xticks(np.arange(0.0, 0.30 + 1e-12, 0.05))
    ax.set_yticks(np.arange(0.15, 0.55 + 1e-12, 0.05))
    ax.set_xlabel(r"$\alpha_2$", fontsize=28)
    ax.set_ylabel(r"$f_s$", fontsize=28)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(ls=":", color="black", alpha=0.16, linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    if N_STATE_VARIABLES != 42:
        raise RuntimeError("This script must use exactly 42 state variables.")
    if not model.BIFURCATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing required bifurcation CSV: {model.BIFURCATION_CSV}"
        )

    classes_by_point, rows = scan_parameter_grid()
    save_rows(rows)
    plot_result(classes_by_point)

    no_solution = sum(not classes for classes in classes_by_point.values())
    coexistence = sum(len(classes) > 1 for classes in classes_by_point.values())
    print(f"Grid points with no detected stable attractor: {no_solution}")
    print(f"Grid points with multiple observable classes: {coexistence}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
