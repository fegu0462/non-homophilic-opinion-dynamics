"""Exhaustive-audit scan of stable fixed points in the 42-bin mean-field map.

The state consists of two stubborn groups and 20 non-stubborn intrinsic-bias
bins on each side.  The parameter rectangle is

    alpha2 in [0, 0.4],  fs in [0, 0.9].

This script is deliberately different from the earlier canonical-branch
plotter.  At every grid point it performs multistart fixed-point iteration,
horizontal continuation in both directions, exact sign-reflection completion,
and a full analytic 42 x 42 Jacobian stability test.  A second vertical
continuation pass propagates every discovered solution between neighbouring
fs rows.  The saved count is therefore a reproducible *detected stable-count*
lower bound; completeness diagnostics are written alongside it.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model


ALPHA_MIN = float(os.environ.get("MF_ALPHA_MIN", "0.0"))
ALPHA_MAX = float(os.environ.get("MF_ALPHA_MAX", "0.4"))
FS_MIN = float(os.environ.get("MF_FS_MIN", "0.0"))
FS_MAX = float(os.environ.get("MF_FS_MAX", "0.9"))

SOLUTION_TOL = 3.0e-5
RESIDUAL_TOL = 2.0e-8
STABILITY_MARGIN = 2.0e-7
MAX_ITER = 60_000
CONTINUATION_MAX_ITER = 8_000

B_CUTOFF = float(os.environ.get("MF_B_CUTOFF", "0.75"))
S_CUTOFF = float(os.environ.get("MF_S_CUTOFF", "0.002"))
SIGN_SYMMETRIC = bool(np.isclose(model.STUBBORN_POSITIVE_SHARE, 0.5))


def sign_reflection(x: np.ndarray) -> np.ndarray:
    """Return the exact sign-reflected 42-D state T x(eta)=-x(-eta)."""
    n = model.BINS_PER_SIDE
    x = np.asarray(x, dtype=float)
    reflected = np.empty_like(x)
    reflected[0] = -x[1]
    reflected[1] = -x[0]
    reflected[2 : 2 + n] = -x[2 + n :][::-1]
    reflected[2 + n :] = -x[2 : 2 + n][::-1]
    return reflected


def append_distinct(items: list[np.ndarray], candidate: np.ndarray) -> bool:
    candidate = np.asarray(candidate, dtype=float)
    if any(np.max(np.abs(candidate - old)) <= SOLUTION_TOL for old in items):
        return False
    items.append(candidate.copy())
    return True


def split_seed(amplitude: float, shift: float = 0.0) -> np.ndarray:
    n = model.BINS_PER_SIDE
    return np.concatenate(
        ([1.0, -1.0], np.full(n, amplitude + shift), np.full(n, -amplitude + shift))
    )


def eta_seed(scale: float, shift: float = 0.0) -> np.ndarray:
    n = model.BINS_PER_SIDE
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    ns = scale * np.concatenate((eta_positive, eta_negative)) + shift
    return np.concatenate(([1.0, -1.0], np.clip(ns, -1.0, 1.0)))


def consensus_seed(value: float) -> np.ndarray:
    return np.concatenate(([1.0, -1.0], np.full(2 * model.BINS_PER_SIDE, value)))


def bin_flip_seed(amplitude: float, number: int, positive_to_negative: bool) -> np.ndarray:
    """Seed branches in which low-|eta| bins have switched cluster."""
    n = model.BINS_PER_SIDE
    seed = split_seed(amplitude)
    if positive_to_negative:
        seed[2 : 2 + number] = -amplitude
    else:
        # Negative eta bins are ordered from -0.975 to -0.025, so the
        # low-|eta| bins lie at the right end.
        seed[2 + 2 * n - number : 2 + 2 * n] = amplitude
    return seed


def seed_banks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return light and comprehensive deterministic seed banks."""
    light: list[np.ndarray] = []
    for amplitude in (0.0, 0.05, 0.20, 0.50, 0.75, 0.92):
        light.append(split_seed(amplitude))
    for amplitude in (0.20, 0.50, 0.80):
        for shift in (-0.04, -0.012, 0.012, 0.04):
            light.append(split_seed(amplitude, shift))
    for value in (-0.85, -0.55, -0.25, 0.0, 0.25, 0.55, 0.85):
        light.append(consensus_seed(value))
    for scale in (0.35, 0.70, 1.0):
        light.append(eta_seed(scale))

    comprehensive = list(light)
    for amplitude in (0.35, 0.65, 0.90):
        for number in (1, 2, 3, 4, 6, 8, 10):
            comprehensive.append(bin_flip_seed(amplitude, number, True))
            comprehensive.append(bin_flip_seed(amplitude, number, False))

    rng = np.random.default_rng(20260924)
    for amplitude in (0.20, 0.50, 0.80, 0.95):
        for noise_scale in (0.035, 0.10):
            for _ in range(3):
                base = split_seed(amplitude)
                base[2:] += noise_scale * rng.normal(size=2 * model.BINS_PER_SIDE)
                comprehensive.append(np.clip(base, -1.0, 1.0))

    # Remove literal duplicates once, before every parameter scan.
    def unique(values: list[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for value in values:
            if not any(np.max(np.abs(value - old)) <= 1e-12 for old in result):
                result.append(value)
        return result

    return unique(light), unique(comprehensive)


LIGHT_SEEDS, COMPREHENSIVE_SEEDS = seed_banks()
MINIMAL_SEEDS = [
    split_seed(0.05),
    split_seed(0.50),
    split_seed(0.92),
    split_seed(0.50, 0.02),
    split_seed(0.50, -0.02),
    consensus_seed(0.0),
    consensus_seed(0.65),
    consensus_seed(-0.65),
]


def expected_update_fast(x: np.ndarray, alpha2: float, pop: model.Population) -> np.ndarray:
    """Expected map without constructing derivatives during iteration."""
    difference = x[:, None] - x[None, :]
    exponent = np.clip(model.GAMMA * (np.abs(difference) - model.DC), -100.0, 100.0)
    confidence = (1.0 + np.exp(-model.GAMMA * model.DC)) / (1.0 + np.exp(exponent))
    connection = (
        (1.0 - alpha2 - model.ALPHA3) * confidence
        + alpha2 * np.abs(x)[None, :]
        + model.ALPHA3 * np.abs(x)[:, None]
    )
    denominator = connection @ pop.weights
    if np.any(denominator <= 1e-14):
        raise RuntimeError("Expected-neighbour denominator reached zero.")
    neighbour_mean = (connection @ (pop.weights * x)) / denominator
    return pop.sigma * pop.eta + (1.0 - pop.sigma) * neighbour_mean


def expected_update_batch(
    x: np.ndarray, alpha2: float, pop: model.Population
) -> np.ndarray:
    """Vectorized expected map for a bank of initial conditions."""
    difference = x[:, :, None] - x[:, None, :]
    exponent = np.clip(model.GAMMA * (np.abs(difference) - model.DC), -100.0, 100.0)
    confidence = (1.0 + np.exp(-model.GAMMA * model.DC)) / (1.0 + np.exp(exponent))
    connection = (
        (1.0 - alpha2 - model.ALPHA3) * confidence
        + alpha2 * np.abs(x)[:, None, :]
        + model.ALPHA3 * np.abs(x)[:, :, None]
    )
    denominator = np.sum(connection * pop.weights[None, None, :], axis=2)
    numerator = np.sum(
        connection * pop.weights[None, None, :] * x[:, None, :], axis=2
    )
    neighbour_mean = numerator / denominator
    return (
        pop.sigma[None, :] * pop.eta[None, :]
        + (1.0 - pop.sigma[None, :]) * neighbour_mean
    )


def iterate(seed: np.ndarray, alpha2: float, pop: model.Population) -> np.ndarray | None:
    x = np.asarray(seed, dtype=float).copy()
    for _ in range(MAX_ITER):
        try:
            update = expected_update_fast(x, alpha2, pop)
        except RuntimeError:
            return None
        error = float(np.max(np.abs(update - x)))
        x = update
        if error <= RESIDUAL_TOL:
            return x
    return None


def verify_stable(x: np.ndarray, alpha2: float, pop: model.Population):
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    residual = float(np.max(np.abs(update - x)))
    if residual > 5.0 * RESIDUAL_TOL:
        return None
    rho = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
    if rho >= 1.0 - STABILITY_MARGIN:
        return None
    return residual, rho


def solve_seeds(
    seeds: list[np.ndarray], alpha2: float, pop: model.Population
) -> list[np.ndarray]:
    unique_seeds: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.max(np.abs(seed - old)) <= 1e-10 for old in unique_seeds):
            unique_seeds.append(np.asarray(seed, dtype=float))

    candidates: list[np.ndarray] = []
    if unique_seeds:
        active = np.stack(unique_seeds)
        for _ in range(MAX_ITER):
            update = expected_update_batch(active, alpha2, pop)
            errors = np.max(np.abs(update - active), axis=1)
            converged = errors <= RESIDUAL_TOL
            for solution in update[converged]:
                append_distinct(candidates, solution)
            active = update[~converged]
            if active.size == 0:
                break

    stable: list[np.ndarray] = []
    for candidate in candidates:
        if verify_stable(candidate, alpha2, pop) is not None:
            append_distinct(stable, candidate)

    # Exact mirror completion is valid only when the positive and negative
    # stubborn weights are equal.  An imbalanced stubborn population breaks
    # that equivariance, so no reflected root may be inserted artificially.
    if SIGN_SYMMETRIC:
        for candidate in list(stable):
            mirror = sign_reflection(candidate)
            append_distinct(stable, mirror)
    stable.sort(key=lambda x: model.observables(x, pop)[1])
    return stable


def continue_seed_bank(
    seeds: list[np.ndarray], alpha2: float, pop: model.Population
) -> list[np.ndarray]:
    """Fast local continuation; final analytic checks are performed on save."""
    unique: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.max(np.abs(seed - old)) <= 1e-10 for old in unique):
            unique.append(np.asarray(seed, dtype=float))
    if not unique:
        return []
    active = np.stack(unique)
    candidates: list[np.ndarray] = []
    previous_best = np.full(active.shape[0], np.inf)
    stagnant = np.zeros(active.shape[0], dtype=int)
    for iteration in range(CONTINUATION_MAX_ITER):
        update = expected_update_batch(active, alpha2, pop)
        errors = np.max(np.abs(update - active), axis=1)
        converged = errors <= RESIDUAL_TOL
        for solution in update[converged]:
            append_distinct(candidates, solution)

        keep = ~converged
        if not np.any(keep):
            break
        improved = errors[keep] < previous_best[keep] * (1.0 - 1e-5)
        next_stagnant = np.where(improved, 0, stagnant[keep] + 1)
        next_best = np.minimum(previous_best[keep], errors[keep])
        next_active = update[keep]

        # A branch that has crossed its fold commonly settles into a cycle or
        # a residual plateau.  Stop carrying it after a long plateau instead
        # of spending MAX_ITER at every neighbouring grid point.
        still_live = next_stagnant < 500
        active = next_active[still_live]
        previous_best = next_best[still_live]
        stagnant = next_stagnant[still_live]
        if active.size == 0:
            break

    completed: list[np.ndarray] = []
    for candidate in candidates:
        append_distinct(completed, candidate)
        if SIGN_SYMMETRIC:
            append_distinct(completed, sign_reflection(candidate))
    return completed


def union_solutions(
    existing: list[np.ndarray], additions: list[np.ndarray]
) -> list[np.ndarray]:
    result = [np.asarray(item, dtype=float).copy() for item in existing]
    for candidate in additions:
        append_distinct(result, candidate)
    return result


def scan_row(
    fs_index: int,
    fs: float,
    alpha_values: np.ndarray,
    discovery_stride: int,
):
    pop = model.make_population(float(fs))
    n_alpha = len(alpha_values)
    ascending: list[list[np.ndarray]] = [[] for _ in range(n_alpha)]
    previous: list[np.ndarray] = []
    for index, alpha2 in enumerate(alpha_values):
        if index % discovery_stride == 0:
            bank = (
                COMPREHENSIVE_SEEDS
                if fs_index % discovery_stride == 0
                else LIGHT_SEEDS
            )
        else:
            bank = MINIMAL_SEEDS
        seeds = previous + bank
        found = solve_seeds(seeds, float(alpha2), pop)
        ascending[index] = found
        previous = found

    descending: list[list[np.ndarray]] = [[] for _ in range(n_alpha)]
    previous = []
    for index in range(n_alpha - 1, -1, -1):
        alpha2 = float(alpha_values[index])
        # The ascending result supplies every already discovered branch at
        # this point; the descending pass is for backward propagation, not a
        # duplicate comprehensive multistart search.
        seeds = previous + ascending[index] + MINIMAL_SEEDS
        found = solve_seeds(seeds, alpha2, pop)
        descending[index] = found
        previous = found

    merged = [
        union_solutions(ascending[index], descending[index]) for index in range(n_alpha)
    ]
    # solve_seeds already symmetry-completes every verified branch.
    return float(fs), merged


def vertical_propagation(
    solutions: list[list[list[np.ndarray]]],
    alpha_values: np.ndarray,
    fs_values: np.ndarray,
) -> int:
    """Propagate discovered branches upward and downward in fs once."""
    additions = 0
    for alpha_index, alpha2 in enumerate(alpha_values):
        previous: list[np.ndarray] = []
        for fs_index, fs in enumerate(fs_values):
            if not previous:
                previous = solutions[fs_index][alpha_index]
                continue
            pop = model.make_population(float(fs))
            found = continue_seed_bank(previous, float(alpha2), pop)
            before = len(solutions[fs_index][alpha_index])
            solutions[fs_index][alpha_index] = union_solutions(
                solutions[fs_index][alpha_index], found
            )
            additions += len(solutions[fs_index][alpha_index]) - before
            previous = solutions[fs_index][alpha_index]

        previous = []
        for fs_index in range(len(fs_values) - 1, -1, -1):
            if not previous:
                previous = solutions[fs_index][alpha_index]
                continue
            fs = float(fs_values[fs_index])
            pop = model.make_population(fs)
            found = continue_seed_bank(previous, float(alpha2), pop)
            before = len(solutions[fs_index][alpha_index])
            solutions[fs_index][alpha_index] = union_solutions(
                solutions[fs_index][alpha_index], found
            )
            additions += len(solutions[fs_index][alpha_index]) - before
            previous = solutions[fs_index][alpha_index]
        if (alpha_index + 1) % 10 == 0 or alpha_index + 1 == len(alpha_values):
            print(
                f"vertical columns {alpha_index + 1}/{len(alpha_values)}; "
                f"additions={additions}",
                flush=True,
            )
    return additions


def horizontal_propagation(
    solutions: list[list[list[np.ndarray]]],
    alpha_values: np.ndarray,
    fs_values: np.ndarray,
) -> int:
    """Propagate every discovered branch left and right in alpha2 once."""
    additions = 0
    for fs_index, fs in enumerate(fs_values):
        pop = model.make_population(float(fs))
        previous: list[np.ndarray] = []
        for alpha_index, alpha2 in enumerate(alpha_values):
            if not previous:
                previous = solutions[fs_index][alpha_index]
                continue
            found = continue_seed_bank(previous, float(alpha2), pop)
            before = len(solutions[fs_index][alpha_index])
            solutions[fs_index][alpha_index] = union_solutions(
                solutions[fs_index][alpha_index], found
            )
            additions += len(solutions[fs_index][alpha_index]) - before
            previous = solutions[fs_index][alpha_index]

        previous = []
        for alpha_index in range(len(alpha_values) - 1, -1, -1):
            if not previous:
                previous = solutions[fs_index][alpha_index]
                continue
            alpha2 = float(alpha_values[alpha_index])
            found = continue_seed_bank(previous, alpha2, pop)
            before = len(solutions[fs_index][alpha_index])
            solutions[fs_index][alpha_index] = union_solutions(
                solutions[fs_index][alpha_index], found
            )
            additions += len(solutions[fs_index][alpha_index]) - before
            previous = solutions[fs_index][alpha_index]

        if (fs_index + 1) % 20 == 0 or fs_index + 1 == len(fs_values):
            print(
                f"horizontal propagation rows {fs_index + 1}/{len(fs_values)}; "
                f"additions={additions}",
                flush=True,
            )
    return additions


def symmetry_missing_count(values: list[np.ndarray]) -> int:
    if not SIGN_SYMMETRIC:
        return 0
    missing = 0
    for value in values:
        mirror = sign_reflection(value)
        if not any(np.max(np.abs(mirror - other)) <= SOLUTION_TOL for other in values):
            missing += 1
    return missing


def validate_all_solutions(
    solutions: list[list[list[np.ndarray]]],
    alpha_values: np.ndarray,
    fs_values: np.ndarray,
) -> int:
    """Apply the analytic 42-D residual/stability test after propagation."""
    removed = 0
    for fs_index, fs in enumerate(fs_values):
        pop = model.make_population(float(fs))
        for alpha_index, alpha2 in enumerate(alpha_values):
            valid: list[np.ndarray] = []
            for value in solutions[fs_index][alpha_index]:
                if verify_stable(value, float(alpha2), pop) is not None:
                    append_distinct(valid, value)
                else:
                    removed += 1
            # Restore exact mirrors only in the balanced, equivariant model.
            if SIGN_SYMMETRIC:
                for value in list(valid):
                    append_distinct(valid, sign_reflection(value))
            solutions[fs_index][alpha_index] = valid
    return removed


def classify(values: list[np.ndarray], pop: model.Population) -> str:
    if not values:
        return "No detected stable point"
    classes: list[tuple[float, float]] = []
    for value in values:
        b_value, s_value = model.observables(value, pop)
        pair = (float(b_value), abs(float(s_value)))
        if not any(abs(pair[0] - old[0]) < 2e-3 and abs(pair[1] - old[1]) < 2e-3 for old in classes):
            classes.append(pair)
    if len(classes) > 1:
        return "Coexisting stable states"
    b_value, abs_s = classes[0]
    if abs_s <= S_CUTOFF:
        return "Bimodal polarized" if b_value > B_CUTOFF else "Neutral unimodal"
    return "Biased bimodal" if b_value > B_CUTOFF else "Biased unimodal"


def save_results(
    solutions: list[list[list[np.ndarray]]],
    alpha_values: np.ndarray,
    fs_values: np.ndarray,
    tag: str,
) -> tuple[Path, Path]:
    summary_path = HERE / f"mf_42bin_full_stability_{tag}_summary.csv"
    attractor_path = HERE / f"mf_42bin_full_stability_{tag}_attractors.csv"
    summary_fields = [
        "alpha2", "fs", "detected_stable_count", "mirror_missing_count",
        "observable_class_count", "macrostate", "max_spectral_radius",
    ]
    attractor_fields = [
        "alpha2", "fs", "attractor", "B", "S", "abs_S",
        "spectral_radius", "residual",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file, attractor_path.open(
        "w", newline="", encoding="utf-8"
    ) as attractor_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        attractor_writer = csv.DictWriter(attractor_file, fieldnames=attractor_fields)
        summary_writer.writeheader()
        attractor_writer.writeheader()
        for fs_index, fs in enumerate(fs_values):
            pop = model.make_population(float(fs))
            for alpha_index, alpha2 in enumerate(alpha_values):
                values = solutions[fs_index][alpha_index]
                observable_classes: list[tuple[float, float]] = []
                rhos = []
                for index, value in enumerate(values, start=1):
                    residual, rho = verify_stable(value, float(alpha2), pop)
                    b_value, s_value = model.observables(value, pop)
                    pair = (float(b_value), abs(float(s_value)))
                    if not any(abs(pair[0] - old[0]) < 2e-3 and abs(pair[1] - old[1]) < 2e-3 for old in observable_classes):
                        observable_classes.append(pair)
                    rhos.append(rho)
                    attractor_writer.writerow(
                        {
                            "alpha2": float(alpha2), "fs": float(fs),
                            "attractor": index, "B": b_value, "S": s_value,
                            "abs_S": abs(s_value), "spectral_radius": rho,
                            "residual": residual,
                        }
                    )
                summary_writer.writerow(
                    {
                        "alpha2": float(alpha2), "fs": float(fs),
                        "detected_stable_count": len(values),
                        "mirror_missing_count": symmetry_missing_count(values),
                        "observable_class_count": len(observable_classes),
                        "macrostate": classify(values, pop),
                        "max_spectral_radius": max(rhos) if rhos else np.nan,
                    }
                )
    return summary_path, attractor_path


def plot_results(
    solutions: list[list[list[np.ndarray]]],
    alpha_values: np.ndarray,
    fs_values: np.ndarray,
    tag: str,
) -> tuple[Path, Path]:
    counts = np.asarray(
        [[len(solutions[i][j]) for j in range(len(alpha_values))] for i in range(len(fs_values))],
        dtype=int,
    )
    states = (
        "Coexisting stable states", "Bimodal polarized", "Biased bimodal",
        "Biased unimodal", "Neutral unimodal", "No detected stable point",
    )
    state_colors = {
        "Coexisting stable states": "#ef9a9a",
        "Bimodal polarized": "#b39ddb",
        "Biased bimodal": "#d58ab4",
        "Biased unimodal": "#ffe08a",
        "Neutral unimodal": "#9ec9ee",
        "No detected stable point": "#ffffff",
    }
    state_code = {state: index for index, state in enumerate(states)}
    state_grid = np.empty_like(counts)
    for i, fs in enumerate(fs_values):
        pop = model.make_population(float(fs))
        for j in range(len(alpha_values)):
            state_grid[i, j] = state_code[classify(solutions[i][j], pop)]

    alpha_edges = np.r_[alpha_values[0] - (alpha_values[1] - alpha_values[0]) / 2,
                        alpha_values + (alpha_values[1] - alpha_values[0]) / 2]
    fs_edges = np.r_[fs_values[0] - (fs_values[1] - fs_values[0]) / 2,
                     fs_values + (fs_values[1] - fs_values[0]) / 2]

    unique_counts = np.unique(counts)
    palette = plt.cm.turbo(np.linspace(0.05, 0.95, len(unique_counts)))
    count_cmap = ListedColormap(palette)
    count_norm = BoundaryNorm(np.arange(len(unique_counts) + 1) - 0.5, len(unique_counts))
    remapped = np.searchsorted(unique_counts, counts)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=190, sharex=True, sharey=True)
    mesh = axes[0].pcolormesh(alpha_edges, fs_edges, remapped, cmap=count_cmap, norm=count_norm, shading="flat")
    cbar = fig.colorbar(mesh, ax=axes[0], fraction=0.046, pad=0.025, ticks=np.arange(len(unique_counts)))
    cbar.ax.set_yticklabels([str(value) for value in unique_counts])
    cbar.set_label("Detected stable fixed points", fontsize=12)

    state_cmap = ListedColormap([state_colors[state] for state in states])
    state_norm = BoundaryNorm(np.arange(len(states) + 1) - 0.5, len(states))
    axes[1].pcolormesh(alpha_edges, fs_edges, state_grid, cmap=state_cmap, norm=state_norm, shading="flat")

    # Every change in the integer count is a detected stable-count boundary.
    for ax in axes:
        for i in range(counts.shape[0] - 1):
            changed = counts[i] != counts[i + 1]
            for j in np.flatnonzero(changed):
                ax.plot([alpha_edges[j], alpha_edges[j + 1]], [0.5 * (fs_values[i] + fs_values[i + 1])] * 2,
                        color="black", lw=0.28, alpha=0.72, zorder=5)
        for j in range(counts.shape[1] - 1):
            changed = counts[:, j] != counts[:, j + 1]
            for i in np.flatnonzero(changed):
                ax.plot([0.5 * (alpha_values[j] + alpha_values[j + 1])] * 2, [fs_edges[i], fs_edges[i + 1]],
                        color="black", lw=0.28, alpha=0.72, zorder=5)
        ax.set_xlim(ALPHA_MIN, ALPHA_MAX)
        ax.set_ylim(FS_MIN, FS_MAX)
        ax.set_xlabel(r"$\alpha_2$", fontsize=20)
        ax.set_xticks(np.arange(0.0, 0.401, 0.05))
        ax.set_yticks(np.arange(0.0, 0.901, 0.10))
        ax.tick_params(labelsize=13)
        ax.grid(ls=":", color="0.3", alpha=0.18, lw=0.6)
    axes[0].set_ylabel(r"$f_s$", fontsize=20)
    axes[0].set_title("(a) Detected stable-count map", fontsize=15, weight="bold")
    axes[1].set_title("(b) Macroscopic state classes", fontsize=15, weight="bold")
    handles = [Patch(facecolor=state_colors[state], edgecolor="0.3", label=state) for state in states if np.any(state_grid == state_code[state])]
    axes[1].legend(handles=handles, fontsize=9.3, loc="lower right", framealpha=0.96)
    fig.suptitle("42-bin mean-field audit: black segments mark every detected stable-count change", fontsize=15.5, weight="bold")
    fig.tight_layout()
    png = HERE / f"mf_42bin_full_stability_{tag}.png"
    pdf = HERE / f"mf_42bin_full_stability_{tag}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--discovery-stride", type=int, default=4)
    parser.add_argument("--skip-vertical", action="store_true")
    parser.add_argument("--resume-horizontal", action="store_true")
    parser.add_argument("--resume-final", action="store_true")
    parser.add_argument("--propagation-cycles", type=int, default=0)
    args = parser.parse_args()

    step = float(args.step)
    alpha_values = np.round(np.arange(ALPHA_MIN, ALPHA_MAX + 0.5 * step, step), 12)
    fs_values = np.round(np.arange(FS_MIN, FS_MAX + 0.5 * step, step), 12)
    tag = f"step{step:.3f}".replace(".", "p")
    if not SIGN_SYMMETRIC:
        share_tag = int(round(100 * model.STUBBORN_POSITIVE_SHARE))
        tag += f"_stubborn_pos{share_tag:02d}"
    checkpoint = HERE / f"mf_42bin_full_stability_{tag}.pkl"

    print(f"state dimension={model.STATE_DIMENSION}", flush=True)
    print(f"grid={len(alpha_values)} x {len(fs_values)} = {len(alpha_values)*len(fs_values)} points", flush=True)
    print(f"light seeds={len(LIGHT_SEEDS)}, comprehensive seeds={len(COMPREHENSIVE_SEEDS)}", flush=True)
    started = time.time()

    horizontal_checkpoint = HERE / f"mf_42bin_full_stability_{tag}_horizontal.pkl"
    if args.resume_final and checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if not np.array_equal(saved["alpha_values"], alpha_values) or not np.array_equal(saved["fs_values"], fs_values):
            raise RuntimeError("Final checkpoint grid does not match requested grid.")
        solutions = saved["solutions"]
        print(f"loaded final checkpoint {checkpoint}", flush=True)
    elif args.resume_horizontal and horizontal_checkpoint.exists():
        with horizontal_checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if not np.array_equal(saved["alpha_values"], alpha_values) or not np.array_equal(saved["fs_values"], fs_values):
            raise RuntimeError("Horizontal checkpoint grid does not match requested grid.")
        solutions = saved["solutions"]
        print(f"loaded horizontal checkpoint {horizontal_checkpoint}", flush=True)
    else:
        solutions_by_fs: dict[float, list[list[np.ndarray]]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    scan_row,
                    fs_index,
                    float(fs),
                    alpha_values,
                    args.discovery_stride,
                ): float(fs)
                for fs_index, fs in enumerate(fs_values)
            }
            completed = 0
            for future in as_completed(futures):
                fs, row = future.result()
                solutions_by_fs[fs] = row
                completed += 1
                if completed % 5 == 0 or completed == len(fs_values):
                    print(f"horizontal rows {completed}/{len(fs_values)}; elapsed={time.time()-started:.1f}s", flush=True)

        solutions = [solutions_by_fs[float(fs)] for fs in fs_values]
        with horizontal_checkpoint.open("wb") as stream:
            pickle.dump(
                {"alpha_values": alpha_values, "fs_values": fs_values, "solutions": solutions},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"saved horizontal checkpoint {horizontal_checkpoint}", flush=True)
    if not args.resume_final and not args.skip_vertical:
        print("starting vertical bidirectional propagation", flush=True)
        additions = vertical_propagation(solutions, alpha_values, fs_values)
        print(f"vertical propagation added {additions} stable solutions", flush=True)

    for cycle in range(args.propagation_cycles):
        print(f"starting convergence propagation cycle {cycle + 1}", flush=True)
        horizontal_additions = horizontal_propagation(
            solutions, alpha_values, fs_values
        )
        vertical_additions = vertical_propagation(
            solutions, alpha_values, fs_values
        )
        print(
            f"cycle {cycle + 1}: horizontal additions={horizontal_additions}, "
            f"vertical additions={vertical_additions}",
            flush=True,
        )

    print("validating every retained solution with the analytic 42-D Jacobian", flush=True)
    removed = validate_all_solutions(solutions, alpha_values, fs_values)
    print(f"analytic validation removed {removed} marginal/invalid candidates", flush=True)

    with checkpoint.open("wb") as stream:
        pickle.dump(
            {"alpha_values": alpha_values, "fs_values": fs_values, "solutions": solutions},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary, attractors = save_results(solutions, alpha_values, fs_values, tag)
    data_only = os.environ.get("MF_DATA_ONLY", "0") == "1"
    if not data_only:
        png, pdf = plot_results(solutions, alpha_values, fs_values, tag)
    missing = sum(symmetry_missing_count(values) for row in solutions for values in row)
    max_count = max(len(values) for row in solutions for values in row)
    print(f"mirror-missing total={missing}; maximum detected stable count={max_count}", flush=True)
    print(f"saved {checkpoint}", flush=True)
    print(f"saved {summary}", flush=True)
    print(f"saved {attractors}", flush=True)
    if not data_only:
        print(f"saved {png}", flush=True)
        print(f"saved {pdf}", flush=True)


if __name__ == "__main__":
    main()
