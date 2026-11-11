"""Compute and plot the 42-bin mean-field bifurcation map in (alpha3, fs).

The source-extremity term is disabled (alpha2=0), while alpha3 is varied and
alpha1=1-alpha3.  The population contains two stubborn bins and 20 midpoint
bins on each side of the non-stubborn intrinsic-bias distribution.

The script produces

1. a multistability-aware B/|S| map of the three canonical collective-state
   branches studied in the manuscript;
2. a categorical state map defined by the 42-dimensional saddle-node and
   supercritical-pitchfork boundaries;
3. CSV files containing the stable attractors and critical boundaries.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.legend_handler import HandlerBase
from scipy.optimize import root
from scipy.stats import kurtosis as sample_kurtosis
from scipy.stats import skew as sample_skew


# ---------------------------------------------------------------------------
# Model and parameter domain
# ---------------------------------------------------------------------------
ALPHA2 = 0.0
ALPHA3_MIN = 0.0
ALPHA3_MAX = 0.70
ALPHA3_STEP = 0.01
FS_MIN = 0.15
FS_MAX = 0.55
FS_STEP = 0.01

ALPHA3_VALUES = np.round(
    np.arange(ALPHA3_MIN, ALPHA3_MAX + 0.5 * ALPHA3_STEP, ALPHA3_STEP), 12
)
FS_VALUES = np.round(
    np.arange(FS_MIN, FS_MAX + 0.5 * FS_STEP, FS_STEP), 12
)

GAMMA = 4.0
DC = 0.5
SIGMA_S = 0.8
SIGMA_NS = 0.1
BINS_PER_SIDE = 20
STATE_DIMENSION = 2 + 2 * BINS_PER_SIDE
BIMODALITY_MODE = os.environ.get("BIMODALITY_MODE", "population").strip().lower()
if BIMODALITY_MODE not in {"population", "sample_corrected"}:
    raise ValueError(f"Unsupported BIMODALITY_MODE={BIMODALITY_MODE!r}")
B_CUTOFF = float(os.environ.get("B_CUTOFF", "0.56"))
MIRROR_TOL = 2.0e-5

ITERATION_TOL = 1e-9
MAX_ITERATIONS = 50_000
SOLUTION_TOL = 1e-5
OBSERVABLE_TOL = 2e-3
STABILITY_TOL = 1e-6
ROOT_XTOL = 2e-10
ROOT_RESIDUAL_TOL = 2e-7

HERE = Path(__file__).resolve().parent
ATTRACTOR_CSV = HERE / "alpha3_fs_42bin_stable_attractors.csv"
BOUNDARY_WIDE_CSV = HERE / "alpha3_fs_42bin_boundaries.csv"
BOUNDARY_LONG_CSV = HERE / "alpha3_fs_42bin_boundaries_long.csv"
COMBINED_PNG = HERE / "alpha3_fs_42bin_bifurcation_maps.png"
COMBINED_PDF = HERE / "alpha3_fs_42bin_bifurcation_maps.pdf"
OBSERVABLE_PNG = HERE / "alpha3_fs_42bin_observable_map.png"
OBSERVABLE_PDF = HERE / "alpha3_fs_42bin_observable_map.pdf"
STATE_PNG = HERE / "alpha3_fs_42bin_state_map.png"
STATE_PDF = HERE / "alpha3_fs_42bin_state_map.pdf"


# ---------------------------------------------------------------------------
# Plot style: identical state colors to the alpha2--fs manuscript map
# ---------------------------------------------------------------------------
B_MIN, B_MAX = 0.30, 1.00
S_MIN, S_MAX = 0.00, 0.60
COLOR_LOW_LOW = np.asarray(to_rgb("#f7f7f7"))
COLOR_HIGH_B = np.asarray(to_rgb("#3478bf"))
COLOR_HIGH_S = np.asarray(to_rgb("#e88937"))
COLOR_HIGH_BS = np.asarray(to_rgb("#7651a4"))

BOUNDARY_COLORS = {
    "Saddle-node": "#00E000",
    "Pitchfork": "#FFD700",
}
STATE_COLORS = {
    "Coexistence\n(biased unimodal and bimodal)": "#ef9a9a",
    "Biased unimodal": "#ffe08a",
    "Neutral unimodal": "#9ec9ee",
}
BOUNDARY_LINEWIDTH = 3.0
BOUNDARY_DASHES = (8, 5)
BOUNDARY_OUTLINE_WIDTH = 5.2


@dataclass
class Population:
    fs: float
    eta: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray
    select_symmetric: np.ndarray
    expand_symmetric: np.ndarray
    q_even: np.ndarray
    q_odd: np.ndarray

    @property
    def dimension(self) -> int:
        return self.eta.size

    @property
    def symmetric_dimension(self) -> int:
        return BINS_PER_SIDE + 1


def make_population(fs: float) -> Population:
    """Create the 42-dimensional eta-resolved population."""
    n = BINS_PER_SIDE
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    eta = np.concatenate(([1.0, -1.0], eta_positive, eta_negative))
    sigma = np.concatenate(([SIGMA_S, SIGMA_S], np.full(2 * n, SIGMA_NS)))
    weights = np.concatenate(
        ([fs / 2.0, fs / 2.0], np.full(2 * n, (1.0 - fs) / (2.0 * n)))
    )

    dimension = 2 + 2 * n
    symmetric_dimension = 1 + n
    expand = np.zeros((dimension, symmetric_dimension))
    q_even = np.zeros_like(expand)
    q_odd = np.zeros_like(expand)

    expand[0, 0], expand[1, 0] = 1.0, -1.0
    q_even[0, 0], q_even[1, 0] = 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
    q_odd[0, 0], q_odd[1, 0] = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)

    for k in range(n):
        positive = 2 + k
        negative = 2 + n + (n - 1 - k)
        expand[positive, 1 + k] = 1.0
        expand[negative, 1 + k] = -1.0
        q_even[positive, 1 + k] = 1.0 / np.sqrt(2.0)
        q_even[negative, 1 + k] = -1.0 / np.sqrt(2.0)
        q_odd[positive, 1 + k] = 1.0 / np.sqrt(2.0)
        q_odd[negative, 1 + k] = 1.0 / np.sqrt(2.0)

    return Population(
        fs=float(fs),
        eta=eta,
        sigma=sigma,
        weights=weights,
        select_symmetric=np.concatenate(([0], np.arange(2, 2 + n))),
        expand_symmetric=expand,
        q_even=q_even,
        q_odd=q_odd,
    )


def update_and_jacobian(
    x: np.ndarray, alpha3: float, pop: Population
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the MF update, its state Jacobian, and d(update)/dalpha3."""
    weights = pop.weights
    difference = x[:, None] - x[None, :]
    distance = np.abs(difference)
    exponent = np.clip(GAMMA * (distance - DC), -100.0, 100.0)
    logistic = 1.0 / (1.0 + np.exp(exponent))
    prefactor = 1.0 + np.exp(-GAMMA * DC)
    confidence = prefactor * logistic
    receiver_extremity = np.abs(x)[:, None]

    alpha1 = 1.0 - alpha3
    connection = alpha1 * confidence + alpha3 * receiver_extremity
    denominator = connection @ weights
    if np.any(denominator <= 1e-14):
        raise RuntimeError("Expected-neighbour denominator reached zero.")
    numerator = connection @ (weights * x)
    neighbour_mean = numerator / denominator
    update = pop.sigma * pop.eta + (1.0 - pop.sigma) * neighbour_mean

    sign_difference = np.sign(difference)
    dconfidence_dr = -GAMMA * prefactor * logistic * (1.0 - logistic)
    dconnection_receiver = (
        alpha1 * dconfidence_dr * sign_difference
        + alpha3 * np.sign(x)[:, None]
    )
    dconnection_source = -alpha1 * dconfidence_dr * sign_difference

    ddenominator = dconnection_source * weights[None, :]
    dnumerator = weights[None, :] * (
        dconnection_source * x[None, :] + connection
    )
    diagonal = np.arange(x.size)
    ddenominator[diagonal, diagonal] += dconnection_receiver @ weights
    dnumerator[diagonal, diagonal] += dconnection_receiver @ (weights * x)

    jacobian = (1.0 - pop.sigma)[:, None] * (
        dnumerator * denominator[:, None]
        - numerator[:, None] * ddenominator
    ) / denominator[:, None] ** 2

    dconnection_alpha3 = receiver_extremity - confidence
    ddenominator_alpha3 = dconnection_alpha3 @ weights
    dnumerator_alpha3 = dconnection_alpha3 @ (weights * x)
    dalpha3 = (1.0 - pop.sigma) * (
        dnumerator_alpha3 * denominator
        - numerator * ddenominator_alpha3
    ) / denominator**2
    return update, jacobian, dalpha3


def iterate_to_stationarity(
    seed: np.ndarray, alpha3: float, pop: Population
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(seed, dtype=float).copy()
    for iteration in range(1, MAX_ITERATIONS + 1):
        x_next = update_and_jacobian(x, alpha3, pop)[0]
        residual = float(np.max(np.abs(x_next - x)))
        x = x_next
        if residual < ITERATION_TOL:
            return x, iteration, residual
    raise RuntimeError(
        f"No convergence at fs={pop.fs:.3f}, alpha3={alpha3:.3f}."
    )


def make_seed(kind: str, amplitude: float = 0.5, shift: float = 0.0) -> np.ndarray:
    n = BINS_PER_SIDE
    if kind == "split":
        ns = np.concatenate(
            (np.full(n, amplitude + shift), np.full(n, -amplitude + shift))
        )
    elif kind == "consensus":
        ns = np.full(2 * n, amplitude)
    elif kind == "eta":
        eta_positive = (np.arange(n) + 0.5) / n
        eta_negative = -1.0 + (np.arange(n) + 0.5) / n
        ns = amplitude * np.concatenate((eta_positive, eta_negative)) + shift
    else:
        raise ValueError(f"Unknown seed type: {kind}")
    return np.concatenate(([1.0, -1.0], np.clip(ns, -1.0, 1.0)))


def base_seeds() -> list[np.ndarray]:
    """Smooth seeds for the canonical bimodal, biased, and neutral branches.

    Random bin-wise seeds are deliberately excluded.  With pure homophily,
    such seeds can reach combinatorial fixed points in which a few individual
    eta bins detach from an otherwise smooth branch.  Those fragmented states
    are outside the manuscript's three-state classification and were likewise
    excluded from the corresponding alpha2--fs map.
    """
    seeds = [
        make_seed("split", 0.02),
        make_seed("split", 0.50),
        make_seed("split", 0.90),
        make_seed("split", 0.50, 0.02),
        make_seed("split", 0.50, -0.02),
        make_seed("consensus", 0.60),
        make_seed("consensus", -0.60),
        make_seed("eta", 1.00),
    ]
    return seeds


BASE_SEEDS = base_seeds()


def append_distinct(
    vectors: list[np.ndarray], candidate: np.ndarray, tolerance: float = SOLUTION_TOL
) -> None:
    if not any(np.max(np.abs(candidate - old)) <= tolerance for old in vectors):
        vectors.append(candidate)


def observables(x: np.ndarray, pop: Population) -> tuple[float, float]:
    values = x[2:]
    weights = pop.weights[2:] / np.sum(pop.weights[2:])
    mean = float(np.sum(weights * values))
    if BIMODALITY_MODE == "sample_corrected":
        if values.size != 2 * BINS_PER_SIDE:
            raise ValueError(f"Expected {2 * BINS_PER_SIDE} non-stubborn bins.")
        if float(np.std(values)) <= 1.0e-12:
            return 0.0, mean
        sample_skewness = float(sample_skew(values, bias=False))
        excess_kurtosis = float(sample_kurtosis(values, fisher=True, bias=False))
        n = int(values.size)
        finite_n_term = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        denominator = excess_kurtosis + finite_n_term
        if not np.isfinite(denominator) or denominator <= 0.0:
            return np.nan, mean
        return float((sample_skewness**2 + 1.0) / denominator), mean
    centered = values - mean
    variance = float(np.sum(weights * centered**2))
    if variance <= 1e-16:
        return np.nan, mean
    skewness = float(np.sum(weights * centered**3) / variance**1.5)
    kurtosis = float(np.sum(weights * centered**4) / variance**2)
    return float((skewness**2 + 1.0) / kurtosis), mean


def sign_reflection(solution: np.ndarray) -> np.ndarray:
    """Return the exact eta-reflected, opinion-sign-reversed binned state."""
    x = np.asarray(solution, dtype=float)
    n = (x.size - 2) // 2
    reflected = np.empty_like(x)
    reflected[0] = -x[1]
    reflected[1] = -x[0]
    reflected[2 : 2 + n] = -x[2 + n : 2 + 2 * n][::-1]
    reflected[2 + n : 2 + 2 * n] = -x[2 : 2 + n][::-1]
    return reflected


def four_state_label(solution: np.ndarray, b_value: float) -> str:
    asymmetric = float(np.max(np.abs(sign_reflection(solution) - solution))) > MIRROR_TOL
    if b_value > B_CUTOFF:
        return "Asymmetric bimodal" if asymmetric else "Symmetric bimodal"
    return "Biased unimodal" if asymmetric else "Neutral unimodal"


def stable_attractors(
    alpha3: float, pop: Population, continuation_seeds: list[np.ndarray]
) -> list[dict[str, object]]:
    seeds: list[np.ndarray] = []
    for seed in continuation_seeds + BASE_SEEDS:
        append_distinct(seeds, np.asarray(seed, dtype=float), tolerance=1e-10)

    candidates: list[np.ndarray] = []
    metadata: dict[int, tuple[int, float]] = {}
    for seed in seeds:
        try:
            solution, iterations, residual = iterate_to_stationarity(
                seed, alpha3, pop
            )
        except RuntimeError:
            continue
        old_count = len(candidates)
        append_distinct(candidates, solution)
        if len(candidates) > old_count:
            metadata[id(candidates[-1])] = (iterations, residual)

    stable: list[dict[str, object]] = []
    for candidate in candidates:
        jacobian = update_and_jacobian(candidate, alpha3, pop)[1]
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
        if spectral_radius < 1.0 - STABILITY_TOL:
            b_value, s_value = observables(candidate, pop)
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


def observable_classes(attractors: list[dict[str, object]]) -> list[dict[str, float]]:
    classes: list[dict[str, float]] = []
    for attractor in attractors:
        matching = next(
            (
                item
                for item in classes
                if abs(float(attractor["B"]) - item["B"]) <= OBSERVABLE_TOL
                and abs(float(attractor["abs_S"]) - item["abs_S"])
                <= OBSERVABLE_TOL
            ),
            None,
        )
        if matching is None:
            classes.append(
                {
                    "B": float(attractor["B"]),
                    "abs_S": float(attractor["abs_S"]),
                }
            )
    return classes


def scan_one_fs(fs: float) -> list[dict[str, object]]:
    pop = make_population(float(fs))
    rows: list[dict[str, object]] = []
    previous_solutions: list[np.ndarray] = []
    for alpha3 in ALPHA3_VALUES:
        attractors = stable_attractors(float(alpha3), pop, previous_solutions)
        previous_solutions = [np.asarray(item["solution"]) for item in attractors]
        classes = observable_classes(attractors)
        for attractor_index, attractor in enumerate(attractors, start=1):
            class_index = next(
                index
                for index, item in enumerate(classes, start=1)
                if abs(float(attractor["B"]) - item["B"]) <= OBSERVABLE_TOL
                and abs(float(attractor["abs_S"]) - item["abs_S"])
                <= OBSERVABLE_TOL
            )
            rows.append(
                {
                    "alpha3": float(alpha3),
                    "fs": float(fs),
                    "attractor": attractor_index,
                    "observable_class": class_index,
                    "B": float(attractor["B"]),
                    "S": float(attractor["S"]),
                    "abs_S": float(attractor["abs_S"]),
                    "state": four_state_label(
                        np.asarray(attractor["solution"]), float(attractor["B"])
                    ),
                    "spectral_radius": float(attractor["spectral_radius"]),
                    "iterations": int(attractor["iterations"]),
                    "residual": float(attractor["residual"]),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stable_scan(workers: int) -> None:
    rows_by_fs: dict[float, list[dict[str, object]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_fs, float(fs)): float(fs) for fs in FS_VALUES
        }
        for future in as_completed(futures):
            fs = futures[future]
            rows_by_fs[fs] = future.result()
            print(f"Completed stable-attractor scan at fs={fs:.2f}", flush=True)
    rows = [row for fs in sorted(rows_by_fs) for row in rows_by_fs[fs]]
    write_rows(ATTRACTOR_CSV, rows)
    print(f"Saved {ATTRACTOR_CSV}")


# ---------------------------------------------------------------------------
# Critical-boundary continuation in the symmetry-reduced 42-D system
# ---------------------------------------------------------------------------
def full_from_symmetric(u: np.ndarray, pop: Population) -> np.ndarray:
    return pop.expand_symmetric @ np.asarray(u, dtype=float)


def symmetric_seed(pop: Population, amplitude: float) -> np.ndarray:
    x = np.concatenate(
        ([1.0, -1.0], np.full(BINS_PER_SIDE, amplitude),
         np.full(BINS_PER_SIDE, -amplitude))
    )
    return x[pop.select_symmetric]


def neutral_seed(pop: Population) -> np.ndarray:
    return (pop.sigma * pop.eta)[pop.select_symmetric]


def symmetric_system(
    u: np.ndarray, alpha3: float, pop: Population
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = full_from_symmetric(u, pop)
    update, update_jacobian, dalpha3 = update_and_jacobian(x, alpha3, pop)
    residual = (update - x)[pop.select_symmetric]
    fixed_point_jacobian = update_jacobian - np.eye(pop.dimension)
    reduced_jacobian = (
        fixed_point_jacobian[pop.select_symmetric] @ pop.expand_symmetric
    )
    return residual, reduced_jacobian, dalpha3[pop.select_symmetric], update_jacobian


def solve_symmetric(u0: np.ndarray, alpha3: float, pop: Population) -> np.ndarray:
    answer = root(
        lambda u: symmetric_system(u, alpha3, pop)[0],
        np.asarray(u0, dtype=float),
        jac=lambda u: symmetric_system(u, alpha3, pop)[1],
        method="hybr",
        options={"xtol": ROOT_XTOL},
    )
    residual = float(np.max(np.abs(symmetric_system(answer.x, alpha3, pop)[0])))
    if residual > ROOT_RESIDUAL_TOL:
        raise RuntimeError(
            f"Symmetric solve failed at fs={pop.fs:.3f}, alpha3={alpha3:.8f}; "
            f"residual={residual:.3e}."
        )
    return answer.x


def sector_eigenvalue(
    u: np.ndarray, alpha3: float, pop: Population, sector: str
) -> float:
    x = full_from_symmetric(u, pop)
    update_jacobian = update_and_jacobian(x, alpha3, pop)[1]
    fixed_point_jacobian = update_jacobian - np.eye(pop.dimension)
    basis = pop.q_even if sector == "even" else pop.q_odd
    eigenvalues = np.linalg.eigvals(basis.T @ fixed_point_jacobian @ basis)
    chosen = eigenvalues[np.argmin(np.abs(eigenvalues))]
    if abs(chosen.imag) > 2e-7:
        raise RuntimeError(f"Unexpected complex critical eigenvalue: {chosen}")
    return float(chosen.real)


def refine_critical_point(
    u0: np.ndarray, alpha0: float, pop: Population, sector: str
) -> tuple[float, np.ndarray]:
    scale = np.sqrt(pop.symmetric_dimension)
    initial = np.concatenate((np.asarray(u0) / scale, [float(alpha0)]))

    def critical_system(z: np.ndarray) -> np.ndarray:
        u = scale * z[:-1]
        alpha3 = float(z[-1])
        fixed_residual = symmetric_system(u, alpha3, pop)[0]
        critical_eigenvalue = sector_eigenvalue(u, alpha3, pop, sector)
        return np.concatenate((fixed_residual, [critical_eigenvalue]))

    answer = root(
        critical_system,
        initial,
        method="hybr",
        options={
            "xtol": 2e-9,
            "maxfev": 100 * (pop.symmetric_dimension + 1),
        },
    )
    residual = float(np.max(np.abs(critical_system(answer.x))))
    if residual > 1e-6:
        raise RuntimeError(
            f"Critical solve failed at fs={pop.fs:.3f}, sector={sector}; "
            f"residual={residual:.3e}."
        )
    return float(answer.x[-1]), scale * answer.x[:-1]


def initial_critical_points() -> dict[str, tuple[float, np.ndarray]]:
    seed_fs = 0.35
    pop = make_population(seed_fs)

    pitch_u = solve_symmetric(neutral_seed(pop), 0.50, pop)
    pitch_alpha, pitch_u = refine_critical_point(
        pitch_u, 0.45, pop, sector="odd"
    )

    saddle_u = solve_symmetric(symmetric_seed(pop, 0.90), 0.02, pop)
    saddle_alpha, saddle_u = refine_critical_point(
        saddle_u, 0.05, pop, sector="even"
    )
    return {
        "Saddle-node": (saddle_alpha, saddle_u),
        "Pitchfork": (pitch_alpha, pitch_u),
    }


def continue_boundary(
    event: str, sector: str, seed_alpha: float, seed_u: np.ndarray
) -> dict[float, tuple[float, np.ndarray]]:
    seed_fs = 0.35
    results = {seed_fs: (float(seed_alpha), seed_u.copy())}
    upward = FS_VALUES[FS_VALUES > seed_fs]
    downward = FS_VALUES[FS_VALUES < seed_fs][::-1]
    for fs_sequence in (upward, downward):
        alpha3, u = float(seed_alpha), seed_u.copy()
        for fs in fs_sequence:
            pop = make_population(float(fs))
            try:
                alpha3, u = refine_critical_point(u, alpha3, pop, sector)
            except RuntimeError as error:
                raise RuntimeError(
                    f"{event} continuation stopped at fs={fs:.3f}."
                ) from error
            results[float(fs)] = (float(alpha3), u.copy())
    return results


def run_boundaries() -> None:
    seeds = initial_critical_points()
    boundaries = {
        "Saddle-node": continue_boundary(
            "Saddle-node", "even", *seeds["Saddle-node"]
        ),
        "Pitchfork": continue_boundary(
            "Pitchfork", "odd", *seeds["Pitchfork"]
        ),
    }

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for fs in FS_VALUES:
        fs_value = float(fs)
        saddle_alpha, saddle_u = boundaries["Saddle-node"][fs_value]
        pitch_alpha, pitch_u = boundaries["Pitchfork"][fs_value]
        wide_rows.append(
            {
                "fs": fs_value,
                "saddle_node_alpha3": saddle_alpha,
                "pitchfork_alpha3": pitch_alpha,
            }
        )
        for event, sector, alpha3, u in (
            ("Saddle-node", "T-even", saddle_alpha, saddle_u),
            ("Pitchfork", "T-odd", pitch_alpha, pitch_u),
        ):
            pop = make_population(fs_value)
            x = full_from_symmetric(u, pop)
            fixed_residual = float(
                np.max(np.abs(symmetric_system(u, alpha3, pop)[0]))
            )
            update_jacobian = update_and_jacobian(x, alpha3, pop)[1]
            multipliers = np.linalg.eigvals(update_jacobian)
            closest = int(np.argmin(np.abs(multipliers - 1.0)))
            noncritical = np.delete(multipliers, closest)
            b_value, s_value = observables(x, pop)
            long_rows.append(
                {
                    "fs": fs_value,
                    "event": event,
                    "alpha3": alpha3,
                    "critical_sector": sector,
                    "fixed_point_residual": fixed_residual,
                    "critical_multiplier_real": float(np.real(multipliers[closest])),
                    "critical_multiplier_imag": float(np.imag(multipliers[closest])),
                    "largest_noncritical_modulus": float(
                        np.max(np.abs(noncritical))
                    ),
                    "B": b_value,
                    "S": s_value,
                }
            )
    write_rows(BOUNDARY_WIDE_CSV, wide_rows)
    write_rows(BOUNDARY_LONG_CSV, long_rows)
    print(f"Saved {BOUNDARY_WIDE_CSV}")
    print(f"Saved {BOUNDARY_LONG_CSV}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def rgba_from_observables(b_values: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    b = np.clip((b_values - B_MIN) / (B_MAX - B_MIN), 0.0, 1.0)
    s = np.clip((s_values - S_MIN) / (S_MAX - S_MIN), 0.0, 1.0)
    rgb = (
        (1.0 - b)[..., None] * (1.0 - s)[..., None] * COLOR_LOW_LOW
        + b[..., None] * (1.0 - s)[..., None] * COLOR_HIGH_B
        + (1.0 - b)[..., None] * s[..., None] * COLOR_HIGH_S
        + b[..., None] * s[..., None] * COLOR_HIGH_BS
    )
    return np.concatenate((rgb, np.full(rgb.shape[:-1] + (1,), 0.94)), axis=-1)


def load_classes() -> dict[tuple[float, float], list[dict[str, float]]]:
    grouped: dict[tuple[float, float], dict[int, dict[str, float]]] = defaultdict(dict)
    with ATTRACTOR_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (float(row["alpha3"]), float(row["fs"]))
            class_id = int(row["observable_class"])
            grouped[key].setdefault(
                class_id,
                {"B": float(row["B"]), "abs_S": float(row["abs_S"])},
            )
    return {
        key: sorted(values.values(), key=lambda item: (item["abs_S"], item["B"]))
        for key, values in grouped.items()
    }


def split_cell_collection(classes_by_point) -> PolyCollection:
    half_alpha = 0.5 * ALPHA3_STEP
    half_fs = 0.5 * FS_STEP
    polygons, colors = [], []
    for fs in FS_VALUES:
        for alpha3 in ALPHA3_VALUES:
            classes = classes_by_point.get((float(alpha3), float(fs)), [])
            if not classes:
                continue
            left, right = alpha3 - half_alpha, alpha3 + half_alpha
            bottom, top = fs - half_fs, fs + half_fs
            class_colors = rgba_from_observables(
                np.asarray([item["B"] for item in classes]),
                np.asarray([item["abs_S"] for item in classes]),
            )
            if len(classes) == 1:
                polygons.append(
                    [(left, bottom), (right, bottom), (right, top), (left, top)]
                )
                colors.append(class_colors[0])
            else:
                polygons.extend(
                    [
                        [(left, bottom), (right, bottom), (right, top)],
                        [(left, bottom), (right, top), (left, top)],
                    ]
                )
                colors.extend([class_colors[0], class_colors[-1]])
    return PolyCollection(
        polygons,
        facecolors=np.asarray(colors),
        edgecolors=(1.0, 1.0, 1.0, 0.58),
        linewidths=0.25,
        antialiased=True,
        rasterized=True,
        zorder=1,
    )


def load_visible_boundaries() -> dict[str, dict[str, np.ndarray | float]]:
    rows: list[dict[str, float]] = []
    with BOUNDARY_WIDE_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "fs": float(row["fs"]),
                    "Saddle-node": float(row["saddle_node_alpha3"]),
                    "Pitchfork": float(row["pitchfork_alpha3"]),
                }
            )

    curves: dict[str, dict[str, np.ndarray | float]] = {}
    for event in ("Saddle-node", "Pitchfork"):
        fs_all = np.asarray([row["fs"] for row in rows])
        alpha_all = np.asarray([row[event] for row in rows])
        crossing_indices = np.where(
            (alpha_all[:-1] < 0.0) & (alpha_all[1:] >= 0.0)
        )[0]
        if crossing_indices.size:
            index = int(crossing_indices[-1])
            entry_fs = float(
                np.interp(
                    0.0,
                    alpha_all[index:index + 2],
                    fs_all[index:index + 2],
                )
            )
        else:
            nonnegative = np.where(alpha_all >= 0.0)[0]
            entry_fs = float(fs_all[nonnegative[0]])

        visible = (alpha_all >= 0.0) & (alpha_all <= ALPHA3_MAX)
        fs_visible = np.concatenate(([entry_fs], fs_all[visible]))
        alpha_visible = np.concatenate(([0.0], alpha_all[visible]))
        unique = np.r_[True, np.diff(fs_visible) > 1e-12]
        curves[event] = {
            "fs": fs_visible[unique],
            "alpha3": alpha_visible[unique],
            "entry_fs": entry_fs,
        }
    return curves


def draw_boundary(ax, curve: dict[str, object], event: str) -> None:
    fs_values = np.asarray(curve["fs"])
    alpha_values = np.asarray(curve["alpha3"])
    fs_dense = np.linspace(fs_values[0], fs_values[-1], 700)
    alpha_dense = np.interp(fs_dense, fs_values, alpha_values)
    ax.plot(
        alpha_dense,
        fs_dense,
        color=BOUNDARY_COLORS[event],
        linewidth=BOUNDARY_LINEWIDTH,
        linestyle="--",
        dashes=BOUNDARY_DASHES,
        dash_capstyle="round",
        path_effects=[
            path_effects.Stroke(
                linewidth=BOUNDARY_OUTLINE_WIDTH,
                foreground="black",
                alpha=0.55,
            ),
            path_effects.Normal(),
        ],
        zorder=6,
    )


def format_axis(ax) -> None:
    ax.set_xlim(ALPHA3_MIN, ALPHA3_MAX)
    ax.set_ylim(FS_MIN, FS_MAX)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7], ["0", "0.2", "0.4", "0.6", "0.7"])
    ax.set_yticks(
        [0.15, 0.25, 0.35, 0.45, 0.55],
        ["0.15", "0.25", "0.35", "0.45", "0.55"],
    )
    ax.set_xlabel(r"$\alpha_3$", fontsize=24)
    ax.set_ylabel(r"$f_s$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.grid(linestyle=":", color="black", alpha=0.16, linewidth=0.8, zorder=0)


def add_color_key(ax) -> None:
    key_ax = ax.inset_axes([0.68, 0.105, 0.285, 0.25])
    b_axis = np.linspace(B_MIN, B_MAX, 121)
    s_axis = np.linspace(S_MIN, S_MAX, 121)
    b_grid, s_grid = np.meshgrid(b_axis, s_axis)
    key_ax.imshow(
        rgba_from_observables(b_grid, s_grid),
        origin="lower",
        aspect="auto",
        extent=(B_MIN, B_MAX, S_MIN, S_MAX),
        interpolation="nearest",
    )
    key_ax.set_xlabel(r"$B$", fontsize=16, labelpad=1)
    key_ax.set_ylabel(r"$|S|$", fontsize=16, labelpad=1)
    key_ax.set_xticks([0.3, 0.5, 1.0], ["0.3", "0.5", "1"])
    key_ax.set_yticks([0.0, 0.6], ["0", "0.6"])
    key_ax.tick_params(axis="both", labelsize=12, pad=1)


def boundary_legend() -> list[Line2D]:
    return [
        Line2D(
            [0], [0],
            color=BOUNDARY_COLORS[event],
            linewidth=BOUNDARY_LINEWIDTH,
            linestyle="--",
            dashes=BOUNDARY_DASHES,
            label=("Saddle-node" if event == "Saddle-node" else
                   "Supercritical pitchfork"),
        )
        for event in ("Saddle-node", "Pitchfork")
    ]


def build_state_map(curves):
    alpha_grid = np.linspace(ALPHA3_MIN, ALPHA3_MAX, 1401)
    fs_grid = np.linspace(FS_MIN, FS_MAX, 1001)
    states = np.full((fs_grid.size, alpha_grid.size), 2, dtype=int)
    saddle, pitchfork = curves["Saddle-node"], curves["Pitchfork"]
    for row, fs in enumerate(fs_grid):
        if fs >= float(pitchfork["entry_fs"]):
            pitch_alpha = float(
                np.interp(fs, pitchfork["fs"], pitchfork["alpha3"])
            )
            states[row, alpha_grid < pitch_alpha] = 1
        if fs >= float(saddle["entry_fs"]):
            saddle_alpha = float(
                np.interp(fs, saddle["fs"], saddle["alpha3"])
            )
            states[row, alpha_grid < saddle_alpha] = 0
    return alpha_grid, fs_grid, states


def draw_observable_panel(ax, classes, curves, add_legend=True) -> None:
    ax.add_collection(split_cell_collection(classes))
    draw_boundary(ax, curves["Saddle-node"], "Saddle-node")
    draw_boundary(ax, curves["Pitchfork"], "Pitchfork")
    if add_legend:
        handles = boundary_legend()
        handles.append(Patch(facecolor="white", edgecolor="black", label="Split cells: coexistence"))
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=10.5,
            framealpha=0.95,
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=2.5,
        )
    add_color_key(ax)
    format_axis(ax)


def draw_state_panel(ax, curves) -> None:
    alpha_grid, fs_grid, states = build_state_map(curves)
    labels = (
        "Coexistence\n(biased unimodal and bimodal)",
        "Biased unimodal",
        "Neutral unimodal",
    )
    cmap = ListedColormap([STATE_COLORS[label] for label in labels])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
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
    draw_boundary(ax, curves["Saddle-node"], "Saddle-node")
    draw_boundary(ax, curves["Pitchfork"], "Pitchfork")
    ax.legend(
        handles=[
            Patch(
                facecolor=STATE_COLORS[label],
                edgecolor="black",
                linewidth=0.9,
                label=label,
            )
            for label in labels
        ],
        loc="lower right",
        fontsize=11,
        framealpha=0.95,
        borderpad=0.45,
        labelspacing=0.35,
    )
    format_axis(ax)


def plot_all() -> None:
    classes = load_classes()
    curves = load_visible_boundaries()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=180)
    draw_observable_panel(axes[0], classes, curves)
    draw_state_panel(axes[1], curves)
    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=22, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.15, top=0.95, wspace=0.22)
    fig.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(COMBINED_PDF, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.8), dpi=180)
    draw_observable_panel(ax, classes, curves)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(OBSERVABLE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OBSERVABLE_PDF, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 5.8), dpi=180)
    draw_state_panel(ax, curves)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(STATE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(STATE_PDF, bbox_inches="tight")
    plt.close(fig)

    for path in (
        COMBINED_PNG, COMBINED_PDF, OBSERVABLE_PNG, OBSERVABLE_PDF,
        STATE_PNG, STATE_PDF,
    ):
        print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-boundaries", action="store_true")
    args = parser.parse_args()

    if not args.skip_scan:
        run_stable_scan(max(1, args.workers))
    elif not ATTRACTOR_CSV.exists():
        raise FileNotFoundError(ATTRACTOR_CSV)

    if not args.skip_boundaries:
        run_boundaries()
    elif not BOUNDARY_WIDE_CSV.exists():
        raise FileNotFoundError(BOUNDARY_WIDE_CSV)

    plot_all()


if __name__ == "__main__":
    main()
