"""Compute the two bifurcation boundaries of the 42-D mean-field model.

The model contains two stubborn groups and 20 eta bins on each side of the
non-stubborn population, giving 42 state variables in total.  Only the two
boundaries used in the manuscript are calculated:

1. the T-even saddle-node boundary of the symmetric bimodal branch;
2. the T-odd pitchfork boundary of the symmetric neutral branch.

The plotting domain and grid match the existing 42-D state map:

    alpha2 in [0.00, 0.30], step 0.01
    fs     in [0.15, 0.55], step 0.01

The critical alpha2 values are obtained by continuous root refinement; they
are not rounded to the background-grid spacing.  Values outside the plotted
alpha2 interval are written as NaN in the wide CSV.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


# -----------------------------------------------------------------------------
# Parameter domain: identical to the previous 42-D alpha2--fs state map.
# -----------------------------------------------------------------------------
ALPHA2_MIN = 0.00
ALPHA2_MAX = 0.30
ALPHA2_STEP = 0.01
FS_MIN = 0.15
FS_MAX = 0.55
FS_STEP = 0.01

ALPHA2_VALUES = np.round(
    np.arange(ALPHA2_MIN, ALPHA2_MAX + 0.5 * ALPHA2_STEP, ALPHA2_STEP), 12
)
FS_VALUES = np.round(
    np.arange(FS_MIN, FS_MAX + 0.5 * FS_STEP, FS_STEP), 12
)

# Model parameters.
GAMMA = 4.0
DC = 0.5
ALPHA3 = 0.0
SIGMA_S = 0.8
SIGMA_NS = 0.1
STUBBORN_POSITIVE_SHARE = float(
    os.environ.get("MF_STUBBORN_POSITIVE_SHARE", "0.5")
)
if not 0.0 < STUBBORN_POSITIVE_SHARE < 1.0:
    raise ValueError("MF_STUBBORN_POSITIVE_SHARE must lie strictly between 0 and 1.")

# By default: 2 stubborn groups + 20 positive-eta bins + 20 negative-eta
# bins = 42.  The environment override is used for bin-convergence checks
# (for example, MF_BINS_PER_SIDE=30 gives the 62-bin discretisation) while
# preserving the manuscript's 42-bin default.
BINS_PER_SIDE = int(os.environ.get("MF_BINS_PER_SIDE", "20"))
STATE_DIMENSION = 2 + 2 * BINS_PER_SIDE

# Root tolerances.
ROOT_XTOL = 2e-10
ROOT_RESIDUAL_TOL = 1e-7

# A well-conditioned point inside both boundary families is used only to
# initialise continuation.  Every output point is then solved independently
# from the 42-D critical equations at its own fs value.
SEED_FS = 0.35
SEED_PITCHFORK_ALPHA2 = 0.157
SEED_SADDLE_ALPHA2 = 0.0194
SEED_SADDLE_STABLE_ALPHA2 = 0.018

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_WIDE_CSV = SCRIPT_DIR / "bifurcation_boundaries_42D.csv"
OUTPUT_LONG_CSV = SCRIPT_DIR / "bifurcation_boundaries_42D_long.csv"
OUTPUT_PNG = SCRIPT_DIR / "bifurcation_boundaries_42D.png"
OUTPUT_PDF = SCRIPT_DIR / "bifurcation_boundaries_42D.pdf"


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
    """Create the 42-D eta-resolved population at one stubborn fraction."""
    n = BINS_PER_SIDE
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    eta = np.concatenate(([1.0, -1.0], eta_positive, eta_negative))
    sigma = np.concatenate(([SIGMA_S, SIGMA_S], np.full(2 * n, SIGMA_NS)))
    weights = np.concatenate(
        (
            [
                fs * STUBBORN_POSITIVE_SHARE,
                fs * (1.0 - STUBBORN_POSITIVE_SHARE),
            ],
            np.full(2 * n, (1.0 - fs) / (2.0 * n)),
        )
    )

    # A sign-reflection-symmetric fixed point has
    # x_S- = -x_S+ and x_-(eta) = -x_+(-eta).  It therefore needs only
    # 1 + BINS_PER_SIDE independent coordinates.
    d = 2 + 2 * n
    m = 1 + n
    expand = np.zeros((d, m))
    q_even = np.zeros((d, m))
    q_odd = np.zeros((d, m))

    expand[0, 0] = 1.0
    expand[1, 0] = -1.0
    q_even[0, 0] = 1.0 / np.sqrt(2.0)
    q_even[1, 0] = -1.0 / np.sqrt(2.0)
    q_odd[0, 0] = 1.0 / np.sqrt(2.0)
    q_odd[1, 0] = 1.0 / np.sqrt(2.0)

    for k in range(n):
        positive = 2 + k
        negative = 2 + n + (n - 1 - k)
        expand[positive, 1 + k] = 1.0
        expand[negative, 1 + k] = -1.0
        q_even[positive, 1 + k] = 1.0 / np.sqrt(2.0)
        q_even[negative, 1 + k] = -1.0 / np.sqrt(2.0)
        q_odd[positive, 1 + k] = 1.0 / np.sqrt(2.0)
        q_odd[negative, 1 + k] = 1.0 / np.sqrt(2.0)

    select = np.concatenate(([0], np.arange(2, 2 + n)))
    return Population(
        fs=float(fs),
        eta=eta,
        sigma=sigma,
        weights=weights,
        select_symmetric=select,
        expand_symmetric=expand,
        q_even=q_even,
        q_odd=q_odd,
    )


def full_from_symmetric(u: np.ndarray, pop: Population) -> np.ndarray:
    return pop.expand_symmetric @ np.asarray(u, dtype=float)


def symmetric_seed(pop: Population, amplitude: float) -> np.ndarray:
    x = np.concatenate(
        (
            [1.0, -1.0],
            np.full(BINS_PER_SIDE, amplitude),
            np.full(BINS_PER_SIDE, -amplitude),
        )
    )
    return x[pop.select_symmetric]


def neutral_seed(pop: Population) -> np.ndarray:
    return (pop.sigma * pop.eta)[pop.select_symmetric]


def update_jacobian_alpha(
    x: np.ndarray, alpha2: float, pop: Population
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the expected update, d(update)/dx and d(update)/dalpha2."""
    w = pop.weights
    difference = x[:, None] - x[None, :]
    distance = np.abs(difference)
    exponent = np.clip(GAMMA * (distance - DC), -100.0, 100.0)
    logistic = 1.0 / (1.0 + np.exp(exponent))
    prefactor = 1.0 + np.exp(-GAMMA * DC)

    confidence = prefactor * logistic
    target_extremity = np.abs(x)[None, :]
    source_extremity = np.abs(x)[:, None]
    alpha1 = 1.0 - alpha2 - ALPHA3
    connection = (
        alpha1 * confidence
        + alpha2 * target_extremity
        + ALPHA3 * source_extremity
    )

    denominator = connection @ w
    if np.any(denominator <= 1e-14):
        raise RuntimeError("Expected-neighbour denominator reached zero.")
    numerator = connection @ (w * x)
    neighbour_mean = numerator / denominator
    update = pop.sigma * pop.eta + (1.0 - pop.sigma) * neighbour_mean

    sign_difference = np.sign(difference)
    dconfidence_dr = -GAMMA * prefactor * logistic * (1.0 - logistic)
    dconnection_target = (
        alpha1 * dconfidence_dr * sign_difference
        + ALPHA3 * np.sign(x)[:, None]
    )
    dconnection_source = (
        -alpha1 * dconfidence_dr * sign_difference
        + alpha2 * np.sign(x)[None, :]
    )

    ddenominator = dconnection_source * w[None, :]
    dnumerator = w[None, :] * (
        dconnection_source * x[None, :] + connection
    )
    diagonal = np.arange(x.size)
    ddenominator[diagonal, diagonal] += dconnection_target @ w
    dnumerator[diagonal, diagonal] += dconnection_target @ (w * x)

    jacobian = (1.0 - pop.sigma)[:, None] * (
        dnumerator * denominator[:, None]
        - numerator[:, None] * ddenominator
    ) / denominator[:, None] ** 2

    dconnection_alpha = target_extremity - confidence
    ddenominator_alpha = dconnection_alpha @ w
    dnumerator_alpha = dconnection_alpha @ (w * x)
    dalpha = (1.0 - pop.sigma) * (
        dnumerator_alpha * denominator - numerator * ddenominator_alpha
    ) / denominator**2
    return update, jacobian, dalpha


def symmetric_system(
    u: np.ndarray, alpha2: float, pop: Population
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = full_from_symmetric(u, pop)
    update, update_jacobian, dalpha = update_jacobian_alpha(x, alpha2, pop)
    residual = (update - x)[pop.select_symmetric]
    fixed_point_jacobian = update_jacobian - np.eye(pop.dimension)
    reduced_jacobian = (
        fixed_point_jacobian[pop.select_symmetric] @ pop.expand_symmetric
    )
    return (
        residual,
        reduced_jacobian,
        dalpha[pop.select_symmetric],
        update_jacobian,
    )


def solve_symmetric(
    u0: np.ndarray, alpha2: float, pop: Population
) -> np.ndarray:
    answer = root(
        lambda u: symmetric_system(u, alpha2, pop)[0],
        np.asarray(u0, dtype=float),
        jac=lambda u: symmetric_system(u, alpha2, pop)[1],
        method="hybr",
        options={"xtol": ROOT_XTOL},
    )
    residual = float(
        np.max(np.abs(symmetric_system(answer.x, alpha2, pop)[0]))
    )
    if residual > ROOT_RESIDUAL_TOL:
        raise RuntimeError(
            f"Symmetric fixed point failed at fs={pop.fs:.3f}, "
            f"alpha2={alpha2:.8f}; residual={residual:.3e}."
        )
    return answer.x


def sector_eigenvalue(
    u: np.ndarray, alpha2: float, pop: Population, sector: str
) -> float:
    """Eigenvalue of d(update-x)/dx closest to zero in one sector."""
    x = full_from_symmetric(u, pop)
    update_jacobian = update_jacobian_alpha(x, alpha2, pop)[1]
    fixed_point_jacobian = update_jacobian - np.eye(pop.dimension)
    basis = pop.q_even if sector == "even" else pop.q_odd
    reduced = basis.T @ fixed_point_jacobian @ basis
    eigenvalues = np.linalg.eigvals(reduced)
    chosen = eigenvalues[np.argmin(np.abs(eigenvalues))]
    if abs(chosen.imag) > 2e-7:
        raise RuntimeError(
            f"Critical {sector} eigenvalue is unexpectedly complex: {chosen}."
        )
    return float(chosen.real)


def refine_critical_point(
    u0: np.ndarray,
    alpha0: float,
    pop: Population,
    sector: str,
) -> tuple[float, np.ndarray]:
    """Solve the fixed-point and critical-eigenvalue equations together."""
    scale = np.sqrt(pop.symmetric_dimension)
    initial = np.concatenate((np.asarray(u0) / scale, [float(alpha0)]))

    def critical_system(z: np.ndarray) -> np.ndarray:
        u = scale * z[:-1]
        alpha2 = float(z[-1])
        fixed_point_residual = symmetric_system(u, alpha2, pop)[0]
        critical_eigenvalue = sector_eigenvalue(u, alpha2, pop, sector)
        return np.concatenate((fixed_point_residual, [critical_eigenvalue]))

    answer = root(
        critical_system,
        initial,
        method="hybr",
        options={
            "xtol": 2e-9,
            "maxfev": 80 * (pop.symmetric_dimension + 1),
        },
    )
    residual = float(np.max(np.abs(critical_system(answer.x))))
    if residual > 1e-6:
        raise RuntimeError(
            f"Critical-point solve failed at fs={pop.fs:.3f}, "
            f"sector={sector}; residual={residual:.3e}."
        )
    return float(answer.x[-1]), scale * answer.x[:-1]


def initial_critical_points() -> dict[str, tuple[float, np.ndarray]]:
    """Construct one 42-D seed on each boundary at fs=0.35."""
    pop = make_population(SEED_FS)

    # Neutral symmetric fixed point, then T-odd pitchfork refinement.
    pitch_u = solve_symmetric(neutral_seed(pop), 0.20, pop)
    pitch_alpha, pitch_u = refine_critical_point(
        pitch_u, SEED_PITCHFORK_ALPHA2, pop, sector="odd"
    )

    # Stable bimodal point just below the fold, then T-even refinement.
    saddle_u = solve_symmetric(
        symmetric_seed(pop, amplitude=0.90),
        SEED_SADDLE_STABLE_ALPHA2,
        pop,
    )
    saddle_alpha, saddle_u = refine_critical_point(
        saddle_u, SEED_SADDLE_ALPHA2, pop, sector="even"
    )
    return {
        "Saddle-node": (saddle_alpha, saddle_u),
        "Pitchfork": (pitch_alpha, pitch_u),
    }


def continue_boundary(
    event: str,
    sector: str,
    seed_alpha: float,
    seed_u: np.ndarray,
) -> dict[float, tuple[float, np.ndarray]]:
    """Continue one critical boundary over the requested fs grid."""
    results = {float(SEED_FS): (float(seed_alpha), seed_u.copy())}
    upward = FS_VALUES[FS_VALUES > SEED_FS]
    downward = FS_VALUES[FS_VALUES < SEED_FS][::-1]

    for fs_sequence in (upward, downward):
        alpha2 = float(seed_alpha)
        u = seed_u.copy()
        for fs in fs_sequence:
            pop = make_population(float(fs))
            try:
                alpha2, u = refine_critical_point(u, alpha2, pop, sector)
            except RuntimeError as error:
                raise RuntimeError(
                    f"{event} continuation stopped at fs={fs:.3f}."
                ) from error
            results[float(fs)] = (float(alpha2), u.copy())
    return results


def observables(x: np.ndarray, pop: Population) -> tuple[float, float]:
    values = x[2:]
    weights = pop.weights[2:] / np.sum(pop.weights[2:])
    mean = float(np.sum(weights * values))
    centered = values - mean
    variance = float(np.sum(weights * centered**2))
    if variance <= 1e-16:
        return np.nan, mean
    skewness = float(np.sum(weights * centered**3) / variance**1.5)
    kurtosis = float(np.sum(weights * centered**4) / variance**2)
    bimodality = float((skewness**2 + 1.0) / kurtosis)
    return bimodality, mean


def diagnostic_row(
    fs: float,
    event: str,
    sector: str,
    alpha2: float,
    u: np.ndarray,
) -> dict[str, object]:
    pop = make_population(fs)
    x = full_from_symmetric(u, pop)
    fixed_point_residual = float(
        np.max(np.abs(symmetric_system(u, alpha2, pop)[0]))
    )
    critical_fixed_point_eigenvalue = sector_eigenvalue(
        u, alpha2, pop, sector
    )
    update_jacobian = update_jacobian_alpha(x, alpha2, pop)[1]
    map_eigenvalues = np.linalg.eigvals(update_jacobian)
    critical_map_multiplier = map_eigenvalues[
        np.argmin(np.abs(map_eigenvalues - 1.0))
    ]
    spectral_radius = float(np.max(np.abs(map_eigenvalues)))
    bimodality, mean_opinion = observables(x, pop)
    return {
        "fs": fs,
        "event": event,
        "alpha2": alpha2,
        "bins_per_side": BINS_PER_SIDE,
        "state_dimension": STATE_DIMENSION,
        "critical_sector": "T-even" if sector == "even" else "T-odd",
        "fixed_point_residual": fixed_point_residual,
        "critical_fixed_point_eigenvalue": critical_fixed_point_eigenvalue,
        "critical_map_multiplier_real": float(np.real(critical_map_multiplier)),
        "critical_map_multiplier_imag": float(np.imag(critical_map_multiplier)),
        "map_spectral_radius": spectral_radius,
        "B": bimodality,
        "S": mean_opinion,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_boundaries(long_rows: list[dict[str, object]]) -> None:
    colors = {"Saddle-node": "#7B2CBF", "Pitchfork": "#D62728"}
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=180)
    for event in ("Saddle-node", "Pitchfork"):
        rows = [row for row in long_rows if row["event"] == event]
        ax.plot(
            [row["alpha2"] for row in rows],
            [row["fs"] for row in rows],
            color=colors[event],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label=f"42D {event.lower()}",
        )

    ax.set_xlim(ALPHA2_MIN, ALPHA2_MAX)
    ax.set_ylim(FS_MIN, FS_MAX)
    ax.set_xlabel(r"$\alpha_2$")
    ax.set_ylabel(r"$f_s$")
    ax.set_xticks(np.arange(ALPHA2_MIN, ALPHA2_MAX + 1e-12, 0.05))
    ax.set_yticks(np.arange(FS_MIN, FS_MAX + 1e-12, 0.05))
    ax.legend(frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not np.any(np.isclose(FS_VALUES, SEED_FS)):
        raise RuntimeError("SEED_FS must lie on the requested fs grid.")

    seeds = initial_critical_points()
    boundaries = {
        "Saddle-node": continue_boundary(
            "Saddle-node", "even", *seeds["Saddle-node"]
        ),
        "Pitchfork": continue_boundary(
            "Pitchfork", "odd", *seeds["Pitchfork"]
        ),
    }

    # The wide table contains every fs grid value.  A NaN means that the
    # continued critical point lies outside the displayed alpha2 interval.
    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for fs in FS_VALUES:
        fs_key = float(fs)
        saddle_alpha, saddle_u = boundaries["Saddle-node"][fs_key]
        pitch_alpha, pitch_u = boundaries["Pitchfork"][fs_key]

        saddle_inside = ALPHA2_MIN <= saddle_alpha <= ALPHA2_MAX
        pitch_inside = ALPHA2_MIN <= pitch_alpha <= ALPHA2_MAX
        wide_rows.append(
            {
                "fs": fs_key,
                "saddle_node_alpha2": saddle_alpha if saddle_inside else np.nan,
                "pitchfork_alpha2": pitch_alpha if pitch_inside else np.nan,
            }
        )

        if saddle_inside:
            long_rows.append(
                diagnostic_row(
                    fs_key,
                    "Saddle-node",
                    "even",
                    saddle_alpha,
                    saddle_u,
                )
            )
        if pitch_inside:
            long_rows.append(
                diagnostic_row(
                    fs_key,
                    "Pitchfork",
                    "odd",
                    pitch_alpha,
                    pitch_u,
                )
            )

    write_csv(OUTPUT_WIDE_CSV, wide_rows)
    write_csv(OUTPUT_LONG_CSV, long_rows)
    plot_boundaries(long_rows)

    print(f"42-D state dimension: {STATE_DIMENSION}")
    print(
        f"Domain: alpha2=[{ALPHA2_MIN:.2f}, {ALPHA2_MAX:.2f}], "
        f"fs=[{FS_MIN:.2f}, {FS_MAX:.2f}], grid step=0.01"
    )
    for event in ("Saddle-node", "Pitchfork"):
        rows = [row for row in long_rows if row["event"] == event]
        print(
            f"{event}: {len(rows)} in-domain points, "
            f"fs={rows[0]['fs']:.2f}--{rows[-1]['fs']:.2f}"
        )
    print(f"Saved {OUTPUT_WIDE_CSV}")
    print(f"Saved {OUTPUT_LONG_CSV}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
