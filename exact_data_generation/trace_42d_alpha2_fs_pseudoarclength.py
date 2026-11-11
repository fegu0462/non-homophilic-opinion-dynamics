"""Trace 42-bin alpha2--fs critical curves by pseudo-arclength continuation.

Unlike a pointwise fs continuation, pseudo-arclength continuation can follow a
critical curve through a turn in fs.  The unknowns are the 21 independent
coordinates of a sign-symmetric fixed point, alpha2, and fs.  The equations
are the 21 fixed-point residuals plus a zero eigenvalue in the requested
symmetry sector.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

import compute_42d_bifurcation_boundaries as base
import distributional_mf_multistability_scatter_42d as search


HERE = Path(__file__).resolve().parent
U_SCALE = np.sqrt(base.BINS_PER_SIDE + 1)
X_SCALE = np.sqrt(base.STATE_DIMENSION)


def pack(u: np.ndarray, alpha2: float, fs: float) -> np.ndarray:
    return np.concatenate((np.asarray(u, dtype=float) / U_SCALE, [alpha2, fs]))


def unpack(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    return U_SCALE * z[:-2], float(z[-2]), float(z[-1])


def critical_residual(z: np.ndarray, sector: str) -> np.ndarray:
    u, alpha2, fs = unpack(z)
    pop = base.make_population(fs)
    fixed = base.symmetric_system(u, alpha2, pop)[0]
    eig = base.sector_eigenvalue(u, alpha2, pop, sector)
    return np.concatenate((fixed, [eig]))


def finite_jacobian(z: np.ndarray, sector: str) -> np.ndarray:
    f0 = critical_residual(z, sector)
    jac = np.empty((f0.size, z.size))
    for column in range(z.size):
        step = 2e-6 * max(1.0, abs(float(z[column])))
        plus = z.copy()
        minus = z.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (
            critical_residual(plus, sector)
            - critical_residual(minus, sector)
        ) / (2.0 * step)
    return jac


def tangent_at(z: np.ndarray, sector: str, previous: np.ndarray | None) -> np.ndarray:
    _, _, vh = np.linalg.svd(finite_jacobian(z, sector), full_matrices=True)
    tangent = vh[-1]
    tangent /= np.linalg.norm(tangent)
    if previous is not None and np.dot(tangent, previous) < 0.0:
        tangent *= -1.0
    return tangent


def correct(
    predictor: np.ndarray,
    tangent: np.ndarray,
    sector: str,
) -> tuple[np.ndarray, float, int]:
    def augmented(z: np.ndarray) -> np.ndarray:
        arc = float(np.dot(z - predictor, tangent))
        return np.concatenate((critical_residual(z, sector), [arc]))

    answer = least_squares(
        augmented,
        predictor,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=1200,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(augmented(answer.x))))
    return answer.x, residual, int(answer.nfev)


def seed_point(event: str) -> tuple[np.ndarray, str]:
    seeds = base.initial_critical_points()
    alpha2, u = seeds[event]
    sector = "even" if event == "Saddle-node" else "odd"
    return pack(u, alpha2, base.SEED_FS), sector


def trace_direction(
    seed: np.ndarray,
    sector: str,
    direction: float,
    ds_initial: float = 0.003,
    max_points: int = 1200,
) -> list[np.ndarray]:
    points = [seed.copy()]
    tangent = tangent_at(seed, sector, None)
    if direction * tangent[-1] < 0.0:
        tangent *= -1.0
    ds = ds_initial

    for _ in range(max_points):
        predictor = points[-1] + ds * tangent
        candidate, residual, nfev = correct(predictor, tangent, sector)
        if residual > 2e-7 or np.linalg.norm(candidate - points[-1]) < 1e-8:
            ds *= 0.5
            if ds < 2e-5:
                break
            continue

        u, alpha2, fs = unpack(candidate)
        if not np.all(np.isfinite(candidate)):
            break
        points.append(candidate)

        new_tangent = tangent_at(candidate, sector, tangent)
        angle = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 6 and angle > 0.995:
            ds = min(ds * 1.25, 0.006)
        elif nfev > 20 or angle < 0.95:
            ds = max(ds * 0.6, 2e-5)

        # Stop only after leaving a generous box around the requested domain.
        if fs < 0.10 or fs > 1.02 or alpha2 < -0.10 or alpha2 > 0.65:
            break
        if np.max(np.abs(u)) > 2.0:
            break
    return points


def diagnostics(event: str, sector: str, z: np.ndarray) -> dict[str, float | str]:
    u, alpha2, fs = unpack(z)
    pop = base.make_population(fs)
    x = base.full_from_symmetric(u, pop)
    update_jacobian = base.update_jacobian_alpha(x, alpha2, pop)[1]
    multipliers = np.linalg.eigvals(update_jacobian)
    critical_index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, critical_index)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": event,
        "sector": sector,
        "fs": fs,
        "alpha2": alpha2,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(base.symmetric_system(u, alpha2, pop)[0]))),
        "sector_zero_eigenvalue": base.sector_eigenvalue(u, alpha2, pop, sector),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
        "min_abs_state": float(np.min(np.abs(x))),
    }


def full_pack(x: np.ndarray, alpha2: float, fs: float) -> np.ndarray:
    return np.concatenate((np.asarray(x, dtype=float) / X_SCALE, [alpha2, fs]))


def full_unpack(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    return X_SCALE * z[:-2], float(z[-2]), float(z[-1])


def full_fold_residual(z: np.ndarray) -> np.ndarray:
    x, alpha2, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, update_jacobian, _ = base.update_jacobian_alpha(x, alpha2, pop)
    eigenvalues = np.linalg.eigvals(update_jacobian - np.eye(pop.dimension))
    chosen = eigenvalues[np.argmin(np.abs(eigenvalues))]
    return np.concatenate((update - x, [float(np.real(chosen))]))


def numerical_jacobian(function, z: np.ndarray) -> np.ndarray:
    f0 = function(z)
    jac = np.empty((f0.size, z.size))
    for column in range(z.size):
        step = 2e-6 * max(1.0, abs(float(z[column])))
        plus = z.copy()
        minus = z.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jac


def generic_tangent(function, z: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    _, _, vh = np.linalg.svd(numerical_jacobian(function, z), full_matrices=True)
    tangent = vh[-1]
    tangent /= np.linalg.norm(tangent)
    if previous is not None and np.dot(tangent, previous) < 0.0:
        tangent *= -1.0
    return tangent


def generic_correct(function, predictor: np.ndarray, tangent: np.ndarray):
    def augmented(z: np.ndarray) -> np.ndarray:
        return np.concatenate((function(z), [np.dot(z - predictor, tangent)]))

    answer = least_squares(
        augmented,
        predictor,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=1600,
        x_scale="jac",
    )
    return answer.x, float(np.max(np.abs(augmented(answer.x)))), int(answer.nfev)


def canonical_seed_bank() -> list[np.ndarray]:
    return [
        search.make_seed("split", 0.02),
        search.make_seed("split", 0.50),
        search.make_seed("split", 0.90),
        search.make_seed("split", 0.50, 0.02),
        search.make_seed("split", 0.50, -0.02),
        search.make_seed("consensus", 0.60),
        search.make_seed("consensus", -0.60),
        search.make_seed("eta", 1.00),
    ]


def asymmetric_fold_seed() -> np.ndarray:
    search.BASE_SEEDS = canonical_seed_bank()
    fs = 0.80
    attractors = search.stable_attractors(0.185, fs, ())
    biased = [item for item in attractors if float(item["S"]) > 0.20]
    if not biased:
        raise RuntimeError("Could not locate the positive biased seed at fs=0.8.")
    x0 = max(biased, key=lambda item: float(item["S"]))["solution"]
    initial = np.concatenate((np.asarray(x0, dtype=float) / X_SCALE, [0.190]))

    def fixed_fs_residual(z: np.ndarray) -> np.ndarray:
        packed = np.concatenate((z[:-1], [z[-1], fs]))
        return full_fold_residual(packed)

    answer = least_squares(
        fixed_fs_residual,
        initial,
        xtol=2e-13,
        ftol=2e-13,
        gtol=2e-13,
        max_nfev=3000,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(fixed_fs_residual(answer.x))))
    if residual > 2e-7:
        raise RuntimeError(f"Asymmetric-fold seed failed; residual={residual:.3e}.")
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def secondary_asymmetric_fold_seed() -> np.ndarray:
    """Seed the high-B, weakly asymmetric fold visible at high f_s."""
    search.BASE_SEEDS = canonical_seed_bank()
    fs = 0.85
    previous: list[np.ndarray] = []
    attractors = []
    for alpha2 in np.arange(0.15, 0.205 + 1e-12, 0.005):
        attractors = search.stable_attractors(float(alpha2), fs, previous)
        previous = [item["solution"] for item in attractors]
    biased = [item for item in attractors if float(item["S"]) > 1e-3]
    if not biased:
        raise RuntimeError("Could not locate the weakly asymmetric seed at fs=0.85.")
    x0 = min(biased, key=lambda item: float(item["S"]))["solution"]
    initial = np.concatenate((np.asarray(x0) / X_SCALE, [0.208]))

    def fixed_fs_residual(z: np.ndarray) -> np.ndarray:
        return full_fold_residual(np.concatenate((z[:-1], [z[-1], fs])))

    answer = least_squares(
        fixed_fs_residual,
        initial,
        xtol=2e-13,
        ftol=2e-13,
        gtol=2e-13,
        max_nfev=3000,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(fixed_fs_residual(answer.x))))
    if residual > 2e-7:
        raise RuntimeError(f"Secondary fold seed failed; residual={residual:.3e}.")
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def trace_generic_direction(
    function,
    seed: np.ndarray,
    direction: float,
    ds_initial: float = 0.002,
    max_points: int = 800,
) -> list[np.ndarray]:
    points = [seed.copy()]
    tangent = generic_tangent(function, seed, None)
    if direction * tangent[-1] < 0.0:
        tangent *= -1.0
    ds = ds_initial
    for _ in range(max_points):
        predictor = points[-1] + ds * tangent
        candidate, residual, nfev = generic_correct(function, predictor, tangent)
        if residual > 2e-7 or np.linalg.norm(candidate - points[-1]) < 1e-8:
            ds *= 0.5
            if ds < 2e-5:
                break
            continue
        x, alpha2, fs = full_unpack(candidate)
        if not np.all(np.isfinite(candidate)):
            break
        points.append(candidate)
        new_tangent = generic_tangent(function, candidate, tangent)
        angle = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 6 and angle > 0.995:
            ds = min(ds * 1.25, 0.005)
        elif nfev > 20 or angle < 0.95:
            ds = max(ds * 0.6, 2e-5)
        if fs < 0.10 or fs > 1.02 or alpha2 < -0.10 or alpha2 > 0.65:
            break
        if np.max(np.abs(x)) > 2.0:
            break
    return points


def full_fold_diagnostics(
    z: np.ndarray, event: str = "Asymmetric saddle-node"
) -> dict[str, float | str]:
    x, alpha2, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, update_jacobian, _ = base.update_jacobian_alpha(x, alpha2, pop)
    multipliers = np.linalg.eigvals(update_jacobian)
    critical_index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, critical_index)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": event,
        "sector": "full",
        "fs": fs,
        "alpha2": alpha2,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "sector_zero_eigenvalue": float(np.real(multipliers[critical_index] - 1.0)),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
        "min_abs_state": float(np.min(np.abs(x))),
    }


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for event in ("Saddle-node", "Pitchfork"):
        seed, sector = seed_point(event)
        downward = trace_direction(seed, sector, -1.0)
        upward = trace_direction(seed, sector, +1.0)
        curve = list(reversed(downward[1:])) + upward
        event_rows = [diagnostics(event, sector, point) for point in curve]
        rows.extend(event_rows)
        fs_values = np.asarray([float(row["fs"]) for row in event_rows])
        alpha_values = np.asarray([float(row["alpha2"]) for row in event_rows])
        print(
            f"{event}: {len(event_rows)} points, "
            f"fs=[{fs_values.min():.6f}, {fs_values.max():.6f}], "
            f"alpha2=[{alpha_values.min():.6f}, {alpha_values.max():.6f}]"
        )

    fold_seed = asymmetric_fold_seed()
    downward = trace_generic_direction(full_fold_residual, fold_seed, -1.0)
    upward = trace_generic_direction(full_fold_residual, fold_seed, +1.0)
    fold_curve = list(reversed(downward[1:])) + upward
    fold_rows = [full_fold_diagnostics(point) for point in fold_curve]
    rows.extend(fold_rows)
    fold_fs = np.asarray([float(row["fs"]) for row in fold_rows])
    fold_alpha = np.asarray([float(row["alpha2"]) for row in fold_rows])
    print(
        f"Asymmetric saddle-node: {len(fold_rows)} points, "
        f"fs=[{fold_fs.min():.6f}, {fold_fs.max():.6f}], "
        f"alpha2=[{fold_alpha.min():.6f}, {fold_alpha.max():.6f}]"
    )

    secondary_seed = secondary_asymmetric_fold_seed()
    downward = trace_generic_direction(full_fold_residual, secondary_seed, -1.0)
    upward = trace_generic_direction(full_fold_residual, secondary_seed, +1.0)
    secondary_curve = list(reversed(downward[1:])) + upward
    secondary_rows = [
        full_fold_diagnostics(point, "Asymmetric saddle-node II")
        for point in secondary_curve
    ]
    rows.extend(secondary_rows)
    second_fs = np.asarray([float(row["fs"]) for row in secondary_rows])
    second_alpha = np.asarray([float(row["alpha2"]) for row in secondary_rows])
    print(
        f"Asymmetric saddle-node II: {len(secondary_rows)} points, "
        f"fs=[{second_fs.min():.6f}, {second_fs.max():.6f}], "
        f"alpha2=[{second_alpha.min():.6f}, {second_alpha.max():.6f}]"
    )

    output = HERE / "bifurcation_boundaries_42D_pseudoarclength.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
