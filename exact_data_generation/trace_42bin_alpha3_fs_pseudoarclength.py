"""Trace the 42-bin alpha3--fs bifurcation curves continuously.

The continuation unknown is (u, alpha3, fs), where u contains the independent
coordinates of a sign-symmetric fixed point.  The equations are the symmetric
fixed-point residual and a zero critical eigenvalue in either the even sector
(saddle-node) or odd sector (pitchfork).  A pseudo-arclength constraint lets
the calculation pass folds in fs, which a row-by-row parameter scan cannot.
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

import compute_plot_42bin_alpha3_fs as base


HERE = Path(__file__).resolve().parent
B_CUTOFF = float(os.environ.get("B_CUTOFF", "0.57"))
B_TAG = f"B{int(round(100 * B_CUTOFF)):03d}"
OUTPUT = HERE / f"alpha3_fs_42bin_{B_TAG}_pseudoarclength_boundaries.csv"
U_SCALE = np.sqrt(base.make_population(0.5).symmetric_dimension)
X_SCALE = np.sqrt(base.make_population(0.5).dimension)


def pack(u: np.ndarray, alpha3: float, fs: float) -> np.ndarray:
    return np.concatenate((np.asarray(u, dtype=float) / U_SCALE, [alpha3, fs]))


def unpack(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    return U_SCALE * z[:-2], float(z[-2]), float(z[-1])


def critical_residual(z: np.ndarray, sector: str) -> np.ndarray:
    u, alpha3, fs = unpack(z)
    pop = base.make_population(fs)
    fixed = base.symmetric_system(u, alpha3, pop)[0]
    critical = base.sector_eigenvalue(u, alpha3, pop, sector)
    return np.concatenate((fixed, [critical]))


def finite_jacobian(z: np.ndarray, sector: str) -> np.ndarray:
    f0 = critical_residual(z, sector)
    jacobian = np.empty((f0.size, z.size))
    for column in range(z.size):
        step = 1.5e-6 * max(1.0, abs(float(z[column])))
        plus = z.copy()
        minus = z.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            critical_residual(plus, sector)
            - critical_residual(minus, sector)
        ) / (2.0 * step)
    return jacobian


def tangent_at(
    z: np.ndarray, sector: str, previous: np.ndarray | None
) -> np.ndarray:
    _, _, vh = np.linalg.svd(finite_jacobian(z, sector), full_matrices=True)
    tangent = vh[-1]
    tangent /= np.linalg.norm(tangent)
    if previous is not None and np.dot(tangent, previous) < 0.0:
        tangent *= -1.0
    return tangent


def correct(
    predictor: np.ndarray, tangent: np.ndarray, sector: str
) -> tuple[np.ndarray, float, int]:
    def augmented(z: np.ndarray) -> np.ndarray:
        arclength = float(np.dot(z - predictor, tangent))
        return np.concatenate((critical_residual(z, sector), [arclength]))

    answer = least_squares(
        augmented,
        predictor,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=1000,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(augmented(answer.x))))
    return answer.x, residual, int(answer.nfev)


def seed_point(event: str) -> tuple[np.ndarray, str]:
    seeds = base.initial_critical_points()
    alpha3, u = seeds[event]
    sector = "even" if event == "Saddle-node" else "odd"
    return pack(u, alpha3, 0.35), sector


def trace_direction(
    seed: np.ndarray,
    sector: str,
    direction: float,
    ds_initial: float = 0.0025,
    max_points: int = 1800,
) -> list[np.ndarray]:
    points = [seed.copy()]
    tangent = tangent_at(seed, sector, None)
    if direction * tangent[-1] < 0.0:
        tangent *= -1.0
    ds = ds_initial

    for _ in range(max_points):
        predictor = points[-1] + ds * tangent
        candidate, residual, nfev = correct(predictor, tangent, sector)
        step_length = float(np.linalg.norm(candidate - points[-1]))
        if residual > 2e-7 or step_length < 1e-8:
            ds *= 0.5
            if ds < 1.5e-5:
                break
            continue

        if not np.all(np.isfinite(candidate)):
            break
        u, alpha3, fs = unpack(candidate)
        points.append(candidate)

        new_tangent = tangent_at(candidate, sector, tangent)
        alignment = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 7 and alignment > 0.995:
            ds = min(ds * 1.22, 0.005)
        elif nfev > 20 or alignment < 0.95:
            ds = max(ds * 0.65, 1.5e-5)

        if fs < 0.05 or fs > 1.05 or alpha3 < -0.12 or alpha3 > 0.90:
            break
        if np.max(np.abs(u)) > 2.0:
            break
        # Prevent cycling indefinitely if a closed component returns to seed.
        if len(points) > 80 and np.linalg.norm(candidate - seed) < 0.35 * ds:
            break
    return points


def full_pack(x: np.ndarray, alpha3: float, fs: float) -> np.ndarray:
    return np.concatenate((np.asarray(x, dtype=float) / X_SCALE, [alpha3, fs]))


def full_unpack(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    return X_SCALE * z[:-2], float(z[-2]), float(z[-1])


def full_fold_residual(z: np.ndarray) -> np.ndarray:
    x, alpha3, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, update_jacobian, _ = base.update_and_jacobian(x, alpha3, pop)
    eigenvalues = np.linalg.eigvals(update_jacobian - np.eye(pop.dimension))
    chosen = eigenvalues[np.argmin(np.abs(eigenvalues))]
    return np.concatenate((update - x, [float(np.real(chosen))]))


def generic_tangent(
    function, z: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    _, _, vh = np.linalg.svd(finite_function_jacobian(function, z), full_matrices=True)
    tangent = vh[-1]
    tangent /= np.linalg.norm(tangent)
    if previous is not None and np.dot(tangent, previous) < 0.0:
        tangent *= -1.0
    return tangent


def finite_function_jacobian(function, z: np.ndarray) -> np.ndarray:
    f0 = function(z)
    jacobian = np.empty((f0.size, z.size))
    for column in range(z.size):
        step = 1.5e-6 * max(1.0, abs(float(z[column])))
        plus = z.copy()
        minus = z.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jacobian


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
    residual = float(np.max(np.abs(augmented(answer.x))))
    return answer.x, residual, int(answer.nfev)


def asymmetric_fold_seed() -> np.ndarray:
    fs = 0.90
    pop = base.make_population(fs)
    attractors = base.stable_attractors(0.44, pop, [])
    biased = [
        item
        for item in attractors
        if float(item["S"]) > 0.10 and float(item["B"]) < B_CUTOFF
    ]
    if not biased:
        raise RuntimeError("Could not locate the low-B biased seed at fs=0.90.")
    x0 = max(biased, key=lambda item: float(item["S"]))["solution"]
    initial = np.concatenate((np.asarray(x0) / X_SCALE, [0.446]))

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
        raise RuntimeError(f"Asymmetric-fold seed failed; residual={residual:.3e}.")
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def asymmetric_right_fold_seed() -> np.ndarray:
    """Locate the right saddle-node branch bounding asymmetric coexistence."""
    fs = 0.75
    alpha3_probe = 0.60
    pop = base.make_population(fs)
    attractors = base.stable_attractors(alpha3_probe, pop, [])
    asymmetric = [item for item in attractors if float(item["S"]) > 1.0e-3]
    if not asymmetric:
        raise RuntimeError("Could not locate the right asymmetric-fold seed.")
    x0 = max(asymmetric, key=lambda item: float(item["S"]))["solution"]
    initial = np.concatenate((np.asarray(x0) / X_SCALE, [0.62]))

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
        raise RuntimeError(
            f"Right asymmetric-fold seed failed; residual={residual:.3e}."
        )
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def trace_generic_direction(
    function,
    seed: np.ndarray,
    direction: float,
    ds_initial: float = 0.0025,
    max_points: int = 1600,
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
            if ds < 1.5e-5:
                break
            continue
        if not np.all(np.isfinite(candidate)):
            break
        x, alpha3, fs = full_unpack(candidate)
        points.append(candidate)
        new_tangent = generic_tangent(function, candidate, tangent)
        alignment = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 7 and alignment > 0.995:
            ds = min(ds * 1.22, 0.005)
        elif nfev > 20 or alignment < 0.95:
            ds = max(ds * 0.65, 1.5e-5)
        if fs < 0.05 or fs > 1.05 or alpha3 < -0.12 or alpha3 > 0.90:
            break
        if np.max(np.abs(x)) > 2.0:
            break
    return points


def full_fold_row(order: int, z: np.ndarray, event: str = "Asymmetric saddle-node") -> dict:
    x, alpha3, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, update_jacobian, _ = base.update_and_jacobian(x, alpha3, pop)
    multipliers = np.linalg.eigvals(update_jacobian)
    critical_index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, critical_index)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": event,
        "sector": "full",
        "order": order,
        "alpha3": alpha3,
        "fs": fs,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_eigenvalue": float(np.real(multipliers[critical_index] - 1.0)),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
    }


def classification_residual(z: np.ndarray) -> np.ndarray:
    u, alpha3, fs = unpack(z)
    pop = base.make_population(fs)
    fixed = base.symmetric_system(u, alpha3, pop)[0]
    x = base.full_from_symmetric(u, pop)
    b_value = base.observables(x, pop)[0]
    return np.concatenate((fixed, [b_value - B_CUTOFF]))


def full_classification_residual(z: np.ndarray) -> np.ndarray:
    """B-threshold contour on a mirror-broken full fixed-point branch."""
    x, alpha3, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, _, _ = base.update_and_jacobian(x, alpha3, pop)
    b_value = base.observables(x, pop)[0]
    return np.concatenate((update - x, [b_value - B_CUTOFF]))


def full_classification_seed() -> np.ndarray:
    fs = 0.60
    pop = base.make_population(fs)
    candidates = []
    for alpha3_probe in (0.55, 0.56, 0.57, 0.58):
        for item in base.stable_attractors(alpha3_probe, pop, []):
            if float(item["S"]) > 1.0e-3:
                candidates.append(
                    (
                        abs(float(item["B"]) - B_CUTOFF),
                        alpha3_probe,
                        np.asarray(item["solution"]),
                    )
                )
    if not candidates:
        raise RuntimeError("Could not locate an asymmetric B-threshold seed.")
    _, alpha3_seed, x0 = min(candidates, key=lambda item: item[0])
    initial = np.concatenate((x0 / X_SCALE, [alpha3_seed]))

    # Keep the seed solve inside the physical opinion interval.  Without
    # bounds, trust-region trial points can leave [-1, 1] and make the
    # normalized expected-neighbour denominator vanish even though the
    # desired solution is a regular physical fixed point.
    lower = np.concatenate(
        (np.full(initial.size - 1, -1.05 / X_SCALE), [-0.12])
    )
    upper = np.concatenate(
        (np.full(initial.size - 1, +1.05 / X_SCALE), [0.90])
    )

    def fixed_fs_residual(z: np.ndarray) -> np.ndarray:
        return full_classification_residual(
            np.concatenate((z[:-1], [z[-1], fs]))
        )

    answer = least_squares(
        fixed_fs_residual,
        initial,
        bounds=(lower, upper),
        xtol=2e-13,
        ftol=2e-13,
        gtol=2e-13,
        max_nfev=3000,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(fixed_fs_residual(answer.x))))
    if residual > 2e-7:
        raise RuntimeError(
            f"Asymmetric B={B_CUTOFF:.2f} seed failed; residual={residual:.3e}."
        )
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def classification_seed() -> np.ndarray:
    if B_CUTOFF >= 0.565:
        # B=0.57 crosses the symmetric branch near (0.56, 0.35).
        fs, alpha3_seed = 0.35, 0.56
    else:
        # B=0.56 crosses the symmetric branch near (0.34, 0.25).
        fs, alpha3_seed = 0.25, 0.34
    pop = base.make_population(fs)
    u0 = base.solve_symmetric(base.neutral_seed(pop), alpha3_seed, pop)
    initial = np.concatenate((u0 / U_SCALE, [alpha3_seed]))

    # The same physical bounds make this seed solve deterministic across
    # SciPy versions by preventing nonphysical trial states.
    lower = np.concatenate(
        (np.full(initial.size - 1, -1.05 / U_SCALE), [-0.12])
    )
    upper = np.concatenate(
        (np.full(initial.size - 1, +1.05 / U_SCALE), [0.90])
    )

    def fixed_fs_residual(z: np.ndarray) -> np.ndarray:
        return classification_residual(np.concatenate((z[:-1], [z[-1], fs])))

    answer = least_squares(
        fixed_fs_residual,
        initial,
        bounds=(lower, upper),
        xtol=2e-13,
        ftol=2e-13,
        gtol=2e-13,
        max_nfev=2000,
        x_scale="jac",
    )
    residual = float(np.max(np.abs(fixed_fs_residual(answer.x))))
    if residual > 2e-7:
        raise RuntimeError(
            f"B={B_CUTOFF:.2f} seed failed; residual={residual:.3e}."
        )
    return np.concatenate((answer.x[:-1], [answer.x[-1], fs]))


def trace_classification_direction(
    seed: np.ndarray,
    direction: float,
    ds_initial: float = 0.0025,
    max_points: int = 1400,
) -> list[np.ndarray]:
    points = [seed.copy()]
    tangent = generic_tangent(classification_residual, seed, None)
    if direction * tangent[-1] < 0.0:
        tangent *= -1.0
    ds = ds_initial
    for _ in range(max_points):
        predictor = points[-1] + ds * tangent
        candidate, residual, nfev = generic_correct(
            classification_residual, predictor, tangent
        )
        if residual > 2e-7 or np.linalg.norm(candidate - points[-1]) < 1e-8:
            ds *= 0.5
            if ds < 1.5e-5:
                break
            continue
        if not np.all(np.isfinite(candidate)):
            break
        u, alpha3, fs = unpack(candidate)
        points.append(candidate)
        new_tangent = generic_tangent(classification_residual, candidate, tangent)
        alignment = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 7 and alignment > 0.995:
            ds = min(ds * 1.22, 0.005)
        elif nfev > 20 or alignment < 0.95:
            ds = max(ds * 0.65, 1.5e-5)
        if fs < 0.05 or fs > 1.05 or alpha3 < -0.12 or alpha3 > 0.90:
            break
        if np.max(np.abs(u)) > 2.0:
            break
    return points


def classification_row(order: int, z: np.ndarray) -> dict:
    u, alpha3, fs = unpack(z)
    pop = base.make_population(fs)
    x = base.full_from_symmetric(u, pop)
    update, update_jacobian, _ = base.update_and_jacobian(x, alpha3, pop)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": f"B={B_CUTOFF:.2f} classification",
        "sector": "symmetric",
        "order": order,
        "alpha3": alpha3,
        "fs": fs,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_eigenvalue": np.nan,
        "largest_noncritical_multiplier_modulus": float(
            np.max(np.abs(np.linalg.eigvals(update_jacobian)))
        ),
    }


def full_classification_row(order: int, z: np.ndarray) -> dict:
    x, alpha3, fs = full_unpack(z)
    pop = base.make_population(fs)
    update, update_jacobian, _ = base.update_and_jacobian(x, alpha3, pop)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": f"B={B_CUTOFF:.2f} asymmetric classification",
        "sector": "full",
        "order": order,
        "alpha3": alpha3,
        "fs": fs,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_eigenvalue": np.nan,
        "largest_noncritical_multiplier_modulus": float(
            np.max(np.abs(np.linalg.eigvals(update_jacobian)))
        ),
    }


def diagnostic_row(event: str, sector: str, order: int, z: np.ndarray) -> dict:
    u, alpha3, fs = unpack(z)
    pop = base.make_population(fs)
    x = base.full_from_symmetric(u, pop)
    residual = base.symmetric_system(u, alpha3, pop)[0]
    update_jacobian = base.update_and_jacobian(x, alpha3, pop)[1]
    multipliers = np.linalg.eigvals(update_jacobian)
    critical_index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, critical_index)
    b_value, s_value = base.observables(x, pop)
    return {
        "event": event,
        "sector": sector,
        "order": order,
        "alpha3": alpha3,
        "fs": fs,
        "B": b_value,
        "S": s_value,
        "fixed_residual": float(np.max(np.abs(residual))),
        "critical_eigenvalue": base.sector_eigenvalue(u, alpha3, pop, sector),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
    }


def main() -> None:
    rows: list[dict] = []
    for event in ("Saddle-node", "Pitchfork"):
        seed, sector = seed_point(event)
        backward = trace_direction(seed, sector, -1.0)
        forward = trace_direction(seed, sector, +1.0)
        combined = list(reversed(backward[1:])) + forward
        for order, point in enumerate(combined):
            rows.append(diagnostic_row(event, sector, order, point))
        print(
            f"{event}: {len(combined)} points; "
            f"alpha3=[{min(unpack(z)[1] for z in combined):.5f}, "
            f"{max(unpack(z)[1] for z in combined):.5f}], "
            f"fs=[{min(unpack(z)[2] for z in combined):.5f}, "
            f"{max(unpack(z)[2] for z in combined):.5f}]"
        )

    fold_seed = asymmetric_fold_seed()
    backward = trace_generic_direction(full_fold_residual, fold_seed, -1.0)
    forward = trace_generic_direction(full_fold_residual, fold_seed, +1.0)
    fold_curve = list(reversed(backward[1:])) + forward
    for order, point in enumerate(fold_curve):
        rows.append(full_fold_row(order, point))
    fold_alpha = [full_unpack(z)[1] for z in fold_curve]
    fold_fs = [full_unpack(z)[2] for z in fold_curve]
    print(
        f"Asymmetric saddle-node: {len(fold_curve)} points; "
        f"alpha3=[{min(fold_alpha):.5f}, {max(fold_alpha):.5f}], "
        f"fs=[{min(fold_fs):.5f}, {max(fold_fs):.5f}]"
    )

    right_seed = asymmetric_right_fold_seed()
    backward = trace_generic_direction(full_fold_residual, right_seed, -1.0)
    forward = trace_generic_direction(full_fold_residual, right_seed, +1.0)
    right_curve = list(reversed(backward[1:])) + forward
    right_event = "Asymmetric saddle-node (right)"
    for order, point in enumerate(right_curve):
        rows.append(full_fold_row(order, point, event=right_event))
    right_alpha = [full_unpack(z)[1] for z in right_curve]
    right_fs = [full_unpack(z)[2] for z in right_curve]
    print(
        f"{right_event}: {len(right_curve)} points; "
        f"alpha3=[{min(right_alpha):.5f}, {max(right_alpha):.5f}], "
        f"fs=[{min(right_fs):.5f}, {max(right_fs):.5f}]"
    )

    b_seed = classification_seed()
    backward = trace_classification_direction(b_seed, -1.0)
    forward = trace_classification_direction(b_seed, +1.0)
    b_curve = list(reversed(backward[1:])) + forward
    for order, point in enumerate(b_curve):
        rows.append(classification_row(order, point))
    b_alpha = [unpack(z)[1] for z in b_curve]
    b_fs = [unpack(z)[2] for z in b_curve]
    print(
        f"B={B_CUTOFF:.2f} classification: {len(b_curve)} points; "
        f"alpha3=[{min(b_alpha):.5f}, {max(b_alpha):.5f}], "
        f"fs=[{min(b_fs):.5f}, {max(b_fs):.5f}]"
    )

    full_b_seed = full_classification_seed()
    backward = trace_generic_direction(
        full_classification_residual, full_b_seed, -1.0, max_points=1200
    )
    forward = trace_generic_direction(
        full_classification_residual, full_b_seed, +1.0, max_points=1200
    )
    full_b_curve = list(reversed(backward[1:])) + forward
    for order, point in enumerate(full_b_curve):
        rows.append(full_classification_row(order, point))
    full_b_alpha = [full_unpack(z)[1] for z in full_b_curve]
    full_b_fs = [full_unpack(z)[2] for z in full_b_curve]
    print(
        f"B={B_CUTOFF:.2f} asymmetric classification: {len(full_b_curve)} points; "
        f"alpha3=[{min(full_b_alpha):.5f}, {max(full_b_alpha):.5f}], "
        f"fs=[{min(full_b_fs):.5f}, {max(full_b_fs):.5f}]"
    )

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
