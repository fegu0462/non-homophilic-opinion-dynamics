"""True pseudo-arclength continuation of 42-bin alpha2--dc critical curves.

Every stable branch present at (alpha2, dc)=(0, 0) is first followed in
alpha2 to its critical multiplier +1.  The resulting 42-D critical solutions
are refined and then continued in the full (x, alpha2, dc) space.  Unlike the
cross-section detector, this file never joins independently sampled dc rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares, root

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model
import scan_42bin_full_stability_map as scan


FS = 0.50
SEED_DC = 0.0
ALPHA_STEP = 5.0e-4
X_SCALE = np.sqrt(model.STATE_DIMENSION)
U_SCALE = np.sqrt(model.BINS_PER_SIDE + 1)
CACHE = HERE / "alpha2_dc_42bin_full_stability_step0p01.pkl"
OUTPUT_CSV = HERE / "alpha2_dc_42bin_true_pseudoarclength_curves.csv"
OUTPUT_SEEDS = HERE / "alpha2_dc_42bin_true_pseudoarclength_seeds.csv"


def pack(x: np.ndarray, alpha2: float, dc: float) -> np.ndarray:
    return np.concatenate((np.asarray(x, dtype=float) / X_SCALE, [alpha2, dc]))


def unpack(z: np.ndarray):
    return X_SCALE * z[:-2], float(z[-2]), float(z[-1])


def fixed_root(x0: np.ndarray, alpha2: float, dc: float, pop):
    model.DC = float(dc)
    answer = root(
        lambda x: model.update_jacobian_alpha(x, alpha2, pop)[0] - x,
        np.asarray(x0, dtype=float),
        jac=lambda x: model.update_jacobian_alpha(x, alpha2, pop)[1]
        - np.eye(pop.dimension),
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 800},
    )
    residual = float(
        np.max(np.abs(model.update_jacobian_alpha(answer.x, alpha2, pop)[0] - answer.x))
    )
    return answer.x, residual


def multiplier_radius(x: np.ndarray, alpha2: float, dc: float, pop) -> float:
    model.DC = float(dc)
    jacobian = model.update_jacobian_alpha(x, alpha2, pop)[1]
    return float(np.max(np.abs(np.linalg.eigvals(jacobian))))


def critical_residual(z: np.ndarray) -> np.ndarray:
    x, alpha2, dc = unpack(z)
    model.DC = float(dc)
    pop = model.make_population(FS)
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    eig = np.linalg.eigvals(jacobian - np.eye(pop.dimension))
    critical = eig[int(np.argmin(np.abs(eig)))]
    return np.concatenate((update - x, [float(np.real(critical))]))


def symmetric_pack(u: np.ndarray, alpha2: float, dc: float) -> np.ndarray:
    return np.concatenate((np.asarray(u, dtype=float) / U_SCALE, [alpha2, dc]))


def symmetric_unpack(z: np.ndarray):
    return U_SCALE * z[:-2], float(z[-2]), float(z[-1])


def pitchfork_residual(z: np.ndarray) -> np.ndarray:
    u, alpha2, dc = symmetric_unpack(z)
    model.DC = float(dc)
    pop = model.make_population(FS)
    fixed = model.symmetric_system(u, alpha2, pop)[0]
    odd_critical = model.sector_eigenvalue(u, alpha2, pop, "odd")
    return np.concatenate((fixed, [odd_critical]))


def fixed_dc_residual(z: np.ndarray, dc: float) -> np.ndarray:
    return critical_residual(np.concatenate((z[:-1], [z[-1], dc])))


def track_to_critical(x0: np.ndarray, pop):
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_alpha = 0.0
    best_rho = multiplier_radius(x, 0.0, SEED_DC, pop)
    previous = x.copy()
    for alpha2 in np.arange(ALPHA_STEP, 0.1201, ALPHA_STEP):
        candidate, residual = fixed_root(x, float(alpha2), SEED_DC, pop)
        if residual > 2e-7:
            break
        jump = float(np.sqrt(np.mean((candidate - previous) ** 2)))
        if jump > 0.075:
            break
        rho = multiplier_radius(candidate, float(alpha2), SEED_DC, pop)
        if rho > 1.0003:
            break
        x = candidate
        previous = candidate
        if rho > best_rho:
            best_x = candidate.copy()
            best_alpha = float(alpha2)
            best_rho = rho
        if rho > 0.99965:
            break
    return best_x, best_alpha, best_rho


def refine_seed(x0: np.ndarray, alpha0: float):
    initial = np.concatenate((np.asarray(x0) / X_SCALE, [alpha0 + 0.0005]))
    answer = least_squares(
        lambda z: fixed_dc_residual(z, SEED_DC),
        initial,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=1800,
        x_scale="jac",
    )
    error = float(np.max(np.abs(fixed_dc_residual(answer.x, SEED_DC))))
    return pack(X_SCALE * answer.x[:-1], float(answer.x[-1]), SEED_DC), error


def make_seeds():
    saved = pickle.load(CACHE.open("rb"))
    dc_index = int(np.argmin(np.abs(saved["dc_values"] - SEED_DC)))
    alpha_index = int(np.argmin(np.abs(saved["alpha_values"] - 0.0)))
    roots = [np.asarray(x, dtype=float) for x in saved["solutions"][dc_index][alpha_index]]
    pop = model.make_population(FS)
    model.DC = SEED_DC

    representatives = []
    for value in roots:
        _, s_value = model.observables(value, pop)
        if s_value > 1e-7 or np.max(np.abs(scan.sign_reflection(value) - value)) < 2e-5:
            representatives.append(value)
    representatives.sort(key=lambda value: model.observables(value, pop)[1])

    seeds = []
    rows = []
    for source_index, value in enumerate(representatives, start=1):
        near_x, near_alpha, near_rho = track_to_critical(value, pop)
        seed, error = refine_seed(near_x, near_alpha)
        x, alpha2, dc = unpack(seed)
        b_value, s_value = model.observables(x, pop)
        if error > 2e-7 or not (-0.005 <= alpha2 <= 0.305):
            print(
                f"reject seed {source_index}: alpha={alpha2:.7f}, error={error:.2e}",
                flush=True,
            )
            continue
        duplicate = False
        for old in seeds:
            old_x, old_alpha, _ = unpack(old)
            state_distance = min(
                float(np.sqrt(np.mean((x - old_x) ** 2))),
                float(np.sqrt(np.mean((scan.sign_reflection(x) - old_x) ** 2))),
            )
            if abs(alpha2 - old_alpha) < 5e-5 and state_distance < 5e-4:
                duplicate = True
                break
        if duplicate:
            continue
        seeds.append(seed)
        rows.append(
            {
                "seed_id": len(seeds),
                "alpha2": alpha2,
                "dc": dc,
                "B": b_value,
                "S": s_value,
                "precritical_rho": near_rho,
                "residual": error,
            }
        )
        print(
            f"seed {len(seeds):02d}: alpha2={alpha2:.9f}, "
            f"S={s_value:.6f}, B={b_value:.6f}, residual={error:.2e}",
            flush=True,
        )

    # The stable neutral branch is not present at alpha2=0, so its pitchfork
    # cannot be found by tracking the alpha2=0 stable roots.  Seed it from the
    # first stable neutral solution at alpha2=0.03 and refine the same critical
    # equations at dc=0.
    neutral_index = int(np.argmin(np.abs(saved["alpha_values"] - 0.03)))
    symmetric_candidates = []
    for value in saved["solutions"][dc_index][neutral_index]:
        symmetry_error = float(np.max(np.abs(scan.sign_reflection(value) - value)))
        if symmetry_error < 2e-5:
            b_value, _ = model.observables(value, pop)
            symmetric_candidates.append((b_value, np.asarray(value, dtype=float)))
    if symmetric_candidates:
        _, neutral = min(symmetric_candidates, key=lambda item: item[0])
        initial = np.concatenate((neutral / X_SCALE, [0.022]))
        answer = least_squares(
            lambda z: fixed_dc_residual(z, SEED_DC),
            initial,
            xtol=2e-12,
            ftol=2e-12,
            gtol=2e-12,
            max_nfev=1800,
            x_scale="jac",
        )
        error = float(np.max(np.abs(fixed_dc_residual(answer.x, SEED_DC))))
        pitch_seed = pack(X_SCALE * answer.x[:-1], float(answer.x[-1]), SEED_DC)
        x, alpha2, dc = unpack(pitch_seed)
        b_value, s_value = model.observables(x, pop)
        if error <= 2e-7:
            seeds.append(pitch_seed)
            rows.append(
                {
                    "seed_id": len(seeds),
                    "alpha2": alpha2,
                    "dc": dc,
                    "B": b_value,
                    "S": s_value,
                    "precritical_rho": np.nan,
                    "residual": error,
                }
            )
            print(
                f"seed {len(seeds):02d} (pitchfork): alpha2={alpha2:.9f}, "
                f"S={s_value:.3e}, B={b_value:.6f}, residual={error:.2e}",
                flush=True,
            )
    with OUTPUT_SEEDS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return seeds


def numerical_jacobian(function, z: np.ndarray) -> np.ndarray:
    f0 = function(z)
    jac = np.empty((f0.size, z.size))
    for column in range(z.size):
        step = 1.5e-6 * max(1.0, abs(float(z[column])))
        plus = z.copy()
        minus = z.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return jac


def tangent_at(function, z: np.ndarray, previous: np.ndarray | None):
    _, _, vh = np.linalg.svd(numerical_jacobian(function, z), full_matrices=True)
    tangent = vh[-1]
    tangent /= np.linalg.norm(tangent)
    if previous is not None and np.dot(tangent, previous) < 0.0:
        tangent *= -1.0
    return tangent


def correct(function, predictor: np.ndarray, tangent: np.ndarray):
    def augmented(z):
        return np.concatenate((function(z), [np.dot(z - predictor, tangent)]))

    answer = least_squares(
        augmented,
        predictor,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=900,
        x_scale="jac",
    )
    error = float(np.max(np.abs(augmented(answer.x))))
    return answer.x, error, int(answer.nfev)


def trace_seed(seed_id: int, seed: np.ndarray):
    points = [seed.copy()]
    tangent = tangent_at(critical_residual, seed, None)
    if tangent[-1] < 0.0:
        tangent *= -1.0
    ds = 0.003
    for _ in range(650):
        predictor = points[-1] + ds * tangent
        candidate, error, nfev = correct(critical_residual, predictor, tangent)
        if error > 2e-7 or np.linalg.norm(candidate - points[-1]) < 1e-8:
            ds *= 0.5
            if ds < 2e-5:
                break
            continue
        x, alpha2, dc = unpack(candidate)
        if not np.all(np.isfinite(candidate)) or np.max(np.abs(x)) > 2.0:
            break
        points.append(candidate)
        new_tangent = tangent_at(critical_residual, candidate, tangent)
        angle = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 6 and angle > 0.996:
            ds = min(ds * 1.25, 0.006)
        elif nfev > 18 or angle < 0.96:
            ds = max(ds * 0.65, 2e-5)
        if dc < -0.025 or dc > 1.025 or alpha2 < -0.04 or alpha2 > 0.36:
            break
    return seed_id, points


def trace_pitchfork_seed(seed_id: int, seed: np.ndarray):
    x, alpha2, dc = unpack(seed)
    pop = model.make_population(FS)
    u = x[pop.select_symmetric]
    symmetric_points = [symmetric_pack(u, alpha2, dc)]
    tangent = tangent_at(pitchfork_residual, symmetric_points[0], None)
    if tangent[-1] < 0.0:
        tangent *= -1.0
    ds = 0.003
    for _ in range(650):
        predictor = symmetric_points[-1] + ds * tangent
        candidate, error, nfev = correct(pitchfork_residual, predictor, tangent)
        if error > 2e-7 or np.linalg.norm(candidate - symmetric_points[-1]) < 1e-8:
            ds *= 0.5
            if ds < 2e-5:
                break
            continue
        u, alpha2, dc = symmetric_unpack(candidate)
        if not np.all(np.isfinite(candidate)) or np.max(np.abs(u)) > 2.0:
            break
        symmetric_points.append(candidate)
        new_tangent = tangent_at(pitchfork_residual, candidate, tangent)
        angle = float(np.clip(np.dot(new_tangent, tangent), -1.0, 1.0))
        tangent = new_tangent
        if nfev <= 6 and angle > 0.996:
            ds = min(ds * 1.25, 0.006)
        elif nfev > 18 or angle < 0.96:
            ds = max(ds * 0.65, 2e-5)
        if dc < -0.025 or dc > 1.025 or alpha2 < -0.04 or alpha2 > 0.36:
            break
    full_points = []
    for point in symmetric_points:
        u, alpha2, dc = symmetric_unpack(point)
        full_points.append(pack(model.full_from_symmetric(u, pop), alpha2, dc))
    return seed_id, full_points


def diagnostics(curve_id: int, order: int, z: np.ndarray):
    x, alpha2, dc = unpack(z)
    model.DC = float(dc)
    pop = model.make_population(FS)
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    multipliers = np.linalg.eigvals(jacobian)
    critical_index = int(np.argmin(np.abs(multipliers - 1.0)))
    noncritical = np.delete(multipliers, critical_index)
    symmetry_error = float(np.max(np.abs(scan.sign_reflection(x) - x)))
    if symmetry_error < 2e-5:
        even = np.min(np.abs(np.linalg.eigvals(pop.q_even.T @ jacobian @ pop.q_even) - 1.0))
        odd = np.min(np.abs(np.linalg.eigvals(pop.q_odd.T @ jacobian @ pop.q_odd) - 1.0))
        event = "Pitchfork" if odd < even else "Symmetric saddle-node"
    else:
        event = "Asymmetric saddle-node"
    b_value, s_value = model.observables(x, pop)
    return {
        "curve_id": curve_id,
        "curve_order": order,
        "alpha2": alpha2,
        "dc": dc,
        "event": event,
        "B": b_value,
        "S": s_value,
        "symmetry_error": symmetry_error,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_multiplier_real": float(np.real(multipliers[critical_index])),
        "critical_multiplier_imag": float(np.imag(multipliers[critical_index])),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(noncritical))),
        "stable_involving": bool(np.max(np.abs(noncritical)) <= 1.002),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = make_seeds()
    traced = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, seed in enumerate(seeds, start=1):
            function = trace_pitchfork_seed if index == len(seeds) else trace_seed
            futures[executor.submit(function, index, seed)] = index
        for future in as_completed(futures):
            curve_id, points = future.result()
            traced[curve_id] = points
            a = np.asarray([unpack(point)[1] for point in points])
            d = np.asarray([unpack(point)[2] for point in points])
            print(
                f"curve {curve_id:02d}: points={len(points)}, "
                f"alpha=[{a.min():.5f},{a.max():.5f}], dc=[{d.min():.5f},{d.max():.5f}]",
                flush=True,
            )

    rows = []
    for curve_id in sorted(traced):
        rows.extend(
            diagnostics(curve_id, order, point)
            for order, point in enumerate(traced[curve_id])
        )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
