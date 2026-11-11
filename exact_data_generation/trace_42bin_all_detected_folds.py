"""Trace the cascade of eta-resolved saddle-node curves in the 42-bin model.

The full stable scan finds 45 stable fixed points at (alpha2, fs)=(0, .9):
one sign-symmetric state and 22 sign-reflected pairs.  Each positive-S member
is continued in alpha2 to its loss of stability, refined with the augmented
fixed-point/fold equations, and then followed in (alpha2, fs) by
pseudo-arclength continuation.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
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
import trace_42d_alpha2_fs_pseudoarclength as arc


CHECKPOINT = HERE / "mf_42bin_full_stability_step0p005.pkl"
OUTPUT_CSV = HERE / "bifurcation_boundaries_42bin_eta_resolved_folds.csv"
SEED_FS = 0.9
ALPHA_STEP = 0.001


def fixed_root(x0: np.ndarray, alpha2: float, pop: model.Population):
    answer = root(
        lambda x: model.update_jacobian_alpha(x, alpha2, pop)[0] - x,
        np.asarray(x0, dtype=float),
        jac=lambda x: model.update_jacobian_alpha(x, alpha2, pop)[1]
        - np.eye(pop.dimension),
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 700},
    )
    residual = float(
        np.max(np.abs(model.update_jacobian_alpha(answer.x, alpha2, pop)[0] - answer.x))
    )
    return answer.x, residual


def multiplier_radius(x: np.ndarray, alpha2: float, pop: model.Population) -> float:
    jacobian = model.update_jacobian_alpha(x, alpha2, pop)[1]
    return float(np.max(np.abs(np.linalg.eigvals(jacobian))))


def track_to_fold_seed(x0: np.ndarray, pop: model.Population):
    """Follow one stable branch in alpha2 until its critical multiplier nears 1."""
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_alpha = 0.0
    best_rho = multiplier_radius(x, 0.0, pop)
    previous_x = x.copy()

    for alpha2 in np.arange(ALPHA_STEP, 0.301, ALPHA_STEP):
        candidate, residual = fixed_root(x, float(alpha2), pop)
        if residual > 2e-7:
            break
        jump = float(np.sqrt(np.mean((candidate - previous_x) ** 2)))
        if jump > 0.10:
            break
        rho = multiplier_radius(candidate, float(alpha2), pop)
        # Stay on the stable sheet; a root-solver jump to another stable branch
        # is rejected by the state-space jump test above.
        if rho >= 1.0 + 2e-4:
            break
        x = candidate
        previous_x = candidate
        if rho > best_rho:
            best_x = candidate.copy()
            best_alpha = float(alpha2)
            best_rho = rho
        if rho > 0.9995:
            break
    return best_x, best_alpha, best_rho


def refine_fixed_fs_fold(
    x0: np.ndarray, alpha0: float, fs: float
) -> tuple[np.ndarray, float, float]:
    def residual(z: np.ndarray) -> np.ndarray:
        packed = np.concatenate((z[:-1], [z[-1], fs]))
        return arc.full_fold_residual(packed)

    initial = np.concatenate((np.asarray(x0) / arc.X_SCALE, [alpha0 + 0.001]))
    answer = least_squares(
        residual,
        initial,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=1800,
        x_scale="jac",
    )
    error = float(np.max(np.abs(residual(answer.x))))
    x = arc.X_SCALE * answer.x[:-1]
    return x, float(answer.x[-1]), error


def seed_folds(limit: int | None = None):
    with CHECKPOINT.open("rb") as stream:
        saved = pickle.load(stream)
    alpha_values = saved["alpha_values"]
    fs_values = saved["fs_values"]
    solutions = saved["solutions"]
    fs_index = int(np.argmin(np.abs(fs_values - SEED_FS)))
    alpha_index = int(np.argmin(np.abs(alpha_values - 0.0)))
    pop = model.make_population(SEED_FS)
    positive = [
        value
        for value in solutions[fs_index][alpha_index]
        if model.observables(value, pop)[1] > 1e-6
    ]
    positive.sort(key=lambda x: model.observables(x, pop)[1])
    if limit is not None:
        positive = positive[:limit]

    seeds = []
    for index, x0 in enumerate(positive, start=1):
        near_x, near_alpha, near_rho = track_to_fold_seed(x0, pop)
        fold_x, fold_alpha, error = refine_fixed_fs_fold(
            near_x, near_alpha, SEED_FS
        )
        b_value, s_value = model.observables(fold_x, pop)
        rho = multiplier_radius(fold_x, fold_alpha, pop)
        if error > 2e-7 or abs(rho - 1.0) > 2e-5:
            raise RuntimeError(
                f"Fold seed {index} failed: alpha={fold_alpha}, "
                f"error={error:.3e}, rho={rho:.9f}."
            )
        packed = arc.full_pack(fold_x, fold_alpha, SEED_FS)
        seeds.append(packed)
        print(
            f"seed {index:02d}/{len(positive):02d}: "
            f"alpha2={fold_alpha:.9f}, S={s_value:.6f}, B={b_value:.6f}, "
            f"pre-rho={near_rho:.6f}, residual={error:.2e}",
            flush=True,
        )
    return seeds


def trace_seed(seed: np.ndarray):
    # The seed lies on the upper boundary of the requested fs range, so only
    # the initially downward arclength direction is required.
    return arc.trace_generic_direction(
        arc.full_fold_residual,
        seed,
        direction=-1.0,
        ds_initial=0.004,
        max_points=750,
    )


def diagnostics(curve_id: int, order: int, packed: np.ndarray):
    x, alpha2, fs = arc.full_unpack(packed)
    pop = model.make_population(fs)
    update, jacobian, _ = model.update_jacobian_alpha(x, alpha2, pop)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    critical = int(np.argmin(np.abs(eigenvalues - 1.0)))
    remaining = np.delete(eigenvalues, critical)
    b_value, s_value = model.observables(x, pop)
    symmetry_error = float(np.max(np.abs(x - scan.sign_reflection(x))))
    return {
        "curve_id": curve_id,
        "curve_order": order,
        "event": "Saddle-node" if symmetry_error > 1e-5 else "Symmetric critical curve",
        "alpha2": alpha2,
        "fs": fs,
        "B": b_value,
        "S": s_value,
        "symmetry_error": symmetry_error,
        "fixed_residual": float(np.max(np.abs(update - x))),
        "critical_multiplier_real": float(np.real(eigenvalues[critical])),
        "critical_multiplier_imag": float(np.imag(eigenvalues[critical])),
        "largest_noncritical_multiplier_modulus": float(np.max(np.abs(remaining))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    seeds = seed_folds(args.limit)
    rows = []
    for curve_id, seed in enumerate(seeds, start=1):
        curve = trace_seed(seed)
        curve_rows = [
            diagnostics(curve_id, order, point)
            for order, point in enumerate(curve)
        ]
        rows.extend(curve_rows)
        alpha = np.asarray([row["alpha2"] for row in curve_rows])
        fs = np.asarray([row["fs"] for row in curve_rows])
        print(
            f"curve {curve_id:02d}: points={len(curve_rows)}, "
            f"alpha2=[{alpha.min():.6f},{alpha.max():.6f}], "
            f"fs=[{fs.min():.6f},{fs.max():.6f}]",
            flush=True,
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
