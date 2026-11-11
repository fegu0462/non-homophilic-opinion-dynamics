"""Classify the continued 42-bin alpha3--fs pitchfork by its cubic coefficient."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import compute_plot_42bin_alpha3_fs as model
import trace_42bin_alpha3_fs_pseudoarclength as trace


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "alpha3_fs_42bin_pitchfork_criticality.csv"


def _normalize_real(vector: np.ndarray) -> np.ndarray:
    vector = np.real_if_close(vector, tol=1000).real
    vector = vector / np.linalg.norm(vector)
    index = int(np.argmax(np.abs(vector)))
    return vector if vector[index] >= 0.0 else -vector


def pitchfork_cubic_coefficient(
    alpha3: float,
    u: np.ndarray,
    population: model.Population,
) -> float:
    """Lyapunov--Schmidt cubic coefficient for F=G-x.

    Positive values are subcritical and negative values are supercritical,
    using the same residual convention as the alpha2--dc calculation.
    """
    x = model.full_from_symmetric(u, population)
    dimension = x.size

    def residual(state: np.ndarray, parameter: float) -> np.ndarray:
        return model.update_and_jacobian(state, parameter, population)[0] - state

    def jacobian(state: np.ndarray, parameter: float) -> np.ndarray:
        return model.update_and_jacobian(state, parameter, population)[1] - np.eye(
            dimension
        )

    j_fixed = jacobian(x, alpha3)
    left_vectors, _, right_vectors = np.linalg.svd(j_fixed)
    q = _normalize_real(right_vectors[-1])
    p = _normalize_real(left_vectors[:, -1])
    p = p / np.dot(p, q)

    def directional_b(
        first: np.ndarray,
        direction: np.ndarray,
        state_step: float,
    ) -> np.ndarray:
        h = state_step / np.max(np.abs(direction))
        return (
            (jacobian(x + h * direction, alpha3) - jacobian(x - h * direction, alpha3))
            @ first
        ) / (2.0 * h)

    def directional_c(state_step: float) -> np.ndarray:
        h = state_step / np.max(np.abs(q))
        return (
            (
                jacobian(x + h * q, alpha3)
                - 2.0 * jacobian(x, alpha3)
                + jacobian(x - h * q, alpha3)
            )
            @ q
        ) / h**2

    def coefficient_at_steps(state_step: float, parameter_step: float) -> float:
        # Retain the parameter derivative calculation as a conditioning check,
        # although the cubic coefficient itself is determined by state terms.
        f_parameter = (
            residual(x, alpha3 + parameter_step)
            - residual(x, alpha3 - parameter_step)
        ) / (2.0 * parameter_step)
        if not np.all(np.isfinite(f_parameter)):
            raise RuntimeError("Non-finite pitchfork parameter derivative.")
        b_qq = directional_b(q, q, state_step)
        bordered = np.block(
            [[j_fixed, q[:, None]], [p[None, :], np.zeros((1, 1))]]
        )
        w_20 = np.linalg.solve(
            bordered, np.concatenate((-b_qq, [0.0]))
        )[:-1]
        return float(
            np.dot(p, directional_c(state_step)) / 6.0
            + 0.5 * np.dot(p, directional_b(q, w_20, state_step))
        )

    coarse = coefficient_at_steps(5.0e-4, 2.0e-4)
    fine = coefficient_at_steps(2.5e-4, 1.0e-4)
    return float((4.0 * fine - coarse) / 3.0)


def main() -> None:
    seed, sector = trace.seed_point("Pitchfork")
    backward = trace.trace_direction(seed, sector, -1.0)
    forward = trace.trace_direction(seed, sector, +1.0)
    curve = list(reversed(backward[1:])) + forward

    rows = []
    for order, point in enumerate(curve):
        u, alpha3, fs = trace.unpack(point)
        if not (0.0 <= alpha3 <= 0.70 and 0.20 <= fs <= 0.90):
            continue
        diagnostic = trace.diagnostic_row("Pitchfork", sector, order, point)
        cubic = pitchfork_cubic_coefficient(alpha3, u, model.make_population(fs))
        rows.append(
            {
                "order": order,
                "alpha3": alpha3,
                "fs": fs,
                "largest_noncritical_multiplier_modulus": diagnostic[
                    "largest_noncritical_multiplier_modulus"
                ],
                "normal_form_cubic": cubic,
                "pitchfork_type": "Subcritical" if cubic > 0.0 else "Supercritical",
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stable = [
        row
        for row in rows
        if float(row["largest_noncritical_multiplier_modulus"]) < 1.0 - 2.0e-6
    ]
    print(f"Saved {OUTPUT}")
    print(
        "all visible pitchfork types:",
        sorted({str(row["pitchfork_type"]) for row in rows}),
    )
    print(
        "stable-involving pitchfork types:",
        sorted({str(row["pitchfork_type"]) for row in stable}),
    )
    for collection_name, collection in (("all", rows), ("stable", stable)):
        for kind in ("Subcritical", "Supercritical"):
            selected = [row for row in collection if row["pitchfork_type"] == kind]
            if selected:
                print(
                    collection_name,
                    kind,
                    "alpha3=",
                    (min(float(row["alpha3"]) for row in selected), max(float(row["alpha3"]) for row in selected)),
                    "fs=",
                    (min(float(row["fs"]) for row in selected), max(float(row["fs"]) for row in selected)),
                )


if __name__ == "__main__":
    main()
