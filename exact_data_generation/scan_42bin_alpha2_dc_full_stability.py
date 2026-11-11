"""Exhaustive 42-bin stable-fixed-point scan in (alpha2, dc) at fs=0.5.

The scan uses the comprehensive eta-bin seed bank, bidirectional alpha2
continuation, bidirectional dc propagation, exact sign-reflection completion,
and the full analytic 42x42 stability test.  It is intentionally independent
of the old two-branch alpha2--dc plotter.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model
import scan_42bin_full_stability_map as exhaustive


FS = 0.50
ALPHA_VALUES = np.round(np.arange(0.0, 0.3001, 0.01), 12)
DC_VALUES = np.round(np.arange(0.0, 1.0001, 0.01), 12)
DISCOVERY_STRIDE = 4

CACHE = HERE / "alpha2_dc_42bin_full_stability_step0p01.pkl"
ATTRACTORS = HERE / "alpha2_dc_42bin_full_stability_step0p01_attractors.csv"
SUMMARY = HERE / "alpha2_dc_42bin_full_stability_step0p01_summary.csv"


def scan_dc_row(dc_index: int, dc: float):
    model.DC = float(dc)
    pop = model.make_population(FS)
    n_alpha = len(ALPHA_VALUES)

    ascending: list[list[np.ndarray]] = [[] for _ in range(n_alpha)]
    previous: list[np.ndarray] = []
    for index, alpha2 in enumerate(ALPHA_VALUES):
        if index % DISCOVERY_STRIDE == 0:
            bank = (
                exhaustive.COMPREHENSIVE_SEEDS
                if dc_index % DISCOVERY_STRIDE == 0
                else exhaustive.LIGHT_SEEDS
            )
        else:
            bank = exhaustive.MINIMAL_SEEDS
        found = exhaustive.solve_seeds(previous + bank, float(alpha2), pop)
        ascending[index] = found
        previous = found

    descending: list[list[np.ndarray]] = [[] for _ in range(n_alpha)]
    previous = []
    for index in range(n_alpha - 1, -1, -1):
        alpha2 = float(ALPHA_VALUES[index])
        found = exhaustive.solve_seeds(
            previous + ascending[index] + exhaustive.MINIMAL_SEEDS,
            alpha2,
            pop,
        )
        descending[index] = found
        previous = found

    merged = [
        exhaustive.union_solutions(ascending[index], descending[index])
        for index in range(n_alpha)
    ]
    return dc_index, float(dc), merged


def dc_propagation(solutions):
    additions = 0
    pop = model.make_population(FS)
    for alpha_index, alpha2 in enumerate(ALPHA_VALUES):
        previous = []
        for dc_index, dc in enumerate(DC_VALUES):
            model.DC = float(dc)
            if previous:
                found = exhaustive.continue_seed_bank(previous, float(alpha2), pop)
                before = len(solutions[dc_index][alpha_index])
                solutions[dc_index][alpha_index] = exhaustive.union_solutions(
                    solutions[dc_index][alpha_index], found
                )
                additions += len(solutions[dc_index][alpha_index]) - before
            previous = solutions[dc_index][alpha_index]

        previous = []
        for dc_index in range(len(DC_VALUES) - 1, -1, -1):
            dc = float(DC_VALUES[dc_index])
            model.DC = dc
            if previous:
                found = exhaustive.continue_seed_bank(previous, float(alpha2), pop)
                before = len(solutions[dc_index][alpha_index])
                solutions[dc_index][alpha_index] = exhaustive.union_solutions(
                    solutions[dc_index][alpha_index], found
                )
                additions += len(solutions[dc_index][alpha_index]) - before
            previous = solutions[dc_index][alpha_index]

        if (alpha_index + 1) % 5 == 0 or alpha_index + 1 == len(ALPHA_VALUES):
            print(
                f"dc propagation columns {alpha_index + 1}/{len(ALPHA_VALUES)}; "
                f"additions={additions}",
                flush=True,
            )
    return additions


def alpha_propagation(solutions):
    additions = 0
    pop = model.make_population(FS)
    for dc_index, dc in enumerate(DC_VALUES):
        model.DC = float(dc)
        previous = []
        for alpha_index, alpha2 in enumerate(ALPHA_VALUES):
            if previous:
                found = exhaustive.continue_seed_bank(previous, float(alpha2), pop)
                before = len(solutions[dc_index][alpha_index])
                solutions[dc_index][alpha_index] = exhaustive.union_solutions(
                    solutions[dc_index][alpha_index], found
                )
                additions += len(solutions[dc_index][alpha_index]) - before
            previous = solutions[dc_index][alpha_index]

        previous = []
        for alpha_index in range(len(ALPHA_VALUES) - 1, -1, -1):
            alpha2 = float(ALPHA_VALUES[alpha_index])
            if previous:
                found = exhaustive.continue_seed_bank(previous, alpha2, pop)
                before = len(solutions[dc_index][alpha_index])
                solutions[dc_index][alpha_index] = exhaustive.union_solutions(
                    solutions[dc_index][alpha_index], found
                )
                additions += len(solutions[dc_index][alpha_index]) - before
            previous = solutions[dc_index][alpha_index]

        if (dc_index + 1) % 10 == 0 or dc_index + 1 == len(DC_VALUES):
            print(
                f"alpha propagation rows {dc_index + 1}/{len(DC_VALUES)}; "
                f"additions={additions}",
                flush=True,
            )
    return additions


def validate(solutions):
    removed = 0
    pop = model.make_population(FS)
    for dc_index, dc in enumerate(DC_VALUES):
        model.DC = float(dc)
        for alpha_index, alpha2 in enumerate(ALPHA_VALUES):
            valid = []
            for value in solutions[dc_index][alpha_index]:
                if exhaustive.verify_stable(value, float(alpha2), pop) is not None:
                    exhaustive.append_distinct(valid, value)
                else:
                    removed += 1
            for value in list(valid):
                exhaustive.append_distinct(valid, exhaustive.sign_reflection(value))
            solutions[dc_index][alpha_index] = valid
    return removed


def observable_class_count(values, pop):
    classes = []
    for value in values:
        b_value, s_value = model.observables(value, pop)
        pair = (float(b_value), abs(float(s_value)))
        if not any(abs(pair[0] - old[0]) < 2e-3 and abs(pair[1] - old[1]) < 2e-3 for old in classes):
            classes.append(pair)
    return len(classes)


def save(solutions):
    with CACHE.open("wb") as stream:
        pickle.dump(
            {
                "alpha_values": ALPHA_VALUES,
                "dc_values": DC_VALUES,
                "fs": FS,
                "solutions": solutions,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    attractor_fields = [
        "alpha2", "dc", "fs", "attractor", "B", "S", "abs_S",
        "spectral_radius", "residual", "mirror_error",
    ]
    summary_fields = [
        "alpha2", "dc", "fs", "detected_stable_count",
        "observable_class_count", "mirror_missing_count", "max_spectral_radius",
    ]
    pop = model.make_population(FS)
    with ATTRACTORS.open("w", newline="", encoding="utf-8") as af, SUMMARY.open(
        "w", newline="", encoding="utf-8"
    ) as sf:
        aw = csv.DictWriter(af, fieldnames=attractor_fields)
        sw = csv.DictWriter(sf, fieldnames=summary_fields)
        aw.writeheader()
        sw.writeheader()
        for dc_index, dc in enumerate(DC_VALUES):
            model.DC = float(dc)
            for alpha_index, alpha2 in enumerate(ALPHA_VALUES):
                values = solutions[dc_index][alpha_index]
                rhos = []
                for attractor, value in enumerate(values, start=1):
                    residual, rho = exhaustive.verify_stable(value, float(alpha2), pop)
                    b_value, s_value = model.observables(value, pop)
                    mirror_error = float(
                        np.max(np.abs(exhaustive.sign_reflection(value) - value))
                    )
                    aw.writerow(
                        {
                            "alpha2": float(alpha2),
                            "dc": float(dc),
                            "fs": FS,
                            "attractor": attractor,
                            "B": b_value,
                            "S": s_value,
                            "abs_S": abs(s_value),
                            "spectral_radius": rho,
                            "residual": residual,
                            "mirror_error": mirror_error,
                        }
                    )
                    rhos.append(rho)
                sw.writerow(
                    {
                        "alpha2": float(alpha2),
                        "dc": float(dc),
                        "fs": FS,
                        "detected_stable_count": len(values),
                        "observable_class_count": observable_class_count(values, pop),
                        "mirror_missing_count": exhaustive.symmetry_missing_count(values),
                        "max_spectral_radius": max(rhos) if rhos else np.nan,
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    solutions = [[[] for _ in ALPHA_VALUES] for _ in DC_VALUES]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(scan_dc_row, index, float(dc)): index
            for index, dc in enumerate(DC_VALUES)
        }
        completed = 0
        for future in as_completed(futures):
            index, dc, row = future.result()
            solutions[index] = row
            completed += 1
            print(
                f"initial dc rows {completed}/{len(DC_VALUES)}; dc={dc:.2f}",
                flush=True,
            )

    additions_dc = dc_propagation(solutions)
    additions_alpha = alpha_propagation(solutions)
    removed = validate(solutions)
    save(solutions)

    counts = Counter(
        len(solutions[i][j])
        for i in range(len(DC_VALUES))
        for j in range(len(ALPHA_VALUES))
    )
    print(f"dc propagation additions={additions_dc}")
    print(f"alpha propagation additions={additions_alpha}")
    print(f"removed by final validation={removed}")
    print(f"stable-count distribution={dict(sorted(counts.items()))}")
    print(f"Saved {CACHE}")
    print(f"Saved {ATTRACTORS}")
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
