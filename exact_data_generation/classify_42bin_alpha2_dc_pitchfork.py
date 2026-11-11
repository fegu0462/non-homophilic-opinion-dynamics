"""Classify the continuously followed 42-bin pitchfork along (alpha2, dc)."""

from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compute_42d_bifurcation_boundaries as model
import compute_plot_42bin_alpha2_dc as old
import scan_42bin_full_stability_map as scan


FS = 0.50
CACHE = HERE / "alpha2_dc_42bin_full_stability_step0p01.pkl"
OUTPUT = HERE / "alpha2_dc_42bin_pitchfork_criticality_fine.csv"
TRICRITICAL = HERE / "alpha2_dc_42bin_tricritical_points.csv"
DC_VALUES = np.round(np.arange(0.0, 1.00001, 0.0005), 12)


def initial_pitchfork():
    saved = pickle.load(CACHE.open("rb"))
    dc_index = int(np.argmin(np.abs(saved["dc_values"] - 0.0)))
    alpha_index = int(np.argmin(np.abs(saved["alpha_values"] - 0.03)))
    pop = model.make_population(FS)
    candidates = []
    for value in saved["solutions"][dc_index][alpha_index]:
        if np.max(np.abs(scan.sign_reflection(value) - value)) < 2e-5:
            b_value, _ = model.observables(value, pop)
            candidates.append((b_value, np.asarray(value, dtype=float)))
    _, neutral = min(candidates, key=lambda item: item[0])
    model.DC = 0.0
    return model.refine_critical_point(
        neutral[pop.select_symmetric], 0.022, pop, sector="odd"
    )


def main():
    pop = model.make_population(FS)
    alpha2, u = initial_pitchfork()
    rows = []
    for index, dc in enumerate(DC_VALUES):
        model.DC = float(dc)
        if index:
            alpha2, u = model.refine_critical_point(u, alpha2, pop, sector="odd")
        cubic = old.pitchfork_cubic_coefficient(float(dc), alpha2, u, pop)
        rows.append(
            {
                "dc": float(dc),
                "alpha2": float(alpha2),
                "normal_form_cubic": cubic,
                "pitchfork_type": "Subcritical" if cubic > 0.0 else "Supercritical",
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    crossings = []
    for left, right in zip(rows[:-1], rows[1:]):
        c0 = left["normal_form_cubic"]
        c1 = right["normal_form_cubic"]
        if c0 == 0.0 or c0 * c1 < 0.0:
            fraction = -c0 / (c1 - c0)
            crossings.append(
                {
                    "dc": left["dc"] + fraction * (right["dc"] - left["dc"]),
                    "alpha2": left["alpha2"]
                    + fraction * (right["alpha2"] - left["alpha2"]),
                    "normal_form_cubic": 0.0,
                    "type": "Tricritical point",
                }
            )
    with TRICRITICAL.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(crossings[0]))
        writer.writeheader()
        writer.writerows(crossings)
    print(f"saved {OUTPUT}")
    print(f"saved {TRICRITICAL}")
    for point in crossings:
        print(
            f"tricritical: alpha2={point['alpha2']:.9f}, dc={point['dc']:.9f}"
        )


if __name__ == "__main__":
    main()
