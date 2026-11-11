"""Recompute and plot the two four-bin mean-field bifurcation slices.

Panel (a) varies alpha2 at fs=0.5.  Panel (b) varies fs at alpha2=0.
No precomputed branch CSV is read: both fixed-point data sets and the critical
points are calculated when the notebook calls the functions in this module.
The rendering delegates to ``src.mean_field_bifurcation_slices`` so that the
fonts, sizes, branch styles, state fills, legends, and panel geometry remain
identical to the manuscript figure.
"""

from __future__ import annotations

import csv
import os
import runpy
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from .. import mean_field_bifurcation_slices as style
from . import compute_4bin_fs_slice as fs_model


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "recomputed" / "four_bin_two_slices"
FIGURE_DIR = ROOT / "figures" / "four_bin_two_slices_full_reproduction"
ALPHA_SOLVER = Path(__file__).with_name(
    "compute_4bin_alpha2_full_continuation.py"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_FIXED_FS = 0.50
ALPHA2_MIN = 0.0
ALPHA2_MAX = 0.70
ALPHA2_STEP = 0.0005

ALPHA_PNG = FIGURE_DIR / "panel_a_alpha2_bifurcation.png"
ALPHA_PDF = FIGURE_DIR / "panel_a_alpha2_bifurcation.pdf"
FS_PNG = FIGURE_DIR / "panel_b_fs_bifurcation.png"
FS_PDF = FIGURE_DIR / "panel_b_fs_bifurcation.pdf"
COMBINED_PNG = FIGURE_DIR / "four_bin_bifurcation_slices.png"
COMBINED_PDF = FIGURE_DIR / "four_bin_bifurcation_slices.pdf"
COMBINED_SVG = FIGURE_DIR / "four_bin_bifurcation_slices.svg"
COMBINED_TIFF = FIGURE_DIR / "four_bin_bifurcation_slices.tiff"

ALPHA_CSV = DATA_DIR / "four_bin_alpha2_branches_fs050.csv"
ALPHA_EVENTS_CSV = DATA_DIR / "four_bin_alpha2_bifurcations_fs050.csv"
FS_CSV = DATA_DIR / "four_bin_fs_branches_alpha2_0.csv"
FS_EVENTS_CSV = DATA_DIR / "four_bin_fs_bifurcations_alpha2_0.csv"


def _alpha_tracks(namespace: dict[str, Any]) -> list[list[dict[str, Any]]]:
    branches: list[list[dict[str, Any]]] = []
    alpha_min = float(namespace["alpha2_values"][0])
    alpha_step = float(namespace["alpha2_values"][1] - alpha_min)
    for stable, key in (
        (True, "stable_tracks"),
        (False, "unstable_tracks"),
    ):
        for track in namespace[key]:
            points = []
            for point in track:
                root = np.asarray(point["root"], dtype=float)
                alpha2 = float(point["alpha2"])
                points.append(
                    {
                        "parameter": alpha2,
                        "mu_plus": float(root[2]),
                        "mu_minus": float(root[3]),
                        "stable": stable,
                        "grid_index": int(round((alpha2 - alpha_min) / alpha_step)),
                        "root": root,
                        "spectral_radius": np.nan,
                    }
                )
            points.sort(key=lambda row: row["grid_index"])
            if len(points) >= 2:
                branches.append(points)
    return branches


def _alpha_events(
    namespace: dict[str, Any],
    branches: list[list[dict[str, Any]]],
) -> tuple[float, float]:
    events = [
        (str(kind), float(parameter))
        for kind, parameter, _root in namespace["bifurcation_points"]
        if 0.0 <= float(parameter) <= 0.30
    ]
    saddle = [value for kind, value in events if "saddle" in kind or "fold" in kind]
    pitchfork = [value for kind, value in events if "pitchfork" in kind]
    if saddle and pitchfork:
        return min(saddle), max(pitchfork)

    # Parameter-dependent fallback used only if the normal-form label is
    # inconclusive at a nonsmooth point.  It still comes from the newly
    # calculated branches rather than a stored manuscript value.
    detected = style.detected_events(branches, use_stability_switch=True)
    detected = [value for value in detected if 0.0 <= value <= 0.30]
    if len(detected) < 2:
        raise RuntimeError("Could not locate both alpha2 bifurcations.")
    return float(detected[0]), float(detected[1])


def compute_alpha_panel() -> tuple[
    dict[str, Any], list[list[dict[str, Any]]], float, float
]:
    """Run the supplied bidirectional continuation code for panel (a)."""
    os.environ["MF4_FS"] = str(ALPHA_FIXED_FS)
    os.environ["MF4_K2"] = "4.0"
    os.environ["MF4_DC"] = "0.5"
    os.environ["MF4_ALPHA3"] = "0.0"
    os.environ["MF4_ALPHA2_MIN"] = str(ALPHA2_MIN)
    os.environ["MF4_ALPHA2_MAX"] = str(ALPHA2_MAX)
    os.environ["MF4_ALPHA2_STEP"] = str(ALPHA2_STEP)
    os.environ["MF4_OUTPUT_DIR"] = str(FIGURE_DIR)
    os.environ["MF4_OUTPUT_PREFIX"] = "panel_a_solver_diagnostic"
    os.environ["MF4_SHOW_FIGURE"] = "0"

    namespace = runpy.run_path(str(ALPHA_SOLVER), run_name="mf4_alpha_solver")
    branches = _alpha_tracks(namespace)
    saddle, pitchfork = _alpha_events(namespace, branches)
    _write_branch_csv(ALPHA_CSV, branches, "alpha2")
    _write_event_csv(
        ALPHA_EVENTS_CSV,
        (("saddle-node", saddle), ("supercritical-pitchfork", pitchfork)),
    )
    return namespace, branches, saddle, pitchfork


def _fs_branches(
    raw_branches: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    branches: list[list[dict[str, Any]]] = []
    for branch in raw_branches:
        points = []
        for point in branch["points"]:
            root = np.asarray(point["root"], dtype=float)
            points.append(
                {
                    "parameter": float(point["fs"]),
                    "mu_plus": float(root[2]),
                    "mu_minus": float(root[3]),
                    "stable": bool(point["is_stable"]),
                    "grid_index": int(point["grid_index"]),
                    "root": root,
                    "spectral_radius": float(point["spectral_radius"]),
                }
            )
        points.sort(key=lambda row: row["grid_index"])
        if len(points) >= 2:
            branches.append(points)
    return branches


def _fs_critical_equations(z: np.ndarray) -> np.ndarray:
    mu = np.asarray(z[:4], dtype=float)
    fs = float(z[4])
    residual = fs_model.fixed_point_residual(mu, fs)
    jacobian_residual = fs_model.numerical_jacobian_map(mu, fs) - np.eye(4)
    return np.concatenate((residual, [np.linalg.det(jacobian_residual)]))


def _refine_one_fs_event(
    coarse: float,
    records: list[dict[str, Any]],
) -> float:
    candidates: list[tuple[float, np.ndarray]] = []
    for entry in records:
        fs = float(entry["fs"])
        if abs(fs - coarse) > 0.03:
            continue
        for record in entry["roots"]:
            candidates.append((fs, np.asarray(record["root"], dtype=float)))

    solutions: list[tuple[float, np.ndarray, float]] = []
    for fs0, root in candidates:
        for parameter_guess in (fs0, coarse):
            solution, _info, ier, _message = fsolve(
                _fs_critical_equations,
                np.concatenate((root, [parameter_guess])),
                full_output=True,
                xtol=1.0e-11,
                maxfev=5000,
            )
            error = float(np.max(np.abs(_fs_critical_equations(solution))))
            fs_critical = float(solution[4])
            if ier != 1 or error > 1.0e-7 or not (0.15 <= fs_critical <= 0.55):
                continue
            root_critical = np.asarray(solution[:4], dtype=float)
            duplicate = any(
                abs(fs_critical - old_fs) < 1.0e-6
                and np.linalg.norm(root_critical - old_root) < 1.0e-4
                for old_fs, old_root, _old_error in solutions
            )
            if not duplicate:
                solutions.append((fs_critical, root_critical, error))

    if not solutions:
        raise RuntimeError(f"Could not refine the f_s event near {coarse:.6f}.")
    return min(solutions, key=lambda row: abs(row[0] - coarse))[0]


def compute_fs_panel() -> tuple[
    list[dict[str, Any]], list[list[dict[str, Any]]], float, float
]:
    """Recompute the alpha2=0 branches and refined critical points."""
    records = fs_model.scan_fixed_points()
    raw_branches = fs_model.track_branches(records)
    branches = _fs_branches(raw_branches)
    coarse = [
        value
        for value in style.detected_events(branches)
        if 0.15 <= value <= 0.55
    ]
    if len(coarse) < 2:
        raise RuntimeError("Could not locate both coarse f_s bifurcations.")
    pitchfork = _refine_one_fs_event(float(coarse[0]), records)
    saddle = _refine_one_fs_event(float(coarse[1]), records)
    _write_branch_csv(FS_CSV, branches, "fs")
    _write_event_csv(
        FS_EVENTS_CSV,
        (("supercritical-pitchfork", pitchfork), ("saddle-node", saddle)),
    )
    return records, branches, saddle, pitchfork


def _write_branch_csv(
    path: Path,
    branches: list[list[dict[str, Any]]],
    parameter_name: str,
) -> None:
    fields = [
        "branch_id",
        parameter_name,
        "is_stable",
        "spectral_radius",
        "mu_s_plus",
        "mu_s_minus",
        "mu_ns_plus",
        "mu_ns_minus",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for branch_id, branch in enumerate(branches):
            for point in branch:
                root = np.asarray(point["root"], dtype=float)
                writer.writerow(
                    {
                        "branch_id": branch_id,
                        parameter_name: f"{float(point['parameter']):.12g}",
                        "is_stable": int(bool(point["stable"])),
                        "spectral_radius": (
                            ""
                            if not np.isfinite(point["spectral_radius"])
                            else f"{float(point['spectral_radius']):.12g}"
                        ),
                        "mu_s_plus": f"{root[0]:.12g}",
                        "mu_s_minus": f"{root[1]:.12g}",
                        "mu_ns_plus": f"{root[2]:.12g}",
                        "mu_ns_minus": f"{root[3]:.12g}",
                    }
                )


def _write_event_csv(
    path: Path,
    events: tuple[tuple[str, float], ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("event_type", "parameter"))
        writer.writeheader()
        for event_type, parameter in events:
            writer.writerow(
                {"event_type": event_type, "parameter": f"{parameter:.12g}"}
            )


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, Path]:
    paths = {
        "png": FIGURE_DIR / f"{stem}.png",
        "pdf": FIGURE_DIR / f"{stem}.pdf",
        "svg": FIGURE_DIR / f"{stem}.svg",
        "tiff": FIGURE_DIR / f"{stem}.tiff",
    }
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["pdf"], dpi=600)
    fig.savefig(paths["svg"])
    fig.savefig(
        paths["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return paths


def render_alpha_panel(
    branches: list[list[dict[str, Any]]],
    saddle: float,
    pitchfork: float,
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(4.95, 4.65), dpi=300)
    style.draw_alpha_panel(ax, branches, saddle, pitchfork, "(a)")
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.76)
    paths = _save_figure(fig, "panel_a_alpha2_bifurcation")
    plt.close(fig)
    return paths


def render_fs_panel(
    branches: list[list[dict[str, Any]]],
    saddle: float,
    pitchfork: float,
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(4.95, 4.65), dpi=300)
    style.draw_fs_panel(ax, branches, saddle, pitchfork, "(b)")
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.76)
    paths = _save_figure(fig, "panel_b_fs_bifurcation")
    plt.close(fig)
    return paths


def render_combined(
    alpha_branches: list[list[dict[str, Any]]],
    alpha_saddle: float,
    alpha_pitchfork: float,
    fs_branches: list[list[dict[str, Any]]],
    fs_saddle: float,
    fs_pitchfork: float,
) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 1, figsize=(4.95, 8.90), dpi=300)
    style.draw_alpha_panel(
        axes[0], alpha_branches, alpha_saddle, alpha_pitchfork, "(a)"
    )
    style.draw_fs_panel(axes[1], fs_branches, fs_saddle, fs_pitchfork, "(b)")
    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.075,
        top=0.90,
        hspace=0.46,
    )
    paths = _save_figure(fig, "four_bin_bifurcation_slices")
    plt.close(fig)
    return paths


def main() -> dict[str, Any]:
    alpha_namespace, alpha_branches, alpha_saddle, alpha_pitchfork = (
        compute_alpha_panel()
    )
    fs_records, fs_branches, fs_saddle, fs_pitchfork = compute_fs_panel()
    alpha_paths = render_alpha_panel(alpha_branches, alpha_saddle, alpha_pitchfork)
    fs_paths = render_fs_panel(fs_branches, fs_saddle, fs_pitchfork)
    combined_paths = render_combined(
        alpha_branches,
        alpha_saddle,
        alpha_pitchfork,
        fs_branches,
        fs_saddle,
        fs_pitchfork,
    )
    return {
        "alpha_namespace": alpha_namespace,
        "alpha_branches": alpha_branches,
        "alpha_saddle": alpha_saddle,
        "alpha_pitchfork": alpha_pitchfork,
        "fs_records": fs_records,
        "fs_branches": fs_branches,
        "fs_saddle": fs_saddle,
        "fs_pitchfork": fs_pitchfork,
        "alpha_paths": alpha_paths,
        "fs_paths": fs_paths,
        "combined_paths": combined_paths,
    }


if __name__ == "__main__":
    result = main()
    for group in ("alpha_paths", "fs_paths", "combined_paths"):
        for path in result[group].values():
            print(f"Saved {path}")
