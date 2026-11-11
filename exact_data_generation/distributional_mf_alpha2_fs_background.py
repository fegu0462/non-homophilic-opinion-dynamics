"""Distributional-MF B/|S| background with precomputed 4-group MF boundaries.

The non-stubborn intrinsic-bias laws U(0,1) and U(-1,0) are discretized into
equal-weight midpoint bins.  For each stubborn fraction, the stable state is
continued from alpha2=0 to alpha2=0.30.  Bifurcation curves are read from the
existing four-group-MF CSV; they are not recomputed here.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


# -----------------------------
# Parameter grid and model
# -----------------------------
ALPHA2_VALUES = np.round(np.arange(0.0, 0.30 + 1e-12, 0.01), 12)
FS_VALUES = np.round(np.arange(0.15, 0.55 + 1e-12, 0.01), 12)

GAMMA = 4.0
DC = 0.5
ALPHA3 = 0.0
SIGMA_S = 0.8
SIGMA_NS = 0.1

N_ETA_BINS_PER_SIDE = 100
PERTURBATION = 1e-4
MAX_ITER = 100_000
CONVERGENCE_TOL = 1e-8


# -----------------------------
# Existing 4-group MF data
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BIFURCATION_CSV = SCRIPT_DIR / "bifurcation_events_alpha2_fs_existing.csv"
SUMMARY_CSV = SCRIPT_DIR / "distributional_mf_alpha2_fs_step0p01_summary.csv"
OUTPUT_PNG = SCRIPT_DIR / (
    "distributional_mf_background_with_4group_boundaries_step0p01.png"
)
OUTPUT_PDF = SCRIPT_DIR / (
    "distributional_mf_background_with_4group_boundaries_step0p01.pdf"
)


# -----------------------------
# Same B/|S| bivariate colors as the previous MC-background figure
# -----------------------------
B_COLOR_MIN = 1.0 / 3.0
B_COLOR_MAX = 1.0
ABS_S_COLOR_MIN = 0.0
ABS_S_COLOR_MAX = 0.6

COLOR_LOW_LOW = np.asarray(to_rgb("#f7f7f7"))
COLOR_HIGH_B = np.asarray(to_rgb("#3478bf"))
COLOR_HIGH_S = np.asarray(to_rgb("#e88937"))
COLOR_HIGH_BS = np.asarray(to_rgb("#7651a4"))

BACKGROUND_ALPHA = 0.94
FONT_SIZE = 15
BOUNDARY_LINE_WIDTH = 3.0
BOUNDARY_DASHES = (8, 5)
BOUNDARY_COLORS = {
    "Saddle-node": "#00E000",
    "Supercritical pitchfork": "#FFD700",
    "Subcritical pitchfork": "#FF3B30",
}


def make_population(fs):
    """Build the two stubborn types and 2*n non-stubborn eta bins."""
    n = N_ETA_BINS_PER_SIDE
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    eta = np.concatenate(([1.0, -1.0], eta_positive, eta_negative))
    sigma = np.concatenate(
        ([SIGMA_S, SIGMA_S], np.full(2 * n, SIGMA_NS))
    )
    weights = np.concatenate(
        (
            [fs / 2.0, fs / 2.0],
            np.full(2 * n, (1.0 - fs) / (2.0 * n)),
        )
    )
    return eta, sigma, weights


def standard_initial_condition():
    """Symmetric benchmark with a tiny positive perturbation to select a side."""
    n = N_ETA_BINS_PER_SIDE
    return np.concatenate(
        (
            [1.0, -1.0],
            np.full(n, 0.5 + PERTURBATION),
            np.full(n, -0.5 + PERTURBATION),
        )
    )


def expected_update(x, alpha2, eta, sigma, weights):
    """One expected-network update of every eta-resolved type."""
    alpha1 = 1.0 - alpha2 - ALPHA3
    distance = np.abs(x[:, None] - x[None, :])
    exponent = np.clip(GAMMA * (distance - DC), -100.0, 100.0)

    c1 = (1.0 + np.exp(-GAMMA * DC)) / (1.0 + np.exp(exponent))
    c2 = np.abs(x)[None, :]
    c3 = np.abs(x)[:, None]
    connection = alpha1 * c1 + alpha2 * c2 + ALPHA3 * c3

    denominator = connection @ weights
    if np.any(denominator <= 1e-14):
        raise RuntimeError("Expected-neighbour denominator reached zero.")
    neighbour_mean = (connection @ (weights * x)) / denominator
    return sigma * eta + (1.0 - sigma) * neighbour_mean


def iterate_to_stationarity(x0, alpha2, eta, sigma, weights):
    x = np.asarray(x0, dtype=float).copy()
    for iteration in range(1, MAX_ITER + 1):
        x_next = expected_update(x, alpha2, eta, sigma, weights)
        error = float(np.max(np.abs(x_next - x)))
        x = x_next
        if error < CONVERGENCE_TOL:
            return x, iteration, error
    raise RuntimeError(
        f"No convergence for fs={weights[0] * 2:.3f}, alpha2={alpha2:.3f}"
    )


def distribution_observables(x, weights):
    """Population-weighted continuum B and S for non-stubborn bins."""
    values = x[2:]
    ns_weights = weights[2:] / np.sum(weights[2:])
    mean_s = float(np.sum(ns_weights * values))
    centered = values - mean_s
    variance = float(np.sum(ns_weights * centered**2))

    if variance <= 1e-16:
        return np.nan, mean_s

    skewness = float(np.sum(ns_weights * centered**3) / variance**1.5)
    raw_kurtosis = float(np.sum(ns_weights * centered**4) / variance**2)
    b_value = float((skewness**2 + 1.0) / raw_kurtosis)
    return b_value, mean_s


def run_distributional_mf_grid():
    """Follow one stable branch while alpha2 increases for every fixed fs."""
    b_map = np.full((FS_VALUES.size, ALPHA2_VALUES.size), np.nan)
    abs_s_map = np.full_like(b_map, np.nan)
    rows = []

    for fs_index, fs in enumerate(FS_VALUES):
        eta, sigma, weights = make_population(float(fs))
        previous_solution = standard_initial_condition()

        for alpha_index, alpha2 in enumerate(ALPHA2_VALUES):
            # Reinsert an infinitesimal positive perturbation before every
            # continuation step.  The negative mirror has identical B and |S|.
            seed = previous_solution.copy()
            seed[2:] = np.clip(seed[2:] + PERTURBATION, -1.0, 1.0)
            solution, iterations, residual = iterate_to_stationarity(
                seed, float(alpha2), eta, sigma, weights
            )
            b_value, s_value = distribution_observables(solution, weights)

            b_map[fs_index, alpha_index] = b_value
            abs_s_map[fs_index, alpha_index] = abs(s_value)
            rows.append(
                {
                    "alpha2": float(alpha2),
                    "fs": float(fs),
                    "B": b_value,
                    "S": s_value,
                    "abs_S": abs(s_value),
                    "iterations": iterations,
                    "residual": residual,
                }
            )
            previous_solution = solution

        print(f"Completed fs={fs:.2f}")

    return b_map, abs_s_map, rows


def bivariate_rgba(b_values, abs_s_values):
    """Bilinear blend: blue=high B, orange=high |S|, purple=both high."""
    b = np.clip(
        (b_values - B_COLOR_MIN) / (B_COLOR_MAX - B_COLOR_MIN), 0.0, 1.0
    )
    s = np.clip(
        (abs_s_values - ABS_S_COLOR_MIN)
        / (ABS_S_COLOR_MAX - ABS_S_COLOR_MIN),
        0.0,
        1.0,
    )
    rgb = (
        (1.0 - b)[..., None] * (1.0 - s)[..., None] * COLOR_LOW_LOW
        + b[..., None] * (1.0 - s)[..., None] * COLOR_HIGH_B
        + (1.0 - b)[..., None] * s[..., None] * COLOR_HIGH_S
        + b[..., None] * s[..., None] * COLOR_HIGH_BS
    )
    return np.concatenate(
        (rgb, np.full(rgb.shape[:-1] + (1,), BACKGROUND_ALPHA)), axis=-1
    )


def load_bifurcation_events():
    """Load only verified boundary points inside the plotted parameter box."""
    if not BIFURCATION_CSV.exists():
        raise FileNotFoundError(f"Missing bifurcation CSV: {BIFURCATION_CSV}")

    events = []
    with BIFURCATION_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("stable_branch_verified", "1") not in {"1", "true", "True"}:
                continue
            event_type = row["type"]
            if event_type not in BOUNDARY_COLORS:
                continue
            alpha2 = float(row["alpha2"])
            fs = float(row["fs"])
            if (
                ALPHA2_VALUES[0] - 1e-12 <= alpha2 <= ALPHA2_VALUES[-1] + 1e-12
                and FS_VALUES[0] - 1e-12 <= fs <= FS_VALUES[-1] + 1e-12
            ):
                events.append(
                    {
                        "alpha2": alpha2,
                        "fs": fs,
                        "type": event_type,
                        "curve_id": int(row["curve_id"]),
                        "curve_order": int(row["curve_order"]),
                    }
                )
    return events


def collapse_boundary_points(events, event_type):
    """Merge mirrored/duplicate continuations into one visible boundary."""
    points = sorted(
        {
            (round(event["fs"], 8), round(event["alpha2"], 8))
            for event in events
            if event["type"] == event_type
        }
    )
    if not points:
        return np.empty(0), np.empty(0)

    groups = [[points[0]]]
    for point in points[1:]:
        previous = groups[-1][-1]
        if abs(point[0] - previous[0]) < 4e-4 and abs(point[1] - previous[1]) < 4e-4:
            groups[-1].append(point)
        else:
            groups.append([point])

    fs = np.asarray([np.mean([point[0] for point in group]) for group in groups])
    alpha2 = np.asarray(
        [np.mean([point[1] for point in group]) for group in groups]
    )
    order = np.argsort(fs)
    return alpha2[order], fs[order]


def plot_boundaries(ax, events):
    present_types = []
    for event_type in BOUNDARY_COLORS:
        alpha2, fs = collapse_boundary_points(events, event_type)
        if alpha2.size == 0:
            continue
        present_types.append(event_type)
        ax.plot(
            alpha2,
            fs,
            color=BOUNDARY_COLORS[event_type],
            linewidth=BOUNDARY_LINE_WIDTH,
            linestyle="--",
            dashes=BOUNDARY_DASHES,
            dash_capstyle="round",
            path_effects=[
                path_effects.Stroke(
                    linewidth=BOUNDARY_LINE_WIDTH + 2.0,
                    foreground="black",
                    alpha=0.50,
                ),
                path_effects.Normal(),
            ],
            zorder=5,
        )
    return present_types


def add_color_key(ax):
    key_ax = ax.inset_axes([0.68, 0.105, 0.285, 0.27])
    b_axis = np.linspace(B_COLOR_MIN, B_COLOR_MAX, 121)
    s_axis = np.linspace(ABS_S_COLOR_MIN, ABS_S_COLOR_MAX, 121)
    b_grid, s_grid = np.meshgrid(b_axis, s_axis)
    key_ax.imshow(
        bivariate_rgba(b_grid, s_grid),
        origin="lower",
        aspect="auto",
        extent=(B_COLOR_MIN, B_COLOR_MAX, ABS_S_COLOR_MIN, ABS_S_COLOR_MAX),
        interpolation="nearest",
    )
    key_ax.set_xlabel(r"Distributional-MF $B$", fontsize=11)
    key_ax.set_ylabel(r"Distributional-MF $|S|$", fontsize=11)
    key_ax.tick_params(labelsize=10)
    key_ax.set_title("State-color key", fontsize=11, fontweight="bold")


def save_summary(rows):
    fields = ["alpha2", "fs", "B", "S", "abs_S", "iterations", "residual"]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_result(b_map, abs_s_map):
    rgba = bivariate_rgba(b_map, abs_s_map)
    alpha_step = float(ALPHA2_VALUES[1] - ALPHA2_VALUES[0])
    fs_step = float(FS_VALUES[1] - FS_VALUES[0])

    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=180)
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            ALPHA2_VALUES[0] - alpha_step / 2.0,
            ALPHA2_VALUES[-1] + alpha_step / 2.0,
            FS_VALUES[0] - fs_step / 2.0,
            FS_VALUES[-1] + fs_step / 2.0,
        ),
        zorder=0,
    )

    events = load_bifurcation_events()
    present_types = plot_boundaries(ax, events)
    handles = [
        Line2D(
            [0],
            [0],
            color=BOUNDARY_COLORS[event_type],
            linewidth=BOUNDARY_LINE_WIDTH,
            linestyle="--",
            dashes=BOUNDARY_DASHES,
            label=event_type,
        )
        for event_type in present_types
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.95)
    add_color_key(ax)

    ax.set_xlim(ALPHA2_VALUES[0], ALPHA2_VALUES[-1])
    ax.set_ylim(FS_VALUES[0], FS_VALUES[-1])
    ax.set_xticks(np.arange(0.0, 0.30 + 1e-12, 0.05))
    ax.set_yticks(np.arange(0.15, 0.55 + 1e-12, 0.05))
    ax.set_xticks(ALPHA2_VALUES, minor=True)
    ax.set_yticks(FS_VALUES, minor=True)
    ax.set_xlabel(r"$\alpha_2$", fontsize=FONT_SIZE + 2)
    ax.set_ylabel(r"Stubborn fraction $f_s$", fontsize=FONT_SIZE + 2)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(which="major", ls=":", color="black", alpha=0.18, linewidth=0.9)
    ax.grid(which="minor", ls=":", color="black", alpha=0.07, linewidth=0.5)
    ax.set_title(
        "Distributional-MF state observables with 4-group MF bifurcation boundaries",
        fontsize=FONT_SIZE + 2,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    if not BIFURCATION_CSV.exists():
        raise FileNotFoundError(f"Missing required CSV: {BIFURCATION_CSV}")
    b_map, abs_s_map, rows = run_distributional_mf_grid()
    save_summary(rows)
    plot_result(b_map, abs_s_map)
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
