"""Four-bin mean-field bifurcation diagram versus the stubborn fraction.

The extremity-attraction and receiver-extremity terms are switched off
(alpha2 = alpha3 = 0).  The stubborn fraction f_s is scanned from 0 to 0.8
in increments of 0.005.  Every fixed point found at each parameter value is
classified using the spectral radius of the Jacobian of the discrete-time
mean-field map and then matched into full four-dimensional branches.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import fsolve, linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "recomputed"
FIGURE_DIR = ROOT / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = FIGURE_DIR / "recomputed_four_bin_mf_fs_bifurcation.png"
OUTPUT_PDF = FIGURE_DIR / "recomputed_four_bin_mf_fs_bifurcation.pdf"
OUTPUT_CSV = DATA_DIR / "four_bin_mf_fs_bifurcation_alpha2_0.csv"

ALPHA2 = 0.0
ALPHA3 = 0.0
ALPHA1 = 1.0
GAMMA = 4.0
DC = 0.5
SIGMA_S = 0.8
SIGMA_NS = 0.1
ETA = np.array([1.0, -1.0, 0.5, -0.5], dtype=float)
FS_VALUES = np.round(np.arange(0.0, 0.8001, 0.005), 10)

ROOT_RESIDUAL_TOL = 2.0e-9
ROOT_DUPLICATE_TOL = 2.0e-5
STABILITY_TOL = 2.0e-6
BRANCH_MATCH_TOL = 0.16
MAX_MISSED_STEPS = 1

STABLE_PLUS = "#3F6FD8"
STABLE_MINUS = "#2F9B45"
UNSTABLE_PLUS = "#EF8FA4"
UNSTABLE_MINUS = "#F4A64F"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.major.size": 3.8,
        "ytick.major.size": 3.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def population_weights(fs: float) -> np.ndarray:
    return np.array(
        [0.5 * fs, 0.5 * fs, 0.5 * (1.0 - fs), 0.5 * (1.0 - fs)],
        dtype=float,
    )


def stubbornness() -> np.ndarray:
    return np.array([SIGMA_S, SIGMA_S, SIGMA_NS, SIGMA_NS], dtype=float)


def mean_field_map(mu: np.ndarray, fs: float) -> np.ndarray:
    """Return the four-bin discrete-time mean-field map G(mu; f_s)."""
    mu = np.asarray(mu, dtype=float)
    distance = np.abs(mu[:, None] - mu[None, :])
    c1 = (1.0 + np.exp(-GAMMA * DC)) / (
        1.0 + np.exp(np.clip(GAMMA * (distance - DC), -100.0, 100.0))
    )
    c2 = np.abs(mu)[None, :]
    c3 = np.abs(mu)[:, None]
    kernel = ALPHA1 * c1 + ALPHA2 * c2 + ALPHA3 * c3

    weights = population_weights(fs)
    denominator = kernel @ weights
    neighbour_mean = (kernel * weights[None, :]) @ mu
    neighbour_mean = np.divide(
        neighbour_mean,
        denominator,
        out=np.zeros_like(neighbour_mean),
        where=denominator > 1.0e-14,
    )
    sigma = stubbornness()
    return sigma * ETA + (1.0 - sigma) * neighbour_mean


def fixed_point_residual(mu: np.ndarray, fs: float) -> np.ndarray:
    return mean_field_map(mu, fs) - np.asarray(mu, dtype=float)


def numerical_jacobian_map(mu: np.ndarray, fs: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    jacobian = np.empty((4, 4), dtype=float)
    for column in range(4):
        step = 1.0e-6 * max(1.0, abs(mu[column]))
        plus = mu.copy()
        minus = mu.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            mean_field_map(plus, fs) - mean_field_map(minus, fs)
        ) / (2.0 * step)
    return jacobian


def build_static_guesses() -> list[np.ndarray]:
    bases = [
        [0.9, -0.9, 0.5, -0.5],
        [0.9, -0.9, 0.0, 0.0],
        [0.9, -0.9, 0.8, -0.8],
        [0.9, -0.9, 0.3, -0.3],
        [0.9, -0.9, 0.8, 0.8],
        [0.9, -0.9, -0.8, -0.8],
        [0.9, -0.9, -0.5, 0.5],
        [1.0, -1.0, 1.0, -1.0],
        [0.5, -0.5, 0.5, -0.5],
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.4, 0.7, 0.5],
        [-0.4, -0.8, -0.5, -0.7],
    ]
    guesses = [np.asarray(value, dtype=float) for value in bases]
    rng = np.random.default_rng(20260922)
    for base in guesses.copy():
        for _ in range(5):
            guesses.append(np.clip(base + rng.normal(0.0, 0.16, 4), -1.0, 1.0))
    guesses.extend(rng.uniform(-0.95, 0.95, size=(80, 4)))
    return [np.asarray(value, dtype=float) for value in guesses]


STATIC_GUESSES = build_static_guesses()


def deduplicate_guesses(guesses: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for guess in guesses:
        candidate = np.asarray(guess, dtype=float)
        if not np.all(np.isfinite(candidate)):
            continue
        if not any(np.max(np.abs(candidate - old)) < 1.0e-9 for old in unique):
            unique.append(candidate)
    return unique


def solve_roots(
    fs: float,
    previous: list[np.ndarray],
    previous_previous: list[np.ndarray],
) -> list[np.ndarray]:
    continuation = list(previous)
    if previous and previous_previous:
        for root in previous:
            nearest = min(previous_previous, key=lambda old: np.linalg.norm(root - old))
            if np.linalg.norm(root - nearest) < 0.25:
                continuation.append(2.0 * root - nearest)

    roots: list[np.ndarray] = []
    for guess in deduplicate_guesses(continuation + STATIC_GUESSES):
        try:
            root, _, ier, _ = fsolve(
                lambda value: fixed_point_residual(value, fs),
                guess,
                full_output=True,
                xtol=1.0e-11,
                maxfev=1600,
            )
        except (FloatingPointError, ValueError):
            continue
        residual = float(np.max(np.abs(fixed_point_residual(root, fs))))
        if ier != 1 or residual > ROOT_RESIDUAL_TOL:
            continue
        if np.max(np.abs(root)) > 1.0001:
            continue
        if any(np.max(np.abs(root - old)) < ROOT_DUPLICATE_TOL for old in roots):
            continue
        roots.append(np.asarray(root, dtype=float))
    roots.sort(key=lambda value: tuple(np.round(value, 10)))
    return roots


def scan_fixed_points() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    previous_previous: list[np.ndarray] = []
    previous: list[np.ndarray] = []
    for fs in FS_VALUES:
        roots = solve_roots(float(fs), previous, previous_previous)
        root_records = []
        for root in roots:
            eigenvalues = np.linalg.eigvals(numerical_jacobian_map(root, float(fs)))
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            root_records.append(
                {
                    "root": root,
                    "spectral_radius": spectral_radius,
                    "is_stable": spectral_radius < 1.0 - STABILITY_TOL,
                }
            )
        records.append({"fs": float(fs), "roots": root_records})
        previous_previous = [value.copy() for value in previous]
        previous = [value.copy() for value in roots]
        print(
            f"f_s={fs:0.3f}: roots={len(root_records)}, "
            f"stable={sum(bool(item['is_stable']) for item in root_records)}"
        )
    return records


def predict_branch(branch: dict[str, object], fs: float) -> np.ndarray:
    points = branch["points"]
    last = points[-1]
    prediction = np.asarray(last["root"], dtype=float)
    if len(points) < 2:
        return prediction
    previous = points[-2]
    step = float(last["fs"] - previous["fs"])
    if abs(step) < 1.0e-14:
        return prediction
    return prediction + (fs - float(last["fs"])) / step * (
        prediction - np.asarray(previous["root"], dtype=float)
    )


def track_branches(records: list[dict[str, object]]) -> list[dict[str, object]]:
    branches: list[dict[str, object]] = []
    next_id = 0
    for grid_index, entry in enumerate(records):
        fs = float(entry["fs"])
        root_records = list(entry["roots"])
        roots = [np.asarray(record["root"], dtype=float) for record in root_records]
        active = [
            branch
            for branch in branches
            if grid_index - int(branch["points"][-1]["grid_index"])
            <= MAX_MISSED_STEPS + 1
        ]
        matched: set[int] = set()
        if active and roots:
            predictions = [predict_branch(branch, fs) for branch in active]
            cost = np.asarray(
                [
                    [np.linalg.norm(prediction - root) for root in roots]
                    for prediction in predictions
                ]
            )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                branch = active[int(row)]
                gap = grid_index - int(branch["points"][-1]["grid_index"])
                if cost[row, column] > BRANCH_MATCH_TOL * np.sqrt(max(1, gap)):
                    continue
                record = root_records[int(column)]
                branch["points"].append(
                    {
                        "fs": fs,
                        "root": roots[int(column)],
                        "is_stable": bool(record["is_stable"]),
                        "spectral_radius": float(record["spectral_radius"]),
                        "grid_index": grid_index,
                    }
                )
                matched.add(int(column))

        for root_index, (record, root) in enumerate(zip(root_records, roots)):
            if root_index in matched:
                continue
            branches.append(
                {
                    "branch_id": next_id,
                    "points": [
                        {
                            "fs": fs,
                            "root": root,
                            "is_stable": bool(record["is_stable"]),
                            "spectral_radius": float(record["spectral_radius"]),
                            "grid_index": grid_index,
                        }
                    ],
                }
            )
            next_id += 1
    return branches


def split_segments(points: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not points:
        return []
    segments = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        gap = int(point["grid_index"]) - int(previous["grid_index"])
        if gap > MAX_MISSED_STEPS + 1 or point["is_stable"] != previous["is_stable"]:
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def write_csv(branches: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "branch_id", "fs", "is_stable", "spectral_radius",
            "mu_s_plus", "mu_s_minus", "mu_ns_plus", "mu_ns_minus",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for branch in branches:
            for point in branch["points"]:
                root = np.asarray(point["root"], dtype=float)
                writer.writerow(
                    {
                        "branch_id": int(branch["branch_id"]),
                        "fs": f"{float(point['fs']):.6f}",
                        "is_stable": int(bool(point["is_stable"])),
                        "spectral_radius": f"{float(point['spectral_radius']):.12g}",
                        "mu_s_plus": f"{root[0]:.12g}",
                        "mu_s_minus": f"{root[1]:.12g}",
                        "mu_ns_plus": f"{root[2]:.12g}",
                        "mu_ns_minus": f"{root[3]:.12g}",
                    }
                )


def plot_branches(branches: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(5.55, 4.05), dpi=220)
    for branch in branches:
        for segment in split_segments(branch["points"]):
            if len(segment) < 2:
                continue
            fs = np.asarray([point["fs"] for point in segment], dtype=float)
            roots = np.asarray([point["root"] for point in segment], dtype=float)
            stable = bool(segment[0]["is_stable"])
            ax.plot(
                fs,
                roots[:, 2],
                color=STABLE_PLUS if stable else UNSTABLE_PLUS,
                ls="-" if stable else "--",
                lw=1.75 if stable else 1.35,
                alpha=0.98 if stable else 0.82,
                marker="o" if stable else "x",
                ms=2.3,
                markevery=max(1, len(segment) // 22),
                markeredgewidth=0.7,
                zorder=4 if stable else 3,
            )
            ax.plot(
                fs,
                roots[:, 3],
                color=STABLE_MINUS if stable else UNSTABLE_MINUS,
                ls="-" if stable else "--",
                lw=1.75 if stable else 1.35,
                alpha=0.98 if stable else 0.82,
                marker="o" if stable else "x",
                ms=2.3,
                markevery=max(1, len(segment) // 22),
                markeredgewidth=0.7,
                zorder=4 if stable else 3,
            )

    ax.axhline(0.0, color="0.45", ls=(0, (3, 2)), lw=0.85, zorder=1)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xticks(np.arange(0.0, 0.81, 0.2))
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel(r"$f_s$", fontsize=12.5, labelpad=3)
    ax.set_ylabel(r"$\mu_{\mathrm{NS}+}^{*},\ \mu_{\mathrm{NS}-}^{*}$", fontsize=12.5, labelpad=4)
    ax.tick_params(axis="both", labelsize=9.5, pad=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#252525")
        ax.spines[side].set_linewidth(0.85)
    ax.grid(True, ls=":", lw=0.6, alpha=0.28)

    handles = [
        Line2D([], [], color=STABLE_PLUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=STABLE_MINUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}-}^{*}$"),
        Line2D([], [], color=UNSTABLE_PLUS, lw=1.35, ls="--", label=r"Unstable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=UNSTABLE_MINUS, lw=1.35, ls="--", label=r"Unstable $\mu_{\mathrm{NS}-}^{*}$"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.96,
        fontsize=8.4,
        handlelength=2.0,
        handletextpad=0.45,
        columnspacing=0.9,
        labelspacing=0.25,
        borderpad=0.40,
    )
    ax.text(
        -0.16,
        1.065,
        "(a)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.02,
        0.96,
        r"$\alpha_2=0$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
    )

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.83)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    records = scan_fixed_points()
    branches = track_branches(records)
    write_csv(branches)
    plot_branches(branches)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_CSV}")
    print(f"Tracked branches: {len(branches)}")


if __name__ == "__main__":
    main()
