"""Compute and plot the two four-bin mean-field bifurcation slices."""

import csv
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import fsolve, linear_sum_assignment


##############
# Parameters #
##############


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "four_bin"
FIGURE_DIR = ROOT / "figures" / "four_bin"

GAMMA = 4.0
DC = 0.5
SIGMA = np.array([0.8, 0.8, 0.1, 0.1])
ETA = np.array([1.0, -1.0, 0.5, -0.5])

STABLE_PLUS = "#3F6FD8"
STABLE_MINUS = "#2F9B45"
UNSTABLE_PLUS = "#EF8FA4"
UNSTABLE_MINUS = "#F4A64F"
FILL_BIMODAL = "#F3A6AC"
FILL_BIASED = "#F7D875"
FILL_NEUTRAL = "#A9CDEC"
FILL_UNSTABLE = "#E8B8C0"
SADDLE_COLOR = "#19C719"
PITCHFORK_COLOR = "#F2CF00"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


####################
# Mean-field model #
####################


def population_weights(f_s):
    """Population fractions of the two stubborn and two flexible groups."""
    return np.array([f_s / 2, f_s / 2, (1 - f_s) / 2, (1 - f_s) / 2])


def mean_field_map(mu, alpha2, f_s):
    """Return one iteration of the four-bin mean-field map."""
    mu = np.asarray(mu, dtype=float)
    distance = np.abs(mu[:, None] - mu[None, :])
    confidence = (1 + np.exp(-GAMMA * DC)) / (
        1 + np.exp(np.clip(GAMMA * (distance - DC), -100, 100))
    )
    interaction = (1 - alpha2) * confidence + alpha2 * np.abs(mu)[None, :]
    weights = population_weights(f_s)
    denominator = interaction @ weights
    neighbour_mean = np.divide(
        (interaction * weights[None, :]) @ mu,
        denominator,
        out=np.zeros(4),
        where=denominator > 1.0e-14,
    )
    return SIGMA * ETA + (1 - SIGMA) * neighbour_mean


def fixed_point_residual(mu, alpha2, f_s):
    return mean_field_map(mu, alpha2, f_s) - np.asarray(mu)


def map_jacobian(mu, alpha2, f_s):
    """Numerical Jacobian of the discrete-time mean-field map."""
    mu = np.asarray(mu, dtype=float)
    jacobian = np.empty((4, 4))
    for column in range(4):
        step = 1.0e-6 * max(1.0, abs(mu[column]))
        plus, minus = mu.copy(), mu.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            mean_field_map(plus, alpha2, f_s)
            - mean_field_map(minus, alpha2, f_s)
        ) / (2 * step)
    return jacobian


def _static_guesses():
    bases = np.array([
        [0.9, -0.9, 0.5, -0.5], [0.9, -0.9, 0.0, 0.0],
        [0.9, -0.9, 0.8, -0.8], [0.9, -0.9, 0.3, -0.3],
        [0.9, -0.9, 0.8, 0.8], [0.9, -0.9, -0.8, -0.8],
        [0.9, -0.9, -0.5, 0.5], [1.0, -1.0, 1.0, -1.0],
        [0.5, -0.5, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.4, 0.7, 0.5], [-0.4, -0.8, -0.5, -0.7],
    ])
    rng = np.random.default_rng(20260922)
    guesses = [row.copy() for row in bases]
    for base in bases:
        guesses.extend(np.clip(base + rng.normal(0, 0.16, (5, 4)), -1, 1))
    guesses.extend(rng.uniform(-0.95, 0.95, (80, 4)))
    return guesses


STATIC_GUESSES = _static_guesses()


##########################
# Fixed points and tracks #
##########################


def _unique(values, tolerance=1.0e-9):
    answer = []
    for value in values:
        value = np.asarray(value, dtype=float)
        if np.all(np.isfinite(value)) and not any(
            np.max(np.abs(value - old)) < tolerance for old in answer
        ):
            answer.append(value)
    return answer


def solve_fixed_points(alpha2, f_s, previous=(), older=()):
    """Find all physical fixed points from continuation and static guesses."""
    guesses = list(previous)
    if previous and older:
        for value in previous:
            nearest = min(older, key=lambda old: np.linalg.norm(value - old))
            if np.linalg.norm(value - nearest) < 0.25:
                guesses.append(2 * value - nearest)
    guesses.extend(STATIC_GUESSES)

    roots = []
    for guess in _unique(guesses):
        root, _, flag, _ = fsolve(
            fixed_point_residual,
            guess,
            args=(alpha2, f_s),
            full_output=True,
            xtol=1.0e-11,
            maxfev=1800,
        )
        error = np.max(np.abs(fixed_point_residual(root, alpha2, f_s)))
        if flag != 1 or error > 2.0e-9 or np.max(np.abs(root)) > 1.0001:
            continue
        if any(np.max(np.abs(root - old)) < 2.0e-5 for old in roots):
            continue
        roots.append(np.asarray(root))
    roots.sort(key=lambda value: tuple(np.round(value, 10)))
    return roots


def scan_slice(parameter, values, fixed_value):
    """Scan fixed points while varying ``alpha2`` or ``f_s``."""
    if parameter not in {"alpha2", "f_s"}:
        raise ValueError("parameter must be 'alpha2' or 'f_s'.")

    records, older, previous = [], [], []
    for index, value in enumerate(np.asarray(values, dtype=float)):
        alpha2, f_s = (value, fixed_value) if parameter == "alpha2" else (fixed_value, value)
        roots = solve_fixed_points(alpha2, f_s, previous, older)
        solutions = []
        for root in roots:
            radius = float(np.max(np.abs(np.linalg.eigvals(map_jacobian(root, alpha2, f_s)))))
            solutions.append({
                "root": root,
                "stable": radius < 1 - 2.0e-6,
                "spectral_radius": radius,
            })
        records.append({"parameter": value, "index": index, "roots": solutions})
        older, previous = previous, [root.copy() for root in roots]
    return records


def track_branches(records):
    """Match full four-dimensional fixed points into continuous branches."""
    branches = []
    for entry in records:
        index, parameter = entry["index"], entry["parameter"]
        solutions = entry["roots"]
        roots = [item["root"] for item in solutions]
        active = [b for b in branches if index - b[-1]["index"] <= 2]
        used = set()

        if active and roots:
            predictions = []
            for branch in active:
                prediction = branch[-1]["root"]
                if len(branch) > 1:
                    prediction = prediction + prediction - branch[-2]["root"]
                predictions.append(prediction)
            cost = np.array([[np.linalg.norm(p - r) for r in roots] for p in predictions])
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] > 0.16:
                    continue
                item = solutions[column]
                active[row].append(_branch_point(entry, item))
                used.add(column)

        for index_in_row, item in enumerate(solutions):
            if index_in_row not in used:
                branches.append([_branch_point(entry, item)])
    return [branch for branch in branches if len(branch) >= 2]


def _branch_point(entry, item):
    root = np.asarray(item["root"])
    return {
        "parameter": float(entry["parameter"]),
        "index": int(entry["index"]),
        "root": root,
        "mu_plus": float(root[2]),
        "mu_minus": float(root[3]),
        "stable": bool(item["stable"]),
        "spectral_radius": float(item["spectral_radius"]),
    }


def _coarse_events(branches):
    parameters = sorted({point["parameter"] for branch in branches for point in branch})
    stable_counts = [
        sum(point["stable"] for branch in branches for point in branch
            if np.isclose(point["parameter"], value, atol=1.0e-12))
        for value in parameters
    ]
    events = [
        (left + right) / 2
        for left, right, n_left, n_right in zip(
            parameters[:-1], parameters[1:], stable_counts[:-1], stable_counts[1:]
        )
        if n_left != n_right
    ]
    merged = []
    for value in events:
        if not merged or value - merged[-1][-1] > 0.006:
            merged.append([value])
        else:
            merged[-1].append(value)
    return [float(np.mean(group)) for group in merged]


def _critical_equations(z, parameter, fixed_value):
    value, mu = float(z[4]), np.asarray(z[:4])
    alpha2, f_s = (value, fixed_value) if parameter == "alpha2" else (fixed_value, value)
    residual = fixed_point_residual(mu, alpha2, f_s)
    determinant = np.linalg.det(map_jacobian(mu, alpha2, f_s) - np.eye(4))
    return np.r_[residual, determinant]


def refine_event(coarse, records, parameter, fixed_value):
    """Refine a grid-detected event by solving F=0 and det(J-I)=0."""
    candidates = []
    for entry in records:
        if abs(entry["parameter"] - coarse) <= 0.03:
            candidates.extend((entry["parameter"], item["root"]) for item in entry["roots"])
    solutions = []
    for value, root in candidates:
        z, _, flag, _ = fsolve(
            _critical_equations,
            np.r_[root, value],
            args=(parameter, fixed_value),
            full_output=True,
            xtol=1.0e-11,
            maxfev=5000,
        )
        error = np.max(np.abs(_critical_equations(z, parameter, fixed_value)))
        if flag == 1 and error < 1.0e-7 and abs(z[4] - coarse) < 0.04:
            solutions.append(float(z[4]))
    if not solutions:
        return float(coarse)
    return min(solutions, key=lambda value: abs(value - coarse))


def compute_slice(parameter, values, fixed_value):
    """Compute branches and the two bifurcation points for one slice."""
    records = scan_slice(parameter, values, fixed_value)
    branches = track_branches(records)
    events = _coarse_events(branches)
    if len(events) < 2:
        switches = sorted(
            (left["parameter"] + right["parameter"]) / 2
            for branch in branches
            for left, right in zip(branch[:-1], branch[1:])
            if left["stable"] != right["stable"]
        )
        events = sorted(set(events + switches))
    if len(events) < 2:
        raise RuntimeError(f"Could not locate both {parameter} bifurcations.")
    events = [refine_event(value, records, parameter, fixed_value) for value in events[:2]]
    return records, branches, sorted(events)


############
# Plotting #
############


def _segments(branch, maximum_gap=2):
    segments = [[branch[0]]]
    for point in branch[1:]:
        previous = segments[-1][-1]
        if point["stable"] != previous["stable"] or point["index"] - previous["index"] > maximum_gap:
            segments.append([point])
        else:
            segments[-1].append(point)
    return segments


def _fill_color(mu_plus, mu_minus, stable):
    if not stable:
        return FILL_UNSTABLE
    if np.mean(np.abs((mu_plus + mu_minus) / 2)) > 0.08:
        return FILL_BIASED
    return FILL_BIMODAL if np.mean(np.abs(mu_plus - mu_minus)) > 0.30 else FILL_NEUTRAL


def _draw_branches(ax, branches):
    for branch in branches:
        for segment in _segments(branch):
            if len(segment) < 2:
                continue
            x = np.array([point["parameter"] for point in segment])
            plus = np.array([point["mu_plus"] for point in segment])
            minus = np.array([point["mu_minus"] for point in segment])
            stable = segment[0]["stable"]
            ax.fill_between(x, plus, minus, color=_fill_color(plus, minus, stable),
                            alpha=0.24 if stable else 0.17, linewidth=0)
            for y, stable_color, unstable_color in (
                (plus, STABLE_PLUS, UNSTABLE_PLUS),
                (minus, STABLE_MINUS, UNSTABLE_MINUS),
            ):
                ax.plot(x, y, color=stable_color if stable else unstable_color,
                        ls="-" if stable else "--", lw=1.75 if stable else 1.3,
                        marker="o" if stable else "x", ms=2.25,
                        markevery=max(1, len(segment) // 24), markeredgewidth=0.7)


def _draw_event(ax, value, color):
    line = ax.axvline(value, color=color, lw=2.1, ls=(0, (7, 5)), zorder=6)
    line.set_path_effects([
        path_effects.Stroke(linewidth=3.2, foreground="black", alpha=0.43),
        path_effects.Normal(),
    ])


def _style_axis(ax, xlabel, panel):
    ax.axhline(0, color="0.45", ls=(0, (3, 2)), lw=0.85)
    ax.set(ylim=(-1.02, 1.02), xlabel=xlabel,
           ylabel=r"$\mu_{\mathrm{NS}+}^{*},\ \mu_{\mathrm{NS}-}^{*}$")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=13)
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, ls=":", lw=0.6, alpha=0.28)
    ax.text(-0.16, 1.04, panel, transform=ax.transAxes, fontsize=14, fontweight="bold")
    handles = [
        Line2D([], [], color=STABLE_PLUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=STABLE_MINUS, lw=1.75, label=r"Stable $\mu_{\mathrm{NS}-}^{*}$"),
        Line2D([], [], color=UNSTABLE_PLUS, lw=1.3, ls="--", label=r"Unstable $\mu_{\mathrm{NS}+}^{*}$"),
        Line2D([], [], color=UNSTABLE_MINUS, lw=1.3, ls="--", label=r"Unstable $\mu_{\mathrm{NS}-}^{*}$"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=2, fontsize=10.2, frameon=True)


def draw_alpha2_panel(ax, branches, events, panel="(a)"):
    _draw_branches(ax, branches)
    _draw_event(ax, events[0], SADDLE_COLOR)
    _draw_event(ax, events[1], PITCHFORK_COLOR)
    ax.set_xlim(0, 0.30)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    _style_axis(ax, r"$\alpha_2$", panel)


def draw_fs_panel(ax, branches, events, panel="(b)"):
    _draw_branches(ax, branches)
    _draw_event(ax, events[0], PITCHFORK_COLOR)
    _draw_event(ax, events[1], SADDLE_COLOR)
    ax.set_xlim(0.15, 0.55)
    ax.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55])
    _style_axis(ax, r"$f_s$", panel)


def save_results(alpha_branches, alpha_events, fs_branches, fs_events):
    """Write branch data and save the two-panel publication figure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(DATA_DIR / "alpha2_branches.csv", alpha_branches, "alpha2")
    _write_csv(DATA_DIR / "fs_branches.csv", fs_branches, "fs")

    fig, axes = plt.subplots(2, 1, figsize=(4.95, 8.90), dpi=300)
    draw_alpha2_panel(axes[0], alpha_branches, alpha_events)
    draw_fs_panel(axes[1], fs_branches, fs_events)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.075, top=0.90, hspace=0.46)
    paths = {}
    for suffix in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"four_bin_bifurcation_slices.{suffix}"
        fig.savefig(path, dpi=600)
        paths[suffix] = path
    plt.close(fig)
    return paths


def _write_csv(path, branches, parameter_name):
    fields = ["branch_id", parameter_name, "is_stable", "spectral_radius",
              "mu_s_plus", "mu_s_minus", "mu_ns_plus", "mu_ns_minus"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for branch_id, branch in enumerate(branches):
            for point in branch:
                root = point["root"]
                writer.writerow({
                    "branch_id": branch_id,
                    parameter_name: point["parameter"],
                    "is_stable": int(point["stable"]),
                    "spectral_radius": point["spectral_radius"],
                    "mu_s_plus": root[0], "mu_s_minus": root[1],
                    "mu_ns_plus": root[2], "mu_ns_minus": root[3],
                })


def compute_default_slices(alpha_step=0.001, fs_step=0.005):
    """Compute both manuscript slices from the model equations."""
    alpha_values = np.arange(0, 0.30001, alpha_step)
    fs_values = np.arange(0.15, 0.55001, fs_step)
    _, alpha_branches, alpha_events = compute_slice("alpha2", alpha_values, 0.50)
    _, fs_branches, fs_events = compute_slice("f_s", fs_values, 0.0)
    return alpha_branches, alpha_events, fs_branches, fs_events
