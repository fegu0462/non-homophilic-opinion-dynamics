"""Monte Carlo model used for the representative trajectory figures.

The three trajectory notebooks import this module and differ only in the
value of ``alpha2`` and in their plotting labels.  Keeping the model here
avoids copying the microscopic update rule into every notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, kurtosis, skew


@dataclass(frozen=True)
class MonteCarloParameters:
    """Parameters shared by the three representative simulations."""

    n: int = 800
    k: float = 80.0
    t_max: int = 100
    f_s: float = 0.50
    sigma_s: float = 0.80
    sigma_ns: float = 0.10
    gamma: float = 4.0
    d_c: float = 0.50
    x0_stubborn_positive: float = 1.0
    x0_stubborn_negative: float = -1.0
    x0_nonstubborn_negative: float = -0.5
    x0_nonstubborn_positive: float = 0.5
    seed: int = 80
    trajectory_seed: int = 123
    trajectories_per_bias_group: int = 60


def build_initial_state(parameters: MonteCarloParameters, rng):
    """Construct the stubborn and non-stubborn populations.

    Non-stubborn intrinsic biases are sampled uniformly in the negative and
    positive halves of the opinion interval.  Their initial opinions are the
    symmetric delta initial condition used in the manuscript: -0.5 and +0.5.
    """

    n_stubborn = int(parameters.n * parameters.f_s)
    n_nonstubborn = parameters.n - n_stubborn
    if n_stubborn % 2 or n_nonstubborn % 2:
        raise ValueError(
            "The stubborn and non-stubborn population sizes must both be even."
        )

    stubborn_half = n_stubborn // 2
    nonstubborn_half = n_nonstubborn // 2

    eta = np.empty(parameters.n, dtype=float)
    sigma = np.empty(parameters.n, dtype=float)
    x = np.empty(parameters.n, dtype=float)

    eta[:stubborn_half] = 1.0
    eta[stubborn_half:n_stubborn] = -1.0
    sigma[:n_stubborn] = parameters.sigma_s
    x[:stubborn_half] = parameters.x0_stubborn_positive
    x[stubborn_half:n_stubborn] = parameters.x0_stubborn_negative

    negative = np.arange(n_stubborn, n_stubborn + nonstubborn_half)
    positive = np.arange(n_stubborn + nonstubborn_half, parameters.n)
    eta[negative] = rng.uniform(-1.0, 0.0, size=nonstubborn_half)
    eta[positive] = rng.uniform(0.0, 1.0, size=nonstubborn_half)
    sigma[n_stubborn:] = parameters.sigma_ns
    # Preserve the random-number stream of the original figure script.  That
    # script first sampled continuous initial opinions and then replaced them
    # by the symmetric delta initial condition used for this figure.
    x[negative] = rng.uniform(-1.0, 0.0, size=nonstubborn_half)
    x[positive] = rng.uniform(0.0, 1.0, size=nonstubborn_half)
    x[negative] = parameters.x0_nonstubborn_negative
    x[positive] = parameters.x0_nonstubborn_positive

    nonstubborn = np.arange(n_stubborn, parameters.n)
    return x, eta, sigma, nonstubborn, negative, positive


def monte_carlo_step(x, eta, sigma, alpha2, parameters, rng):
    """Advance the microscopic stochastic model by one time step."""

    alpha1 = 1.0 - alpha2
    distance = np.abs(x[:, None] - x[None, :])
    exponent = np.clip(
        parameters.gamma * (distance - parameters.d_c),
        -100.0,
        100.0,
    )
    homophily = (
        alpha1
        * (1.0 + np.exp(-parameters.gamma * parameters.d_c))
        / (1.0 + np.exp(exponent))
    )
    extremity_attraction = alpha2 * np.abs(x)[None, :]
    probability = np.tanh(
        (parameters.k / parameters.n)
        * (homophily + extremity_attraction)
    )
    np.maximum(probability, 0.0, out=probability)
    np.fill_diagonal(probability, 0.0)

    adjacency = rng.random(probability.shape) < probability
    degree = adjacency.sum(axis=1)
    safe_degree = np.where(degree == 0, 1, degree)
    neighbour_average = (adjacency @ x) / safe_degree
    neighbour_average[degree == 0] = x[degree == 0]
    return sigma * eta + (1.0 - sigma) * neighbour_average


def simulate_trajectory(alpha2, parameters=None):
    """Simulate one state and retain trajectories and population means."""

    if parameters is None:
        parameters = MonteCarloParameters()
    if not 0.0 <= alpha2 <= 1.0:
        raise ValueError("alpha2 must lie in [0, 1].")

    rng = np.random.default_rng(parameters.seed)
    x, eta, sigma, nonstubborn, negative, positive = build_initial_state(
        parameters,
        rng,
    )

    trace_rng = np.random.default_rng(parameters.trajectory_seed)
    sample_count = parameters.trajectories_per_bias_group
    sampled_negative = trace_rng.choice(
        negative,
        size=min(sample_count, negative.size),
        replace=False,
    )
    sampled_positive = trace_rng.choice(
        positive,
        size=min(sample_count, positive.size),
        replace=False,
    )
    sampled_indices = np.concatenate((sampled_negative, sampled_positive))
    sampled_is_negative = np.concatenate(
        (
            np.ones(sampled_negative.size, dtype=bool),
            np.zeros(sampled_positive.size, dtype=bool),
        )
    )
    sample_permutation = trace_rng.permutation(sampled_indices.size)
    sampled_indices = sampled_indices[sample_permutation]
    sampled_is_negative = sampled_is_negative[sample_permutation]

    times = np.arange(parameters.t_max + 1, dtype=int)
    sampled_history = np.empty((times.size, sampled_indices.size), dtype=float)
    negative_mean = np.empty(times.size, dtype=float)
    positive_mean = np.empty(times.size, dtype=float)
    total_mean = np.empty(times.size, dtype=float)

    for time in times:
        sampled_history[time] = x[sampled_indices]
        negative_mean[time] = float(np.mean(x[negative]))
        positive_mean[time] = float(np.mean(x[positive]))
        total_mean[time] = float(np.mean(x[nonstubborn]))
        if time < parameters.t_max:
            x = monte_carlo_step(
                x=x,
                eta=eta,
                sigma=sigma,
                alpha2=alpha2,
                parameters=parameters,
                rng=rng,
            )

    final_opinions = x[nonstubborn].copy()
    final_negative_opinions = x[negative].copy()
    final_positive_opinions = x[positive].copy()
    return {
        "alpha2": float(alpha2),
        "parameters": parameters,
        "times": times,
        "sampled_history": sampled_history,
        "sampled_is_negative": sampled_is_negative,
        "negative_mean": negative_mean,
        "positive_mean": positive_mean,
        "total_mean": total_mean,
        "final_opinions": final_opinions,
        "final_negative_opinions": final_negative_opinions,
        "final_positive_opinions": final_positive_opinions,
        "B": bimodality_coefficient(final_opinions),
        "S": float(np.mean(final_opinions)),
    }


def bimodality_coefficient(values):
    """Return the finite-sample-corrected bimodality coefficient."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_values = values.size
    if n_values < 4 or np.std(values) < 1.0e-12:
        return 0.0

    sample_skewness = float(skew(values, bias=False))
    excess_kurtosis = float(kurtosis(values, fisher=True, bias=False))
    correction = (
        3.0 * (n_values - 1) ** 2
        / ((n_values - 2) * (n_values - 3))
    )
    denominator = excess_kurtosis + correction
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float((sample_skewness**2 + 1.0) / denominator)


CASE_INFO = {
    "bp": ("(a)", "Bimodal polarized", "#6F4E9C"),
    "bu": ("(b)", "Biased unimodal", "#D9772B"),
    "ab": ("(c)", "Asymmetric bimodal polarized", "#6F4E9C"),
}


@dataclass(frozen=True)
class BiasInitialOpinions:
    """Initial opinions ordered from weakest to strongest positive bias and
    from strongest to weakest negative bias, respectively.
    """

    positive: np.ndarray
    negative: np.ndarray


@dataclass(frozen=True)
class FigureStyle:
    """The few visual choices that remain visible in the notebook."""

    font_family: str = "STIXGeneral"
    kde_bandwidth: float = 0.075
    histogram_bins: int = 30
    complete_size: tuple[float, float] = (7.20, 4.70)
    trajectory_size: tuple[float, float] = (7.20, 2.75)
    distribution_size: tuple[float, float] = (7.20, 2.45)
    dpi: int = 300


BLUE = "#2166AC"
RED = "#B2182B"
LIGHT_BLUE = "#92C5DE"
LIGHT_RED = "#F4A6B6"
MF_BLUE = "#00A6D6"
MF_RED = "#EF3B6B"


def _model_parameters(seed: int) -> MonteCarloParameters:
    return MonteCarloParameters(
        n=800,
        k=40,
        t_max=50,
        f_s=0.50,
        sigma_s=0.80,
        sigma_ns=0.10,
        gamma=4.0,
        d_c=0.50,
        seed=int(seed),
        trajectory_seed=123,
        trajectories_per_bias_group=200,
    )


def _build_population(parameters: MonteCarloParameters, initial: BiasInitialOpinions):
    n_stubborn = int(parameters.n * parameters.f_s)
    n_nonstubborn = parameters.n - n_stubborn
    group_size = n_nonstubborn // 2
    if n_stubborn % 2 or n_nonstubborn % 2:
        raise ValueError("The stubborn and non-stubborn populations must be even.")

    positive_initial = np.asarray(initial.positive, dtype=float)
    negative_initial = np.asarray(initial.negative, dtype=float)
    if positive_initial.shape != (group_size,) or negative_initial.shape != (group_size,):
        raise ValueError(f"Each bias group must contain {group_size} initial opinions.")
    if np.any(np.abs(np.r_[positive_initial, negative_initial]) > 1.0):
        raise ValueError("Initial opinions must lie in [-1, 1].")

    stubborn_half = n_stubborn // 2
    x = np.empty(parameters.n, dtype=float)
    eta = np.empty(parameters.n, dtype=float)
    sigma = np.empty(parameters.n, dtype=float)

    x[:stubborn_half], eta[:stubborn_half] = parameters.x0_stubborn_positive, 1.0
    x[stubborn_half:n_stubborn], eta[stubborn_half:n_stubborn] = (
        parameters.x0_stubborn_negative,
        -1.0,
    )
    sigma[:n_stubborn] = parameters.sigma_s

    positive = np.arange(n_stubborn, n_stubborn + group_size)
    negative = np.arange(n_stubborn + group_size, parameters.n)
    eta[positive] = (np.arange(group_size, dtype=float) + 0.5) / group_size
    eta[negative] = -1.0 + (np.arange(group_size, dtype=float) + 0.5) / group_size
    sigma[n_stubborn:] = parameters.sigma_ns
    x[positive] = positive_initial
    x[negative] = negative_initial

    nonstubborn = np.r_[positive, negative]
    return x, eta, sigma, nonstubborn, negative, positive


def _mean_field_step(x, eta, sigma, alpha2, parameters):
    """Expected-neighbour update corresponding to the stochastic MC step."""

    distance = np.abs(x[:, None] - x[None, :])
    exponent = np.clip(
        parameters.gamma * (distance - parameters.d_c),
        -100.0,
        100.0,
    )
    confidence = (
        (1.0 - alpha2)
        * (1.0 + np.exp(-parameters.gamma * parameters.d_c))
        / (1.0 + np.exp(exponent))
    )
    interaction = confidence + alpha2 * np.abs(x)[None, :]
    probability = np.tanh((parameters.k / parameters.n) * interaction)
    np.fill_diagonal(probability, 0.0)
    degree = probability.sum(axis=1)
    neighbour_mean = (probability @ x) / np.maximum(degree, 1.0)
    neighbour_mean[degree == 0.0] = x[degree == 0.0]
    return sigma * eta + (1.0 - sigma) * neighbour_mean


def _mean_field_fixed_point(x0, eta, sigma, alpha2, parameters):
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(5000):
        updated = _mean_field_step(x, eta, sigma, alpha2, parameters)
        if np.max(np.abs(updated - x)) < 1.0e-12:
            return updated
        x = updated
    raise RuntimeError("The mean-field iteration did not converge.")


def _sample_trajectories(negative, positive, parameters):
    rng = np.random.default_rng(parameters.trajectory_seed)
    count = parameters.trajectories_per_bias_group
    negative_sample = rng.choice(negative, min(count, negative.size), replace=False)
    positive_sample = rng.choice(positive, min(count, positive.size), replace=False)
    indices = np.r_[negative_sample, positive_sample]
    labels = np.r_[
        np.ones(negative_sample.size, dtype=bool),
        np.zeros(positive_sample.size, dtype=bool),
    ]
    order = rng.permutation(indices.size)
    return indices[order], labels[order]


def simulate_case(
    key: str,
    initial: BiasInitialOpinions,
    seed: int,
    alpha2: float = 0.03,
):
    """Run one fresh MC realization and compute its MF reference."""

    if key not in CASE_INFO:
        raise KeyError(f"Unknown case: {key}")
    parameters = _model_parameters(seed)
    x, eta, sigma, nonstubborn, negative, positive = _build_population(
        parameters,
        initial,
    )
    initial_x = x.copy()
    sampled, sampled_is_negative = _sample_trajectories(
        negative,
        positive,
        parameters,
    )

    rng = np.random.default_rng(parameters.seed)
    times = np.arange(parameters.t_max + 1)
    history = np.empty((times.size, sampled.size), dtype=float)
    negative_mean = np.empty(times.size, dtype=float)
    positive_mean = np.empty(times.size, dtype=float)

    for time in times:
        history[time] = x[sampled]
        negative_mean[time] = np.mean(x[negative])
        positive_mean[time] = np.mean(x[positive])
        if time < parameters.t_max:
            x = monte_carlo_step(x, eta, sigma, alpha2, parameters, rng)

    mf = _mean_field_fixed_point(initial_x, eta, sigma, alpha2, parameters)
    final = x[nonstubborn].copy()
    panel, title, density_color = CASE_INFO[key]
    return {
        "key": key,
        "panel": panel,
        "title": title,
        "density_color": density_color,
        "parameters": parameters,
        "times": times,
        "sampled_history": history,
        "sampled_is_negative": sampled_is_negative,
        "negative_mean": negative_mean,
        "positive_mean": positive_mean,
        "final_opinions": final,
        "final_negative_opinions": x[negative].copy(),
        "final_positive_opinions": x[positive].copy(),
        "negative_mf": float(np.mean(mf[negative])),
        "positive_mf": float(np.mean(mf[positive])),
        "B": bimodality_coefficient(final),
        "S": float(abs(np.mean(final))),
    }


def run_three_states(
    initial_opinions: Mapping[str, BiasInitialOpinions],
    seeds: Mapping[str, int],
    alpha2: float = 0.03,
):
    """Run all three cases directly from the supplied initial opinions."""

    return {
        key: simulate_case(key, initial_opinions[key], seeds[key], alpha2)
        for key in CASE_INFO
    }


def print_summary(results):
    for result in results.values():
        p = result["parameters"]
        print(
            f"{result['title']}: N={p.n}, K={p.k:g}, T={p.t_max}, "
            f"seed={p.seed}, B={result['B']:.3f}, S={result['S']:.3f}"
        )


def apply_figure_style(style: FigureStyle):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [style.font_family, "Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _style_axis(ax):
    ax.tick_params(direction="out", top=False, right=False, width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def _draw_trajectory(ax, result):
    negative = result["sampled_is_negative"]
    history = result["sampled_history"]
    times = result["times"]
    ax.plot(times, history[:, negative], color=LIGHT_BLUE, alpha=0.10, lw=0.35)
    ax.plot(times, history[:, ~negative], color=LIGHT_RED, alpha=0.10, lw=0.35)
    ax.plot(times, result["negative_mean"], color=BLUE, lw=1.25)
    ax.plot(times, result["positive_mean"], color=RED, lw=1.25)
    ax.axhline(result["negative_mf"], color=MF_BLUE, ls="--", lw=1.05)
    ax.axhline(result["positive_mf"], color=MF_RED, ls="--", lw=1.05)
    ax.set(xlim=(0, 50), ylim=(-1.02, 1.02), xlabel=r"$t$", ylabel=r"$x_i(t)$")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    _style_axis(ax)


def _draw_distribution(ax, result, style):
    bins = np.linspace(-1.0, 1.0, style.histogram_bins + 1)
    bin_width = bins[1] - bins[0]
    groups = [
        result["final_negative_opinions"],
        result["final_positive_opinions"],
    ]
    total = sum(group.size for group in groups)
    weights = [np.full(group.size, 1.0 / (total * bin_width)) for group in groups]
    ax.hist(
        groups,
        bins=bins,
        weights=weights,
        stacked=True,
        color=[BLUE, RED],
        alpha=0.32,
        edgecolor="none",
    )

    values = result["final_opinions"]
    grid = np.linspace(-1.0, 1.0, 1000)
    bandwidth_factor = style.kde_bandwidth / np.std(values, ddof=1)
    density = gaussian_kde(values, bw_method=bandwidth_factor)(grid)
    color = result["density_color"]
    ax.fill_between(grid, 0.0, density, color=color, alpha=0.10, linewidth=0)
    ax.plot(grid, density, color=color, lw=1.35)

    ax.axvline(result["negative_mean"][-1], color=BLUE, lw=1.25)
    ax.axvline(result["positive_mean"][-1], color=RED, lw=1.25)
    ax.axvline(result["negative_mf"], color=MF_BLUE, ls="--", lw=1.05)
    ax.axvline(result["positive_mf"], color=MF_RED, ls="--", lw=1.05)
    ax.set(xlim=(-1.0, 1.0), ylim=(0, None), xlabel=r"$x_i$", ylabel=r"$p(x_i)$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    _style_axis(ax)


def _legend_handles():
    return [
        Line2D([0], [0], color=BLUE, lw=1.25, label="Negative-bias MC mean"),
        Line2D([0], [0], color=RED, lw=1.25, label="Positive-bias MC mean"),
        Line2D([0], [0], color=MF_BLUE, ls="--", lw=1.05, label="Negative-bias MF"),
        Line2D([0], [0], color=MF_RED, ls="--", lw=1.05, label="Positive-bias MF"),
    ]


def make_trajectory_figure(results, style: FigureStyle):
    apply_figure_style(style)
    fig, axes = plt.subplots(1, 3, figsize=style.trajectory_size, dpi=style.dpi)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.20, top=0.68, wspace=0.34)
    for ax, result in zip(axes, results.values()):
        _draw_trajectory(ax, result)
        ax.set_title(result["title"], fontsize=9, fontweight="bold", pad=7)
        ax.text(-0.14, 1.10, result["panel"], transform=ax.transAxes,
                fontsize=10.5, fontweight="bold")
    fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=4, frameon=True, fontsize=6.2, handlelength=2.2, columnspacing=0.9)
    return fig


def make_distribution_figure(results, style: FigureStyle):
    apply_figure_style(style)
    fig, axes = plt.subplots(1, 3, figsize=style.distribution_size, dpi=style.dpi)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.77, wspace=0.34)
    for ax, result in zip(axes, results.values()):
        _draw_distribution(ax, result, style)
        ax.set_title(
            rf"$B={result['B']:.3f},\quad S={result['S']:.3f}$",
            fontsize=8,
            pad=5,
        )
    return fig


def make_complete_figure(results, style: FigureStyle):
    apply_figure_style(style)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=style.complete_size,
        dpi=style.dpi,
        gridspec_kw={"height_ratios": [1.0, 0.78]},
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.115,
        top=0.715,
        wspace=0.34,
        hspace=0.56,
    )
    for column, result in enumerate(results.values()):
        _draw_trajectory(axes[0, column], result)
        _draw_distribution(axes[1, column], result, style)
        axes[0, column].set_title(result["title"], fontsize=9, fontweight="bold", pad=7)
        axes[0, column].text(
            -0.14,
            1.10,
            result["panel"],
            transform=axes[0, column].transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
        axes[1, column].set_title(
            rf"$B={result['B']:.3f},\quad S={result['S']:.3f}$",
            fontsize=8,
            pad=5,
        )
    fig.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.82),
               ncol=4, frameon=True, fontsize=6.2, handlelength=2.2, columnspacing=0.9)
    return fig


def save_figure(fig, output_base):
    """Save one figure after the rendered multi-panel alignment check."""

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=str(output_base) + ".alignment.json",
            overlay_svg=str(output_base) + ".alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    except ImportError:
        pass
    fig.savefig(str(output_base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(output_base) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_base) + ".svg", bbox_inches="tight")

