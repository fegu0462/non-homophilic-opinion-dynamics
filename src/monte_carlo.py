"""Monte Carlo model used for the representative trajectory figures.

The three trajectory notebooks import this module and differ only in the
value of ``alpha2`` and in their plotting labels.  Keeping the model here
avoids copying the microscopic update rule into every notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, skew


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
