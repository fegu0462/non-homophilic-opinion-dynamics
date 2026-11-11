"""Initial-condition experiments used for the second trajectory figure.

The bimodal-polarized and biased-unimodal cases use the two-group Monte Carlo
initialization from :mod:`src.monte_carlo`.  The asymmetric-bimodal case uses
the sign pattern of a stable 42-bin mean-field solution: 30 non-stubborn bins
start at ``+0.5`` and 10 bins start at ``-0.5``.  Only that sign pattern is
used to initialize the Monte Carlo population; the mean-field amplitudes are
retained solely for the dashed reference lines in the figure.
"""

from __future__ import annotations

import numpy as np

from .monte_carlo import (
    MonteCarloParameters,
    bimodality_coefficient,
    monte_carlo_step,
)


# Stable 42-bin mean-field solution at fs=0.5 and alpha2=0.03.  The first two
# entries of the complete solution are the stubborn bins; the array below
# contains only the 40 non-stubborn bins.
ASYMMETRIC_MF_NONSTUBBORN = np.array(
    [
        0.66842919,
        0.67426926,
        0.68010016,
        0.68592194,
        0.69173460,
        0.69753818,
        0.70333267,
        0.70911807,
        0.71489439,
        0.72066159,
        0.72641967,
        0.73216859,
        0.73790831,
        0.74363878,
        0.74935995,
        0.75507176,
        0.76077412,
        0.76646697,
        0.77215021,
        0.77782374,
        -0.80810554,
        -0.80175545,
        -0.79535990,
        -0.78891711,
        -0.78242507,
        -0.77588160,
        -0.76928423,
        -0.76263025,
        -0.75591663,
        -0.74913999,
        0.60951172,
        0.61544699,
        0.62137231,
        0.62728780,
        0.63319358,
        0.63908974,
        0.64497638,
        0.65085358,
        0.65672140,
        0.66257992,
    ],
    dtype=float,
)


def _build_asymmetric_population(parameters: MonteCarloParameters):
    """Embed the 42-bin mean-field sign pattern in an individual population."""

    n_bins = ASYMMETRIC_MF_NONSTUBBORN.size
    n_stubborn = int(round(parameters.n * parameters.f_s))
    n_nonstubborn = parameters.n - n_stubborn
    if n_stubborn % 2:
        raise ValueError("The stubborn population must contain an even number of agents.")
    if n_nonstubborn % n_bins:
        raise ValueError("The non-stubborn population must be divisible by 40 bins.")

    stubborn_per_sign = n_stubborn // 2
    agents_per_bin = n_nonstubborn // n_bins
    eta_bins = np.concatenate(
        (
            (np.arange(20, dtype=float) + 0.5) / 20.0,
            -1.0 + (np.arange(20, dtype=float) + 0.5) / 20.0,
        )
    )

    x = np.empty(parameters.n, dtype=float)
    eta = np.empty(parameters.n, dtype=float)
    sigma = np.empty(parameters.n, dtype=float)
    group_id = np.full(parameters.n, -1, dtype=int)

    cursor = 0
    stop = cursor + stubborn_per_sign
    x[cursor:stop] = parameters.x0_stubborn_positive
    eta[cursor:stop] = 1.0
    sigma[cursor:stop] = parameters.sigma_s
    cursor = stop

    stop = cursor + stubborn_per_sign
    x[cursor:stop] = parameters.x0_stubborn_negative
    eta[cursor:stop] = -1.0
    sigma[cursor:stop] = parameters.sigma_s
    cursor = stop

    for bin_index, (eta_value, mean_field_value) in enumerate(
        zip(eta_bins, ASYMMETRIC_MF_NONSTUBBORN)
    ):
        stop = cursor + agents_per_bin
        x[cursor:stop] = 0.5 if mean_field_value > 0.0 else -0.5
        eta[cursor:stop] = eta_value
        sigma[cursor:stop] = parameters.sigma_ns
        group_id[cursor:stop] = bin_index
        cursor = stop

    nonstubborn = np.flatnonzero(group_id >= 0)
    negative_bins = ASYMMETRIC_MF_NONSTUBBORN < 0.0
    negative = np.flatnonzero(negative_bins[group_id.clip(min=0)] & (group_id >= 0))
    positive = np.flatnonzero((~negative_bins[group_id.clip(min=0)]) & (group_id >= 0))
    return x, eta, sigma, nonstubborn, negative, positive


def simulate_asymmetric_bimodal(alpha2=0.03, parameters=None):
    """Simulate the asymmetric-bimodal initial condition.

    The manuscript realization uses ``N=800``, ``K=40``, ``t_max=80`` and
    ``seed=20261188``.  Passing an explicit parameter object keeps those
    choices visible and editable in the notebook.
    """

    if parameters is None:
        parameters = MonteCarloParameters(
            n=800,
            k=40,
            t_max=80,
            f_s=0.50,
            sigma_s=0.80,
            sigma_ns=0.10,
            gamma=4.0,
            d_c=0.50,
            seed=20261188,
            trajectory_seed=123,
            trajectories_per_bias_group=400,
        )
    if not 0.0 <= alpha2 <= 1.0:
        raise ValueError("alpha2 must lie in [0, 1].")

    x, eta, sigma, nonstubborn, negative, positive = _build_asymmetric_population(
        parameters
    )
    rng = np.random.default_rng(parameters.seed)
    times = np.arange(parameters.t_max + 1, dtype=int)
    sampled_indices = np.concatenate((negative, positive))
    sampled_is_negative = np.concatenate(
        (
            np.ones(negative.size, dtype=bool),
            np.zeros(positive.size, dtype=bool),
        )
    )

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
    negative_mf = float(np.mean(ASYMMETRIC_MF_NONSTUBBORN[ASYMMETRIC_MF_NONSTUBBORN < 0]))
    positive_mf = float(np.mean(ASYMMETRIC_MF_NONSTUBBORN[ASYMMETRIC_MF_NONSTUBBORN > 0]))
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
        "negative_mf": negative_mf,
        "positive_mf": positive_mf,
    }

