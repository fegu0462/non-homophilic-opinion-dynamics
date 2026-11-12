"""Monte Carlo model for the trajectory and distribution figures."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, skew


##############
# Parameters #
##############


@dataclass(frozen=True)
class MonteCarloParameters:
    n: int = 800
    k: float = 40.0
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


# The 40 non-stubborn components of the asymmetric 42-bin fixed point.
ASYMMETRIC_MF_NONSTUBBORN = np.array(
    [
        0.66842919, 0.67426926, 0.68010016, 0.68592194, 0.69173460,
        0.69753818, 0.70333267, 0.70911807, 0.71489439, 0.72066159,
        0.72641967, 0.73216859, 0.73790831, 0.74363878, 0.74935995,
        0.75507176, 0.76077412, 0.76646697, 0.77215021, 0.77782374,
       -0.80810554,-0.80175545,-0.79535990,-0.78891711,-0.78242507,
       -0.77588160,-0.76928423,-0.76263025,-0.75591663,-0.74913999,
        0.60951172, 0.61544699, 0.62137231, 0.62728780, 0.63319358,
        0.63908974, 0.64497638, 0.65085358, 0.65672140, 0.66257992,
    ],
    dtype=float,
)


#############
# Statistics #
#############


def bimodality_coefficient(values):
    """Return the finite-sample-corrected bimodality coefficient."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 4 or np.std(values) < 1.0e-12:
        return 0.0
    g1 = float(skew(values, bias=False))
    g2 = float(kurtosis(values, fisher=True, bias=False))
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = g2 + correction
    return float((g1 * g1 + 1.0) / denominator) if denominator > 0 else np.nan


####################
# Initial conditions #
####################


def build_initial_state(parameters, rng):
    """Create the standard symmetric two-group initial condition."""
    n_s = int(parameters.n * parameters.f_s)
    n_ns = parameters.n - n_s
    if n_s % 2 or n_ns % 2:
        raise ValueError("The stubborn and non-stubborn populations must be even.")

    s_half, ns_half = n_s // 2, n_ns // 2
    x = np.empty(parameters.n)
    eta = np.empty(parameters.n)
    sigma = np.empty(parameters.n)

    eta[:s_half], eta[s_half:n_s] = 1.0, -1.0
    sigma[:n_s] = parameters.sigma_s
    x[:s_half] = parameters.x0_stubborn_positive
    x[s_half:n_s] = parameters.x0_stubborn_negative

    negative = np.arange(n_s, n_s + ns_half)
    positive = np.arange(n_s + ns_half, parameters.n)
    eta[negative] = rng.uniform(-1.0, 0.0, ns_half)
    eta[positive] = rng.uniform(0.0, 1.0, ns_half)
    sigma[n_s:] = parameters.sigma_ns

    # Keep the random stream identical to the original simulation script.
    rng.uniform(-1.0, 0.0, ns_half)
    rng.uniform(0.0, 1.0, ns_half)
    x[negative] = parameters.x0_nonstubborn_negative
    x[positive] = parameters.x0_nonstubborn_positive
    return x, eta, sigma, np.arange(n_s, parameters.n), negative, positive


def build_asymmetric_state(parameters):
    """Use the sign pattern of the asymmetric 42-bin fixed point."""
    n_s = int(parameters.n * parameters.f_s)
    n_ns = parameters.n - n_s
    if n_s % 2 or n_ns % 40:
        raise ValueError("The stubborn population must be even and N_ns divisible by 40.")

    x = np.empty(parameters.n)
    eta = np.empty(parameters.n)
    sigma = np.empty(parameters.n)
    group = np.full(parameters.n, -1, dtype=int)
    s_half = n_s // 2

    x[:s_half], eta[:s_half] = parameters.x0_stubborn_positive, 1.0
    x[s_half:n_s], eta[s_half:n_s] = parameters.x0_stubborn_negative, -1.0
    sigma[:n_s] = parameters.sigma_s

    eta_bins = np.r_[(np.arange(20) + 0.5) / 20.0,
                     -1.0 + (np.arange(20) + 0.5) / 20.0]
    agents_per_bin = n_ns // 40
    cursor = n_s
    for bin_id, (eta_value, fixed_point) in enumerate(
        zip(eta_bins, ASYMMETRIC_MF_NONSTUBBORN)
    ):
        stop = cursor + agents_per_bin
        x[cursor:stop] = 0.5 if fixed_point > 0 else -0.5
        eta[cursor:stop] = eta_value
        sigma[cursor:stop] = parameters.sigma_ns
        group[cursor:stop] = bin_id
        cursor = stop

    nonstubborn = np.arange(n_s, parameters.n)
    is_negative_bin = ASYMMETRIC_MF_NONSTUBBORN < 0
    negative = nonstubborn[is_negative_bin[group[nonstubborn]]]
    positive = nonstubborn[~is_negative_bin[group[nonstubborn]]]
    return x, eta, sigma, nonstubborn, negative, positive


#######################
# Monte Carlo dynamics #
#######################


def monte_carlo_step(x, eta, sigma, alpha2, parameters, rng):
    """Advance all agents by one synchronous Monte Carlo step."""
    distance = np.abs(x[:, None] - x[None, :])
    confidence = (1.0 + np.exp(-parameters.gamma * parameters.d_c)) / (
        1.0 + np.exp(np.clip(parameters.gamma * (distance - parameters.d_c), -100, 100))
    )
    interaction = (1.0 - alpha2) * confidence + alpha2 * np.abs(x)[None, :]
    probability = np.tanh((parameters.k / parameters.n) * interaction)
    np.maximum(probability, 0.0, out=probability)
    np.fill_diagonal(probability, 0.0)

    adjacency = rng.random(probability.shape) < probability
    degree = adjacency.sum(axis=1)
    neighbour_mean = (adjacency @ x) / np.maximum(degree, 1)
    neighbour_mean[degree == 0] = x[degree == 0]
    return sigma * eta + (1.0 - sigma) * neighbour_mean


def _sample_indices(negative, positive, parameters):
    rng = np.random.default_rng(parameters.trajectory_seed)
    count = parameters.trajectories_per_bias_group
    negative = rng.choice(negative, min(count, negative.size), replace=False)
    positive = rng.choice(positive, min(count, positive.size), replace=False)
    indices = np.r_[negative, positive]
    labels = np.r_[np.ones(negative.size, dtype=bool),
                   np.zeros(positive.size, dtype=bool)]
    order = rng.permutation(indices.size)
    return indices[order], labels[order]


def _simulate(alpha2, parameters, initial_state, sample_all=False):
    if not 0.0 <= alpha2 <= 1.0:
        raise ValueError("alpha2 must lie in [0, 1].")

    rng = np.random.default_rng(parameters.seed)
    x, eta, sigma, nonstubborn, negative, positive = initial_state(parameters, rng)
    if sample_all:
        sampled = np.r_[negative, positive]
        sampled_is_negative = np.r_[np.ones(negative.size, dtype=bool),
                                    np.zeros(positive.size, dtype=bool)]
    else:
        sampled, sampled_is_negative = _sample_indices(negative, positive, parameters)

    times = np.arange(parameters.t_max + 1)
    history = np.empty((times.size, sampled.size))
    negative_mean = np.empty(times.size)
    positive_mean = np.empty(times.size)
    total_mean = np.empty(times.size)

    for time in times:
        history[time] = x[sampled]
        negative_mean[time] = np.mean(x[negative])
        positive_mean[time] = np.mean(x[positive])
        total_mean[time] = np.mean(x[nonstubborn])
        if time < parameters.t_max:
            x = monte_carlo_step(x, eta, sigma, alpha2, parameters, rng)

    final = x[nonstubborn].copy()
    return {
        "alpha2": float(alpha2),
        "parameters": parameters,
        "times": times,
        "sampled_history": history,
        "sampled_is_negative": sampled_is_negative,
        "negative_mean": negative_mean,
        "positive_mean": positive_mean,
        "total_mean": total_mean,
        "final_opinions": final,
        "B": bimodality_coefficient(final),
        "S": float(abs(np.mean(final))),
    }


def simulate_trajectory(alpha2, parameters=None):
    """Simulate the standard symmetric initial condition."""
    parameters = parameters or MonteCarloParameters()
    return _simulate(alpha2, parameters, build_initial_state)


def simulate_asymmetric_bimodal(alpha2=0.03, parameters=None):
    """Simulate the asymmetric-bimodal initial condition."""
    parameters = parameters or MonteCarloParameters(
        t_max=50,
        seed=62,
        trajectories_per_bias_group=200,
    )
    result = _simulate(
        alpha2,
        parameters,
        lambda p, _rng: build_asymmetric_state(p),
        sample_all=True,
    )
    values = ASYMMETRIC_MF_NONSTUBBORN
    result["negative_mf"] = float(np.mean(values[values < 0]))
    result["positive_mf"] = float(np.mean(values[values > 0]))
    return result
