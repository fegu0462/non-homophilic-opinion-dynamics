import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

# matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve
from scipy.linalg import eigvals
from scipy.interpolate import PchipInterpolator
try:
    from tqdm import tqdm
except ImportError:
    # tqdm is optional: the calculation is unchanged when no progress-bar
    # package is installed (for example in the paper's Python 3.9 kernel).
    def tqdm(iterable, desc="Progress", unit="step", **_kwargs):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        if not total:
            yield from iterable
            return
        report_every = max(1, total // 10)
        for index, value in enumerate(iterable, start=1):
            if index == 1 or index % report_every == 0 or index == total:
                print(f"{desc}: {index}/{total} {unit}")
            yield value
from itertools import product
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": ["STIXGeneral", "Arial", "Helvetica", "sans-serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ==========================================
# 1. 参数配置区：4 群体 mean-field 系统
# ==========================================
fs = float(os.environ.get("MF4_FS", "0.5"))
k2 = float(os.environ.get("MF4_K2", "4.0"))
dc = float(os.environ.get("MF4_DC", "0.5"))

alpha3_fixed = float(os.environ.get("MF4_ALPHA3", "0.0"))
alpha2_min = float(os.environ.get("MF4_ALPHA2_MIN", "0.0"))
alpha2_max = float(os.environ.get("MF4_ALPHA2_MAX", "0.7"))
alpha2_step = float(os.environ.get("MF4_ALPHA2_STEP", "0.0005"))
alpha2_values = np.arange(
    alpha2_min,
    alpha2_max + 0.5 * alpha2_step,
    alpha2_step,
)

# 4 个群体：
# mu1: stubborn +1
# mu2: stubborn -1
# mu3: non-stubborn +0.5
# mu4: non-stubborn -0.5
N_total = 4

p_vec = np.array([
    fs * 0.5,
    fs * 0.5,
    (1.0 - fs) * 0.5,
    (1.0 - fs) * 0.5,
])

sigma = np.array([
    0.8,
    0.8,
    0.1,
    0.1,
])

mu_0 = np.array([
    1.0,
    -1.0,
    0.5,
    -0.5,
])


# ==========================================
# 2. 核心方程与雅可比矩阵
# ==========================================
def F_system(mu, alpha1, alpha2, alpha3):
    # 这个函数会被延拓和补根反复调用。向量化与原双重循环逐元素等价，
    # 但能显著降低总运行时间。
    mu = np.asarray(mu, dtype=float)
    distance = np.abs(mu[:, None] - mu[None, :])
    exp_arg = np.clip(k2 * (distance - dc), -100.0, 100.0)
    C1 = (1.0 + np.exp(-k2 * dc)) / (1.0 + np.exp(exp_arg))
    C2 = np.abs(mu)[None, :]
    C3 = np.abs(mu)[:, None]
    Pi = alpha1 * C1 + alpha2 * C2 + alpha3 * C3

    denominator = Pi @ p_vec
    safe_denominator = np.where(denominator > 1e-12, denominator, 1.0)
    W = (p_vec[None, :] * Pi) / safe_denominator[:, None]
    W = np.where(denominator[:, None] > 1e-12, W, 0.0)

    G_mu = sigma * mu_0 + (1.0 - sigma) * (W @ mu)

    return G_mu - mu


def get_jacobian(mu_star, alpha1, alpha2, alpha3, eps=1e-5):
    n = len(mu_star)
    J = np.zeros((n, n))

    for j in range(n):
        mu_p = mu_star.copy()
        mu_m = mu_star.copy()

        mu_p[j] += eps
        mu_m[j] -= eps

        J[:, j] = (
            F_system(mu_p, alpha1, alpha2, alpha3)
            - F_system(mu_m, alpha1, alpha2, alpha3)
        ) / (2.0 * eps)

    return J


# ==========================================
# 3. 固定点搜索
# ==========================================
stable_mu3_pts = []
stable_mu4_pts = []
unstable_mu3_pts = []
unstable_mu4_pts = []

stable_fill_regions = []
scan_records = []
near_zero_stability_records = []

STATE_TOL = 1e-5
SEPARATION_TOL = 0.2

COLOR_BIMODAL = "#f6b6b6"
COLOR_BIASED_UNIMODAL = "#fff2a8"
COLOR_NEUTRAL_UNIMODAL = "#d7ecff"

# ==========================================
# 3.1 分叉 Taylor 诊断参数
# ==========================================
PAIR_DISTANCE_CUTOFF = 0.16
PLUS_ONE_GAP_CUTOFF = 0.02
EVENT_ALPHA2_CLUSTER_WIDTH = 0.006
EVENT_STATE_CLUSTER_WIDTH = 0.08
NONSMOOTH_TOL = 0.02
TAYLOR_H = 5e-4
PARAM_H = 1e-4
COEF_REL_TOL = 0.08
MAX_PRINT_EVENTS = 30

# ==========================================
# 3.2 0 附近稳定性判断参数
# ==========================================
ZERO_STABILITY_TOL = 1e-4
ONE_SIDED_ZERO_TOL = 1e-5
SAFE_JAC_EPS = 1e-5
MAX_PRINT_NEAR_ZERO_ROOTS = 40

# ==========================================
# 3.3 低成本完整分支恢复参数
# ==========================================
# 核心方法不是在每个 alpha2 上重复大量随机初值，而是：
#   1) 用前两个网格点做割线预测；
#   2) 在接近奇异的根上沿最小奇异向量做少量 +/- 扰动；
#   3) 从 alpha2 两端各扫描一次并合并结果。
# 这样可以补回陡峭的不稳定分支，同时比每点使用全部随机初值更省时。
ROOT_RESIDUAL_TOL = 1e-8
ROOT_DUPLICATE_TOL = 2e-6
ROOT_PHYSICAL_BOUND = 1.2
FSOLVE_XTOL = 1e-10
FSOLVE_MAXFEV = 1800

STATIC_RESEED_EVERY = 50
ANCHOR_RANDOM_GUESSES = 8
BRANCH_MATCH_TOL = 0.30

CRITICAL_MIN_SINGULAR_TOL = 3e-2
CRITICAL_PLUS_ONE_TOL = 3e-2
CRITICAL_PERTURB_AMPLITUDES = (0.004, 0.012, 0.035, 0.080)
MAX_CRITICAL_CLOSURE_PASSES = 2


def classify_stable_root(root):
    mu3 = root[2]
    mu4 = root[3]

    mu_sum = mu3 + mu4
    separation = abs(mu3 - mu4)

    if abs(mu_sum) <= STATE_TOL and separation > SEPARATION_TOL:
        return "bimodal"

    # A biased state is identified by broken positive/negative symmetry.
    # Close to the pitchfork, mu3 and mu4 can still have opposite signs, so
    # requiring mu3 * mu4 > 0 incorrectly leaves this region unclassified.
    if abs(mu_sum) > STATE_TOL:
        return "biased_unimodal"

    if abs(mu_sum) <= STATE_TOL and separation <= SEPARATION_TOL:
        return "neutral_unimodal"

    return None


def color_for_state(state):
    if state == "bimodal":
        return COLOR_BIMODAL
    if state == "biased_unimodal":
        return COLOR_BIASED_UNIMODAL
    if state == "neutral_unimodal":
        return COLOR_NEUTRAL_UNIMODAL
    return None


def nonsmooth_distance(root):
    distances = [abs(v) for v in root]

    for i in range(len(root)):
        for j in range(i + 1, len(root)):
            distances.append(abs(root[i] - root[j]))

    return min(distances)


def zero_distance(root):
    return float(np.min(np.abs(root)))


def get_zero_aware_jacobian_candidates(mu_star, alpha1, alpha2, alpha3):
    n = len(mu_star)
    F0 = F_system(mu_star, alpha1, alpha2, alpha3)

    zero_sensitive_columns = [
        j for j in range(n)
        if abs(mu_star[j]) <= ONE_SIDED_ZERO_TOL
    ]

    if len(zero_sensitive_columns) == 0:
        J = np.zeros((n, n))

        for j in range(n):
            eps_j = SAFE_JAC_EPS

            if abs(mu_star[j]) < ZERO_STABILITY_TOL:
                eps_j = max(min(SAFE_JAC_EPS, 0.25 * abs(mu_star[j])), 1e-8)

            mu_p = mu_star.copy()
            mu_m = mu_star.copy()

            mu_p[j] += eps_j
            mu_m[j] -= eps_j

            J[:, j] = (
                F_system(mu_p, alpha1, alpha2, alpha3)
                - F_system(mu_m, alpha1, alpha2, alpha3)
            ) / (2.0 * eps_j)

        return [("safe-central", J)]

    candidates = []

    for signs in product([-1.0, 1.0], repeat=len(zero_sensitive_columns)):
        side_by_col = dict(zip(zero_sensitive_columns, signs))
        J = np.zeros((n, n))

        for j in range(n):
            if j in side_by_col:
                side = side_by_col[j]
                eps_j = SAFE_JAC_EPS

                if side > 0:
                    mu_p = mu_star.copy()
                    mu_p[j] += eps_j
                    J[:, j] = (
                        F_system(mu_p, alpha1, alpha2, alpha3) - F0
                    ) / eps_j
                else:
                    mu_m = mu_star.copy()
                    mu_m[j] -= eps_j
                    J[:, j] = (
                        F0 - F_system(mu_m, alpha1, alpha2, alpha3)
                    ) / eps_j
            else:
                eps_j = SAFE_JAC_EPS

                if abs(mu_star[j]) < ZERO_STABILITY_TOL:
                    eps_j = max(min(SAFE_JAC_EPS, 0.25 * abs(mu_star[j])), 1e-8)

                mu_p = mu_star.copy()
                mu_m = mu_star.copy()

                mu_p[j] += eps_j
                mu_m[j] -= eps_j

                J[:, j] = (
                    F_system(mu_p, alpha1, alpha2, alpha3)
                    - F_system(mu_m, alpha1, alpha2, alpha3)
                ) / (2.0 * eps_j)

        label = "one-sided-zero " + str(
            {col: int(side_by_col[col]) for col in zero_sensitive_columns}
        )
        candidates.append((label, J))

    return candidates


def evaluate_stability_zero_aware(root, alpha1, alpha2, alpha3):
    jacobian_candidates = get_zero_aware_jacobian_candidates(
        root,
        alpha1,
        alpha2,
        alpha3,
    )

    candidate_infos = []

    for label, J_R in jacobian_candidates:
        J_G = J_R + np.eye(N_total)

        eig_G = eigvals(J_G)
        spectral_radius = np.max(np.abs(eig_G))

        eig_closest_to_plus_one = eig_G[np.argmin(np.abs(eig_G - 1.0))]
        eig_closest_to_minus_one = eig_G[np.argmin(np.abs(eig_G + 1.0))]

        candidate_infos.append(
            {
                "label": label,
                "J_R": J_R,
                "J_G": J_G,
                "eig_G": eig_G,
                "spectral_radius": spectral_radius,
                "eig_closest_to_plus_one": eig_closest_to_plus_one,
                "eig_closest_to_minus_one": eig_closest_to_minus_one,
            }
        )

    worst_idx = int(np.argmax([info["spectral_radius"] for info in candidate_infos]))
    worst_info = candidate_infos[worst_idx]

    max_spectral_radius = worst_info["spectral_radius"]
    is_stable = max_spectral_radius < 1.0 - 1e-5

    return {
        "is_stable": is_stable,
        "J_R": worst_info["J_R"],
        "J_G": worst_info["J_G"],
        "eig_G": worst_info["eig_G"],
        "spectral_radius": max_spectral_radius,
        "eig_closest_to_plus_one": worst_info["eig_closest_to_plus_one"],
        "eig_closest_to_minus_one": worst_info["eig_closest_to_minus_one"],
        "worst_label": worst_info["label"],
        "all_candidate_infos": candidate_infos,
        "zero_distance": zero_distance(root),
        "near_zero": zero_distance(root) <= ZERO_STABILITY_TOL,
    }


def F_by_alpha2(mu, alpha2):
    alpha3 = alpha3_fixed
    alpha1 = 1.0 - alpha2 - alpha3
    return F_system(mu, alpha1, alpha2, alpha3)


def J_by_alpha2(mu, alpha2):
    alpha3 = alpha3_fixed
    alpha1 = 1.0 - alpha2 - alpha3
    return get_jacobian(mu, alpha1, alpha2, alpha3)


np.random.seed(42)

base_guesses = [
    np.array([0.9, -0.9, 0.5, -0.5]),
    np.array([0.9, -0.9, 0.0, 0.0]),
    np.array([0.9, -0.9, 0.8, -0.8]),
    np.array([0.9, -0.9, 0.3, -0.3]),
    np.array([0.9, -0.9, 0.8, 0.8]),
    np.array([0.9, -0.9, -0.8, -0.8]),
    np.array([0.9, -0.9, -0.5, 0.5]),
    np.array([1.0, -1.0, 1.0, -1.0]),
    np.array([0.5, -0.5, 0.5, -0.5]),
    np.array([0.0, 0.0, 0.0, 0.0]),
]

perturbed_guesses = []
for guess in base_guesses:
    for _ in range(5):
        noisy_guess = np.clip(
            guess + np.random.normal(0.0, 0.15, N_total),
            -1.0,
            1.0,
        )
        perturbed_guesses.append(noisy_guess)

random_guesses = [
    np.random.uniform(-0.9, 0.9, N_total)
    for _ in range(40)
]

static_guesses = base_guesses + perturbed_guesses + random_guesses


def symmetry_transform(root):
    """Z2 对称变换：一个非对称根被找到后，免费生成其镜像根。"""
    root = np.asarray(root, dtype=float)
    return np.array(
        [-root[1], -root[0], -root[3], -root[2]],
        dtype=float,
    )


def append_unique_root(roots, candidate, tol=ROOT_DUPLICATE_TOL):
    if candidate is None:
        return False
    candidate = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(candidate)):
        return False
    if np.max(np.abs(candidate)) > ROOT_PHYSICAL_BOUND:
        return False
    if any(np.max(np.abs(candidate - root)) <= tol for root in roots):
        return False
    roots.append(candidate)
    return True


def deduplicate_guesses(guesses, tol=1e-9):
    unique = []
    for guess in guesses:
        if guess is None:
            continue
        guess = np.asarray(guess, dtype=float)
        if not np.all(np.isfinite(guess)):
            continue
        if all(np.max(np.abs(guess - old)) > tol for old in unique):
            unique.append(guess)
    return unique


solver_statistics = {
    "fsolve_calls": 0,
    "accepted_roots": 0,
}


def solve_one_root(alpha2, guess):
    """Polish one continuation/perturbation guess at fixed alpha2."""
    alpha3 = alpha3_fixed
    alpha1 = 1.0 - alpha2 - alpha3
    solver_statistics["fsolve_calls"] += 1

    try:
        root, _, _, _ = fsolve(
            lambda mu: F_system(mu, alpha1, alpha2, alpha3),
            np.asarray(guess, dtype=float),
            full_output=True,
            xtol=FSOLVE_XTOL,
            maxfev=FSOLVE_MAXFEV,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None

    residual = float(
        np.max(np.abs(F_system(root, alpha1, alpha2, alpha3)))
    )
    if residual > ROOT_RESIDUAL_TOL:
        return None
    if not np.all(np.isfinite(root)):
        return None
    if np.max(np.abs(root)) > ROOT_PHYSICAL_BOUND:
        return None

    solver_statistics["accepted_roots"] += 1
    return np.asarray(root, dtype=float)


def solve_from_guesses(alpha2, guesses):
    roots = []
    for guess in deduplicate_guesses(guesses):
        append_unique_root(roots, solve_one_root(alpha2, guess))
    return roots


def add_symmetry_partners(roots, alpha2):
    """Validate and append mirror roots without another nonlinear solve."""
    alpha3 = alpha3_fixed
    alpha1 = 1.0 - alpha2 - alpha3
    for root in list(roots):
        mirror = symmetry_transform(root)
        residual = float(
            np.max(np.abs(F_system(mirror, alpha1, alpha2, alpha3)))
        )
        if residual <= ROOT_RESIDUAL_TOL:
            append_unique_root(roots, mirror)


def critical_direction_guesses(roots, alpha2):
    """Near a +1 multiplier, perturb along the full critical state direction."""
    guesses = []
    for root in roots:
        J_F = J_by_alpha2(root, alpha2)
        _, singular_values, Vh = np.linalg.svd(J_F)
        q = Vh[-1, :].real
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-14:
            continue
        q /= q_norm

        J_G = J_F + np.eye(N_total)
        eig_G = eigvals(J_G)
        plus_one_gap = float(np.min(np.abs(eig_G - 1.0)))
        min_singular = float(singular_values[-1])

        if (
            min_singular > CRITICAL_MIN_SINGULAR_TOL
            and plus_one_gap > CRITICAL_PLUS_ONE_TOL
        ):
            continue

        for amplitude in CRITICAL_PERTURB_AMPLITUDES:
            guesses.append(root + amplitude * q)
            guesses.append(root - amplitude * q)
    return guesses


def secant_predictor_guesses(previous_roots, older_roots):
    """Predict steep branches from their last two full-state points."""
    guesses = list(previous_roots)
    if not previous_roots or not older_roots:
        return guesses

    for root in previous_roots:
        nearest = min(
            older_roots,
            key=lambda old: np.linalg.norm(root - old),
        )
        if np.linalg.norm(root - nearest) <= BRANCH_MATCH_TOL:
            guesses.append(2.0 * root - nearest)
    return guesses


def close_critical_side_branches(alpha2, roots):
    """Iteratively discover roots born near a pitchfork without a random sweep."""
    for _ in range(MAX_CRITICAL_CLOSURE_PASSES):
        perturbation_guesses = critical_direction_guesses(roots, alpha2)
        new_roots = solve_from_guesses(alpha2, perturbation_guesses)
        added = False
        for root in new_roots:
            added |= append_unique_root(roots, root)
        add_symmetry_partners(roots, alpha2)
        if not added:
            break


def scan_one_direction(alpha_grid, description):
    roots_by_alpha = {}
    previous_roots = []
    older_roots = []
    previous_alpha = None

    for step_index, alpha2 in enumerate(
        tqdm(alpha_grid, desc=description, unit="step")
    ):
        guesses = secant_predictor_guesses(previous_roots, older_roots)

        # A nearly singular root at the previous layer is the cheapest and most
        # useful seed source for steep stable/unstable pitchfork side branches.
        if previous_alpha is not None:
            guesses.extend(
                critical_direction_guesses(previous_roots, previous_alpha)
            )

        # Full static bank is used only at the endpoint.  Sparse small anchor
        # banks protect against an entirely disconnected branch at low cost.
        if step_index == 0:
            guesses.extend(static_guesses)
        elif step_index % STATIC_RESEED_EVERY == 0:
            guesses.extend(base_guesses)
            guesses.extend(random_guesses[:ANCHOR_RANDOM_GUESSES])

        roots = solve_from_guesses(alpha2, guesses)
        if not roots:
            # Recovery fallback: rare, but do not let one failed layer break all
            # following continuation points.
            roots = solve_from_guesses(alpha2, static_guesses)

        add_symmetry_partners(roots, alpha2)
        close_critical_side_branches(alpha2, roots)
        roots.sort(key=lambda root: tuple(root))
        roots_by_alpha[float(alpha2)] = [root.copy() for root in roots]

        older_roots = [root.copy() for root in previous_roots]
        previous_roots = [root.copy() for root in roots]
        previous_alpha = float(alpha2)

    return roots_by_alpha


print("正在执行 4 群体 mean-field 固定点双向延拓扫描...")
print(f"alpha3 fixed = {alpha3_fixed}")
print(
    f"alpha2 scan  = {alpha2_values[0]:.5f} to "
    f"{alpha2_values[-1]:.5f}, "
    f"step={alpha2_values[1] - alpha2_values[0]:.2e}"
)
print("root recovery = bidirectional continuation + secant predictor + critical-mode perturbations")

forward_roots = scan_one_direction(
    alpha2_values,
    "Forward continuation",
)
backward_roots = scan_one_direction(
    alpha2_values[::-1],
    "Backward continuation",
)

roots_by_alpha = {}
backward_added_count = 0
for alpha2 in alpha2_values:
    key = float(alpha2)
    combined = []
    for root in forward_roots.get(key, []):
        append_unique_root(combined, root)
    before_backward = len(combined)
    for root in backward_roots.get(key, []):
        append_unique_root(combined, root)
    backward_added_count += len(combined) - before_backward
    add_symmetry_partners(combined, alpha2)
    combined.sort(key=lambda root: tuple(root))
    roots_by_alpha[key] = combined

print(
    f"continuation fsolve calls = {solver_statistics['fsolve_calls']}, "
    f"backward-only recovered roots = {backward_added_count}"
)

# Stability classification and plotting data are evaluated only after all
# forward/backward roots have been merged.  Unstable roots are never filtered.
for alpha2 in alpha2_values:
    alpha3 = alpha3_fixed
    alpha1 = 1.0 - alpha2 - alpha3
    found_roots = roots_by_alpha[float(alpha2)]
    root_records = []

    for root in found_roots:
        stability_info = evaluate_stability_zero_aware(
            root,
            alpha1,
            alpha2,
            alpha3,
        )

        eig_G = stability_info["eig_G"]
        spectral_radius = stability_info["spectral_radius"]
        eig_closest_to_plus_one = stability_info["eig_closest_to_plus_one"]
        eig_closest_to_minus_one = stability_info["eig_closest_to_minus_one"]
        is_stable = stability_info["is_stable"]
        mu3 = root[2]
        mu4 = root[3]

        if stability_info["near_zero"]:
            near_zero_stability_records.append(
                {
                    "alpha2": alpha2,
                    "root": root.copy(),
                    "is_stable": is_stable,
                    "zero_distance": stability_info["zero_distance"],
                    "nonsmooth_distance": nonsmooth_distance(root),
                    "spectral_radii": [
                        info["spectral_radius"]
                        for info in stability_info["all_candidate_infos"]
                    ],
                    "worst_label": stability_info["worst_label"],
                }
            )

        root_record = {
            "root": root,
            "is_stable": is_stable,
            "spectral_radius": spectral_radius,
            "eig_closest_to_plus_one": eig_closest_to_plus_one,
            "eig_closest_to_minus_one": eig_closest_to_minus_one,
            "nonsmooth_distance": nonsmooth_distance(root),
            "zero_distance": stability_info["zero_distance"],
            "stability_label": stability_info["worst_label"],
        }
        root_records.append(root_record)

        if is_stable:
            stable_mu3_pts.append((alpha2, mu3))
            stable_mu4_pts.append((alpha2, mu4))
            state = classify_stable_root(root)
            color = color_for_state(state)
            stable_fill_regions.append((alpha2, mu3, mu4, color))
        else:
            unstable_mu3_pts.append((alpha2, mu3))
            unstable_mu4_pts.append((alpha2, mu4))

    scan_records.append(
        {
            "alpha2": alpha2,
            "roots": root_records,
        }
    )


def print_root_count_segments(records):
    """Show where recovered stable/unstable root counts actually change."""
    if not records:
        return

    segments = []
    segment_start = float(records[0]["alpha2"])
    first_stable = sum(root["is_stable"] for root in records[0]["roots"])
    current_counts = (
        int(first_stable),
        int(len(records[0]["roots"]) - first_stable),
    )

    for previous, entry in zip(records[:-1], records[1:]):
        stable_count = sum(root["is_stable"] for root in entry["roots"])
        counts = (int(stable_count), int(len(entry["roots"]) - stable_count))
        if counts == current_counts:
            continue
        segments.append(
            (segment_start, float(previous["alpha2"]), current_counts)
        )
        segment_start = float(entry["alpha2"])
        current_counts = counts

    segments.append(
        (segment_start, float(records[-1]["alpha2"]), current_counts)
    )

    print("\nRecovered root-count segments (S=stable, U=unstable):")
    for left, right, (stable_count, unstable_count) in segments:
        print(
            f"  alpha2={left:.6f}..{right:.6f}: "
            f"S={stable_count}, U={unstable_count}, "
            f"total={stable_count + unstable_count}"
        )


print_root_count_segments(scan_records)


def print_near_zero_stability_summary(records):
    print("\n" + "=" * 110)
    print("Near-zero stability checks")
    print("=" * 110)
    print(f"zero stability tolerance      = {ZERO_STABILITY_TOL}")
    print(f"one-sided zero tolerance      = {ONE_SIDED_ZERO_TOL}")
    print(f"safe Jacobian eps             = {SAFE_JAC_EPS}")
    print(f"near-zero roots checked       = {len(records)}")
    print("-" * 110)

    if len(records) == 0:
        print("No fixed point was close enough to mu_i = 0 to require special stability handling.")
        print("=" * 110)
        return

    for idx, row in enumerate(records[:MAX_PRINT_NEAR_ZERO_ROOTS], start=1):
        radii_str = ", ".join([f"{r:.6e}" for r in row["spectral_radii"]])

        print(f"\nNear-zero root {idx}")
        print("-" * 110)
        print(f"alpha2                  = {row['alpha2']:.6f}")
        print(f"stable after check       = {row['is_stable']}")
        print(f"zero distance            = {row['zero_distance']:.6e}")
        print(f"nonsmooth distance       = {row['nonsmooth_distance']:.6e}")
        print(f"worst Jacobian label     = {row['worst_label']}")
        print(f"spectral radii checked   = [{radii_str}]")
        print(f"root                     = {row['root']}")

    if len(records) > MAX_PRINT_NEAR_ZERO_ROOTS:
        print(
            f"\nOnly first {MAX_PRINT_NEAR_ZERO_ROOTS} near-zero roots printed. "
            f"Total near-zero roots = {len(records)}."
        )

    print("\nInterpretation:")
    print("  Near mu_i = 0, central differences can average the two sides of |mu_i|.")
    print("  This code uses one-sided Jacobian candidates when a root is extremely close to zero.")
    print("  Stability is judged conservatively by the largest spectral radius among those candidates.")
    print("=" * 110)


print_near_zero_stability_summary(near_zero_stability_records)


# ==========================================
# 4. Taylor / normal-form 分叉诊断
# ==========================================
def plus_one_gap(record):
    return abs(record["eig_closest_to_plus_one"] - 1.0)


def collect_pair_candidates(scan_records):
    candidates = []

    for entry in scan_records:
        alpha2 = entry["alpha2"]
        roots = entry["roots"]

        stable_roots = [r for r in roots if r["is_stable"]]
        unstable_roots = [r for r in roots if not r["is_stable"]]

        for stable_root in stable_roots:
            for unstable_root in unstable_roots:
                distance = np.linalg.norm(stable_root["root"] - unstable_root["root"])

                pair_plus_gap = min(
                    plus_one_gap(stable_root),
                    plus_one_gap(unstable_root),
                )

                if distance <= PAIR_DISTANCE_CUTOFF and pair_plus_gap <= PLUS_ONE_GAP_CUTOFF:
                    if plus_one_gap(stable_root) <= plus_one_gap(unstable_root):
                        critical_record = stable_root
                    else:
                        critical_record = unstable_root

                    midpoint = 0.5 * (stable_root["root"] + unstable_root["root"])

                    candidates.append(
                        {
                            "alpha2": alpha2,
                            "distance": distance,
                            "stable": stable_root,
                            "unstable": unstable_root,
                            "critical": critical_record,
                            "midpoint": midpoint,
                            "plus_one_gap": pair_plus_gap,
                            "nonsmooth_midpoint": nonsmooth_distance(midpoint),
                        }
                    )

    return candidates


def cluster_candidates(candidates):
    candidates = sorted(
        candidates,
        key=lambda c: (c["distance"], c["plus_one_gap"]),
    )

    clusters = []

    for candidate in candidates:
        assigned = False

        for cluster in clusters:
            best = cluster["best"]

            alpha2_close = (
                abs(candidate["alpha2"] - best["alpha2"])
                <= EVENT_ALPHA2_CLUSTER_WIDTH
            )
            state_close = (
                np.linalg.norm(candidate["midpoint"] - best["midpoint"])
                <= EVENT_STATE_CLUSTER_WIDTH
            )

            if alpha2_close and state_close:
                cluster["members"].append(candidate)

                candidate_score = candidate["distance"] + candidate["plus_one_gap"]
                best_score = best["distance"] + best["plus_one_gap"]

                if candidate_score < best_score:
                    cluster["best"] = candidate

                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "best": candidate,
                    "members": [candidate],
                }
            )

    clusters = sorted(
        clusters,
        key=lambda c: (c["best"]["alpha2"], c["best"]["midpoint"][2]),
    )

    return clusters


def critical_vectors(A):
    U, singular_values, Vh = np.linalg.svd(A)

    right_vec = Vh[-1, :].real
    left_vec = U[:, -1].real

    norm_right = np.linalg.norm(right_vec)
    norm_left = np.linalg.norm(left_vec)

    if norm_right > 0:
        right_vec = right_vec / norm_right
    if norm_left > 0:
        left_vec = left_vec / norm_left

    dot_val = float(np.dot(left_vec, right_vec))

    if abs(dot_val) < 1e-12:
        return left_vec, right_vec, singular_values[-1], dot_val

    left_vec = left_vec / dot_val

    return left_vec, right_vec, singular_values[-1], float(np.dot(left_vec, right_vec))


def normal_form_coefficients(mu_star, alpha2_star):
    A = J_by_alpha2(mu_star, alpha2_star)
    left_vec, right_vec, min_singular_value, normalization = critical_vectors(A)

    q = right_vec
    p = left_vec

    h = TAYLOR_H
    hp = PARAM_H

    F0 = F_by_alpha2(mu_star, alpha2_star)

    Fp = (
        F_by_alpha2(mu_star, alpha2_star + hp)
        - F_by_alpha2(mu_star, alpha2_star - hp)
    ) / (2.0 * hp)

    Fxx = (
        F_by_alpha2(mu_star + h * q, alpha2_star)
        - 2.0 * F0
        + F_by_alpha2(mu_star - h * q, alpha2_star)
    ) / (h ** 2)

    Fxxx = (
        F_by_alpha2(mu_star + 2.0 * h * q, alpha2_star)
        - 2.0 * F_by_alpha2(mu_star + h * q, alpha2_star)
        + 2.0 * F_by_alpha2(mu_star - h * q, alpha2_star)
        - F_by_alpha2(mu_star - 2.0 * h * q, alpha2_star)
    ) / (2.0 * h ** 3)

    Fxp = (
        F_by_alpha2(mu_star + h * q, alpha2_star + hp)
        - F_by_alpha2(mu_star - h * q, alpha2_star + hp)
        - F_by_alpha2(mu_star + h * q, alpha2_star - hp)
        + F_by_alpha2(mu_star - h * q, alpha2_star - hp)
    ) / (4.0 * h * hp)

    coef_a = float(np.dot(p, Fp))
    coef_b = 0.5 * float(np.dot(p, Fxx))
    coef_c = float(np.dot(p, Fxp))
    coef_d = (1.0 / 6.0) * float(np.dot(p, Fxxx))

    J_G = A + np.eye(N_total)
    eig_G = eigvals(J_G)
    eig_closest_to_plus_one = eig_G[np.argmin(np.abs(eig_G - 1.0))]

    return {
        "A": A,
        "q": q,
        "p": p,
        "min_singular_value": min_singular_value,
        "normalization": normalization,
        "eig_closest_to_plus_one": eig_closest_to_plus_one,
        "coef_a": coef_a,
        "coef_b": coef_b,
        "coef_c": coef_c,
        "coef_d": coef_d,
        "nonsmooth_distance": nonsmooth_distance(mu_star),
    }


def classify_by_normal_form(coeffs):
    a = coeffs["coef_a"]
    b = coeffs["coef_b"]
    c = coeffs["coef_c"]
    d = coeffs["coef_d"]

    ref = max(abs(a), abs(b), abs(c), abs(d), 1e-12)

    a_small = abs(a) <= COEF_REL_TOL * ref
    b_small = abs(b) <= COEF_REL_TOL * ref
    c_small = abs(c) <= COEF_REL_TOL * ref
    d_small = abs(d) <= COEF_REL_TOL * ref

    if coeffs["nonsmooth_distance"] < NONSMOOTH_TOL:
        return (
            "near nonsmooth boundary; smooth Taylor classification is unreliable"
        )

    if (not a_small) and (not b_small):
        return "saddle-node / fold-like normal form: a != 0 and b != 0"

    if a_small and b_small and (not c_small) and (not d_small):
        if c * d < 0:
            return "supercritical pitchfork-like normal form: b ≈ 0, c != 0, d != 0, c*d < 0"
        return "subcritical pitchfork-like normal form: b ≈ 0, c != 0, d != 0, c*d > 0"

    if a_small and (not b_small) and (not c_small):
        return "transcritical-like normal form: a ≈ 0, b != 0, c != 0"

    if (not b_small):
        return "fold-like but parameter unfolding is weak/ambiguous"

    return "ambiguous higher-order normal form"


def print_taylor_diagnostics(scan_records):
    candidates = collect_pair_candidates(scan_records)
    clusters = cluster_candidates(candidates)

    print("\n" + "=" * 110)
    print("Taylor / normal-form bifurcation diagnostics")
    print("=" * 110)
    print(f"pair distance cutoff       = {PAIR_DISTANCE_CUTOFF}")
    print(f"+1 eigenvalue gap cutoff   = {PLUS_ONE_GAP_CUTOFF}")
    print(f"state cluster width        = {EVENT_STATE_CLUSTER_WIDTH}")
    print(f"alpha2 cluster width       = {EVENT_ALPHA2_CLUSTER_WIDTH}")
    print(f"Taylor h                   = {TAYLOR_H}")
    print(f"parameter h                = {PARAM_H}")
    print(f"raw candidate pairs        = {len(candidates)}")
    print(f"distinct candidate events  = {len(clusters)}")
    print("-" * 110)

    if len(clusters) == 0:
        print("No bifurcation candidates found under current thresholds.")
        print("=" * 110)
        return

    for idx, cluster in enumerate(clusters[:MAX_PRINT_EVENTS], start=1):
        candidate = cluster["best"]
        alpha2_c = candidate["alpha2"]

        critical_root = candidate["critical"]["root"]
        coeffs = normal_form_coefficients(critical_root, alpha2_c)
        event_type = classify_by_normal_form(coeffs)

        print(f"\nCandidate {idx}")
        print("-" * 110)
        print(f"alpha2_c                         = {alpha2_c:.6f}")
        print(f"type from Taylor coefficients      = {event_type}")
        print(f"cluster members                   = {len(cluster['members'])}")
        print(f"stable-unstable distance          = {candidate['distance']:.6e}")
        print(f"+1 eigenvalue gap from pair        = {candidate['plus_one_gap']:.6e}")
        print(f"nonsmooth midpoint distance       = {candidate['nonsmooth_midpoint']:.6e}")

        print("\ncritical root used for Taylor expansion:")
        print(f"  root = {critical_root}")
        print(f"  eig closest to +1 = {coeffs['eig_closest_to_plus_one']}")
        print(f"  min singular value of D_mu F = {coeffs['min_singular_value']:.6e}")
        print(f"  nonsmooth distance at root   = {coeffs['nonsmooth_distance']:.6e}")

        print("\nnormal-form coefficients for F(mu, alpha2)=0:")
        print(f"  a = p^T F_alpha                    = {coeffs['coef_a']:.6e}")
        print(f"  b = 1/2 p^T F_mumu(q,q)             = {coeffs['coef_b']:.6e}")
        print(f"  c = p^T F_mu_alpha(q)               = {coeffs['coef_c']:.6e}")
        print(f"  d = 1/6 p^T F_mumumu(q,q,q)         = {coeffs['coef_d']:.6e}")

        print("\ncritical right eigenvector q:")
        print(f"  q = {coeffs['q']}")

        print("\nclosest stable root:")
        print(f"  root = {candidate['stable']['root']}")
        print(f"  spectral radius = {candidate['stable']['spectral_radius']:.6e}")
        print(f"  eig closest to +1 = {candidate['stable']['eig_closest_to_plus_one']}")

        print("\nclosest unstable root:")
        print(f"  root = {candidate['unstable']['root']}")
        print(f"  spectral radius = {candidate['unstable']['spectral_radius']:.6e}")
        print(f"  eig closest to +1 = {candidate['unstable']['eig_closest_to_plus_one']}")

    if len(clusters) > MAX_PRINT_EVENTS:
        print(
            f"\nOnly first {MAX_PRINT_EVENTS} candidate events printed. "
            f"Total events = {len(clusters)}."
        )

    print("\nInterpretation guide:")
    print("  F(mu, alpha2) = G(mu, alpha2) - mu is the fixed-point residual.")
    print("  q is the critical right null direction of D_mu F.")
    print("  p is the corresponding left null direction, normalized so p^T q = 1.")
    print("  saddle-node/fold: usually a != 0 and b != 0.")
    print("  pitchfork-like: usually a ≈ 0, b ≈ 0, c != 0, d != 0, with symmetry.")
    print("  transcritical-like: usually a ≈ 0, b != 0, c != 0.")
    print("  if nonsmooth distance is small, smooth Taylor classification should be treated cautiously.")
    print("=" * 110)


print_taylor_diagnostics(scan_records)


# ==========================================
# 5. 绘图：横坐标 alpha2，纵坐标 mu3/mu4
# ==========================================
TRACK_MATCH_TOL = 0.12
POINT_MARK_INTERVAL = 0.005
SMOOTH_POINTS_PER_INTERVAL = 8


def root_counts(entry):
    stable_count = sum(record["is_stable"] for record in entry["roots"])
    return int(stable_count), int(len(entry["roots"]) - stable_count)


def critical_equations(z):
    """Fixed-point equations plus det(D_mu F)=0 at a static bifurcation."""
    mu = np.asarray(z[:N_total], dtype=float)
    alpha2 = float(z[-1])
    fixed_point_residual = F_by_alpha2(mu, alpha2)
    singularity_residual = np.linalg.det(J_by_alpha2(mu, alpha2))
    return np.concatenate((fixed_point_residual, [singularity_residual]))


def refine_critical_point(left_entry, right_entry):
    """Refine one grid-detected event instead of using a hard-coded alpha2."""
    alpha_left = float(left_entry["alpha2"])
    alpha_right = float(right_entry["alpha2"])
    alpha_mid = 0.5 * (alpha_left + alpha_right)

    # Roots whose map multiplier is closest to +1 are the best initial
    # guesses for saddle-node and pitchfork events of this discrete map.
    candidates = []
    for entry in (left_entry, right_entry):
        for record in entry["roots"]:
            candidates.append(
                (
                    float(abs(record["eig_closest_to_plus_one"] - 1.0)),
                    np.asarray(record["root"], dtype=float),
                )
            )

    for _, root_guess in sorted(candidates, key=lambda item: item[0]):
        z0 = np.concatenate((root_guess, [alpha_mid]))
        try:
            critical, _, ier, _ = fsolve(
                critical_equations,
                z0,
                full_output=True,
                xtol=1e-11,
                maxfev=5000,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

        mu_c = np.asarray(critical[:N_total], dtype=float)
        alpha_c = float(critical[-1])
        residual = critical_equations(critical)

        # Keep the solution tied to the count-change interval that generated
        # it.  A small margin allows Newton refinement beyond a coarse grid.
        grid_width = max(alpha_right - alpha_left, 1e-12)
        inside_local_interval = (
            alpha_left - 4.0 * grid_width
            <= alpha_c
            <= alpha_right + 4.0 * grid_width
        )
        valid = (
            ier == 1
            and inside_local_interval
            and np.max(np.abs(residual[:N_total])) < 1e-8
            and abs(residual[-1]) < 1e-7
            and np.max(np.abs(mu_c)) <= ROOT_PHYSICAL_BOUND
        )
        if valid:
            return alpha_c, mu_c

    # Safe fallback: the event is known only to lie between these two grid
    # layers.  This is still parameter-dependent and carries half a grid step
    # of numerical uncertainty.
    closest_root = min(candidates, key=lambda item: item[0])[1]
    return alpha_mid, closest_root


def detect_bifurcation_points(records):
    """Locate and identify all static bifurcations in the current scan."""
    detected = []

    for left_entry, right_entry in zip(records[:-1], records[1:]):
        left_stable_count, _ = root_counts(left_entry)
        right_stable_count, _ = root_counts(right_entry)

        # Do not draw a vertical guide when only unstable roots are created.
        # A displayed bifurcation must change the number of stable solutions.
        if left_stable_count == right_stable_count:
            continue

        alpha_c, mu_c = refine_critical_point(left_entry, right_entry)

        # A symmetric critical root is not sufficient to identify a
        # pitchfork: the balanced bimodal branch can also undergo a symmetric
        # saddle-node.  Use the local normal-form coefficients instead.
        normal_form = classify_by_normal_form(
            normal_form_coefficients(mu_c, alpha_c)
        )
        if "pitchfork" in normal_form:
            event_type = "pitchfork"
        elif "saddle-node" in normal_form or "fold" in normal_form:
            event_type = "saddle-node"
        else:
            # A count change is definitely present; retain a neutral label if
            # the local Taylor test is inconclusive (for example at a
            # nonsmooth point) instead of assigning the wrong bifurcation.
            event_type = "unclassified"

        if any(
            old_type == event_type and abs(old_alpha - alpha_c) < 1e-4
            for old_type, old_alpha, _ in detected
        ):
            continue
        detected.append((event_type, alpha_c, mu_c))

    detected.sort(key=lambda item: item[1])
    return detected


bifurcation_points = detect_bifurcation_points(scan_records)
print("\nAutomatically detected bifurcation points for the current parameters:")
if bifurcation_points:
    for event_type, alpha_c, mu_c in bifurcation_points:
        print(
            f"  {event_type:12s}: alpha2 = {alpha_c:.10f}, "
            f"critical root = {mu_c}"
        )
else:
    print("  none inside the scanned alpha2 interval")


def build_solution_tracks(records, stable_flag):
    """Join roots at adjacent alpha2 values by nearest-neighbour continuation."""
    tracks = []
    active_track_indices = []

    for entry in records:
        alpha2 = float(entry["alpha2"])
        candidates = [
            record for record in entry["roots"]
            if bool(record["is_stable"]) == bool(stable_flag)
        ]

        possible_matches = []
        for track_index in active_track_indices:
            previous_root = tracks[track_index][-1]["root"]
            for candidate_index, candidate in enumerate(candidates):
                distance = float(
                    np.linalg.norm(previous_root - candidate["root"])
                )
                possible_matches.append(
                    (distance, track_index, candidate_index)
                )

        used_tracks = set()
        used_candidates = set()
        new_active = []
        for distance, track_index, candidate_index in sorted(possible_matches):
            if distance > TRACK_MATCH_TOL:
                continue
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            tracks[track_index].append(
                {
                    "alpha2": alpha2,
                    "root": candidates[candidate_index]["root"].copy(),
                }
            )
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
            new_active.append(track_index)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            tracks.append(
                [{"alpha2": alpha2, "root": candidate["root"].copy()}]
            )
            new_active.append(len(tracks) - 1)

        active_track_indices = new_active

    return [track for track in tracks if len(track) >= 2]


def smooth_track(track):
    """Return shape-preserving smooth curves through one recovered branch."""
    x = np.asarray([point["alpha2"] for point in track])
    roots = np.asarray([point["root"] for point in track])
    order = np.argsort(x)
    x = x[order]
    roots = roots[order]

    if x.size < 3:
        return x, roots[:, 2], roots[:, 3]

    number = (x.size - 1) * SMOOTH_POINTS_PER_INTERVAL + 1
    x_smooth = np.linspace(x[0], x[-1], number)
    mu3_smooth = PchipInterpolator(x, roots[:, 2])(x_smooth)
    mu4_smooth = PchipInterpolator(x, roots[:, 3])(x_smooth)
    return x_smooth, mu3_smooth, mu4_smooth


def split_stable_track_by_color(track):
    """Split only when the physical class, and hence fill color, changes."""
    segments = []
    current_segment = []
    current_color = None

    for point in track:
        color = color_for_state(classify_stable_root(point["root"]))
        if color is None:
            if len(current_segment) >= 2:
                segments.append((current_segment, current_color))
            current_segment = []
            current_color = None
            continue

        if current_color is None or color == current_color:
            current_segment.append(point)
            current_color = color
        else:
            if len(current_segment) >= 2:
                segments.append((current_segment, current_color))
            current_segment = [point]
            current_color = color

    if len(current_segment) >= 2:
        segments.append((current_segment, current_color))
    return segments


stable_tracks = build_solution_tracks(scan_records, stable_flag=True)
unstable_tracks = build_solution_tracks(scan_records, stable_flag=False)

fig, ax = plt.subplots(figsize=(12.8, 7.6), dpi=400)


def extend_fill_segment_to_critical_point(segment):
    """Close the sub-grid gap between a branch segment and a bifurcation."""
    extended = [
        {"alpha2": point["alpha2"], "root": point["root"].copy()}
        for point in segment
    ]
    alpha2_step = float(alpha2_values[1] - alpha2_values[0])
    gap_tolerance = 1.1 * alpha2_step

    for _, alpha_c, mu_c in bifurcation_points:
        left_gap = alpha_c - float(extended[-1]["alpha2"])
        right_gap = float(extended[0]["alpha2"]) - alpha_c

        if 0.0 < left_gap <= gap_tolerance:
            extended.append(
                {"alpha2": float(alpha_c), "root": np.asarray(mu_c).copy()}
            )
        elif 0.0 < right_gap <= gap_tolerance:
            extended.insert(
                0,
                {"alpha2": float(alpha_c), "root": np.asarray(mu_c).copy()},
            )

    extended.sort(key=lambda point: point["alpha2"])
    return extended


# Colored backgrounds are drawn only between mu3 and mu4 belonging to the same
# stable fixed-point branch. Unstable solutions never receive a background.
for track in stable_tracks:
    for segment, fill_color in split_stable_track_by_color(track):
        fill_segment = extend_fill_segment_to_critical_point(segment)
        x_smooth, mu3_smooth, mu4_smooth = smooth_track(fill_segment)
        ax.fill_between(
            x_smooth,
            mu3_smooth,
            mu4_smooth,
            color=fill_color,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )


def draw_tracks(tracks, stable):
    """Draw smooth branch lines and visible markers at sampled solutions."""
    alpha2_step = float(alpha2_values[1] - alpha2_values[0])
    mark_every = max(1, int(round(POINT_MARK_INTERVAL / alpha2_step)))

    if stable:
        styles = (
            (2, "royalblue", "-", "o"),
            (3, "forestgreen", "-", "o"),
        )
        line_width = 2.3
        marker_size = 2.8
        line_alpha = 0.98
    else:
        styles = (
            (2, "#e87c8e", "--", "x"),
            (3, "#f3a55b", "--", "x"),
        )
        line_width = 1.8
        marker_size = 2.7
        line_alpha = 0.82

    for track in tracks:
        x_raw = np.asarray([point["alpha2"] for point in track])
        roots_raw = np.asarray([point["root"] for point in track])
        x_smooth, mu3_smooth, mu4_smooth = smooth_track(track)

        marker_indices = np.arange(0, x_raw.size, mark_every)
        if marker_indices[-1] != x_raw.size - 1:
            marker_indices = np.append(marker_indices, x_raw.size - 1)

        for root_column, color, line_style, marker in styles:
            y_smooth = mu3_smooth if root_column == 2 else mu4_smooth
            ax.plot(
                x_smooth,
                y_smooth,
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=3 if stable else 2,
            )
            ax.plot(
                x_raw[marker_indices],
                roots_raw[marker_indices, root_column],
                linestyle="none",
                marker=marker,
                markersize=marker_size,
                markerfacecolor=color if stable else "none",
                markeredgecolor=color,
                markeredgewidth=1.1,
                alpha=line_alpha,
                zorder=5 if stable else 4,
            )


draw_tracks(unstable_tracks, stable=False)
draw_tracks(stable_tracks, stable=True)

ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.2, alpha=0.7)
event_colors = {
    "saddle-node": "#12A83A",
    "pitchfork": "#E0B000",
    "unclassified": "#555555",
}
for event_type, alpha_c, _ in bifurcation_points:
    ax.axvline(
        alpha_c,
        color=event_colors[event_type],
        linestyle=(0, (8, 5)),
        linewidth=2.5,
        zorder=6,
    )
ax.set_xlabel(r"$\alpha_2$", fontsize=37, labelpad=8)
ax.set_ylabel(
    r"$\mu_{NS^{+}}^*,\ \mu_{NS^{-}}^*$",
    fontsize=36,
    labelpad=12,
)

ax.set_xlim(0.0, 0.30)
ax.set_ylim(-1.05, 1.05)
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.tick_params(axis="both", which="major", labelsize=28, width=1.6, length=8)
ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.32)

legend_handles = [
    Line2D(
        [0], [0], color="royalblue", marker="o", linewidth=2.3,
        markersize=4.5, label=r"Stable $\mu_{NS^{+}}^*$",
    ),
    Line2D(
        [0], [0], color="forestgreen", marker="o", linewidth=2.3,
        markersize=4.5, label=r"Stable $\mu_{NS^{-}}^*$",
    ),
    Line2D(
        [0], [0], color="#e87c8e", linestyle="--", marker="x",
        linewidth=1.8, markersize=4.5, label=r"Unstable $\mu_{NS^{+}}^*$",
    ),
    Line2D(
        [0], [0], color="#f3a55b", linestyle="--", marker="x",
        linewidth=1.8, markersize=4.5, label=r"Unstable $\mu_{NS^{-}}^*$",
    ),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    framealpha=0.96,
    fontsize=16,
    borderpad=0.7,
)

fig.tight_layout()
output_directory = Path(
    os.environ.get("MF4_OUTPUT_DIR", Path(__file__).resolve().parent)
)
output_directory.mkdir(parents=True, exist_ok=True)
default_prefix = f"bifurcation_alpha2_mu3_mu4_fs{int(round(100 * fs)):03d}_print"
output_prefix = os.environ.get("MF4_OUTPUT_PREFIX", default_prefix)
filename_png = output_directory / f"{output_prefix}.png"
filename_pdf = output_directory / f"{output_prefix}.pdf"
filename_svg = output_directory / f"{output_prefix}.svg"
filename_tiff = output_directory / f"{output_prefix}.tiff"
fig.savefig(filename_png, dpi=400, bbox_inches="tight")
fig.savefig(filename_pdf, bbox_inches="tight")
fig.savefig(filename_svg, bbox_inches="tight")
fig.savefig(
    filename_tiff,
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)
print(f"\n完成。PNG 已保存为: {filename_png}")
print(f"PDF 已保存为: {filename_pdf}")
print(f"SVG 已保存为: {filename_svg}")
print(f"TIFF 已保存为: {filename_tiff}")
if os.environ.get("MF4_SHOW_FIGURE", "1").lower() not in {"0", "false", "no"}:
    plt.show()
else:
    plt.close(fig)
