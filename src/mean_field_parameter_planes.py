"""Publication figure: three 42-bin four-state parameter planes.

Panels
------
(a) (alpha2, fs), at dc=0.5 and alpha3=0;
(b) (alpha2, dc), at fs=0.5 and alpha3=0;
(c) (alpha3, fs), at dc=0.5 and alpha2=0.

The coloured backgrounds are obtained from stable 42-bin fixed points.  The
overlaid dynamical boundaries come from the corresponding continuation data.
The dotted grey B-threshold curves are classification boundaries, not
bifurcations.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, UnivariateSpline
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


B_CUTOFF = float(os.environ.get("B_CUTOFF", "0.57"))
B_TAG = f"B{int(round(100 * B_CUTOFF)):03d}"
BIMODALITY_MODE = os.environ.get("BIMODALITY_MODE", "population").strip().lower()
if BIMODALITY_MODE not in {"population", "sample_corrected"}:
    raise ValueError(f"Unsupported BIMODALITY_MODE={BIMODALITY_MODE!r}")
MODE_SUFFIX = "_sample_corrected" if BIMODALITY_MODE == "sample_corrected" else ""
STATE_MODE = os.environ.get("STATE_MODE", "four").strip().lower()
if STATE_MODE not in {"four", "three"}:
    raise ValueError(f"Unsupported STATE_MODE={STATE_MODE!r}")
STATE_TAG = "four_state" if STATE_MODE == "four" else "three_state"
SHOW_BF_PREVIEW = os.environ.get("SHOW_BF_PREVIEW", "0") == "1"
SEPARATE_C_NEAR_LINES = os.environ.get("SEPARATE_C_NEAR_LINES", "0") == "1"
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "").strip().strip("_")
MIRROR_TOL = 2.0e-5
S_TOL = 1.0e-3

SB = "Symmetric bimodal"
AB = "Asymmetric bimodal"
BU = "Biased unimodal"
NU = "Neutral unimodal"

COEX_BU = "BU-bimodal coexistence"
COEX_NU = "NU-bimodal coexistence"
COEX_BIMODAL = "Symmetric + asymmetric bimodal"
COEX_BU_NU = "Biased–neutral unimodal coexistence"

STATE_CLASSES = (
    COEX_BU,
    BU,
    COEX_NU,
    COEX_BIMODAL,
    SB,
    AB,
    NU,
)
STATE_COLORS = {
    COEX_BU: "#ef9a9a",
    BU: "#ffe08a",
    COEX_NU: "#c9b9e8",
    COEX_BIMODAL: "#8475c1",
    SB: "#b39ddb",
    AB: "#58bdb8",
    NU: "#9ec9ee",
}
STATE_LABELS = {
    COEX_BU: "Biased-unimodal–bimodal coexistence",
    BU: "Biased unimodal",
    COEX_NU: "Neutral-unimodal–bimodal coexistence",
    COEX_BIMODAL: "Symmetric + asymmetric bimodal",
    SB: "Symmetric bimodal",
    AB: "Asymmetric bimodal",
    NU: "Neutral unimodal",
}
CODE = {label: index for index, label in enumerate(STATE_CLASSES)}

BIMODAL = "Bimodal"
THREE_STATE_CLASSES = (COEX_BU, BU, COEX_NU, BIMODAL, NU)
THREE_STATE_COLORS = {
    COEX_BU: STATE_COLORS[COEX_BU],
    BU: STATE_COLORS[BU],
    COEX_NU: STATE_COLORS[COEX_NU],
    BIMODAL: "#9f89d0",
    NU: STATE_COLORS[NU],
}
THREE_STATE_LABELS = {
    COEX_BU: "BP–BU coex.",
    BU: "BU",
    COEX_NU: "BP–NU coex.",
    BIMODAL: "BP",
    NU: "NU",
}
THREE_STATE_CODE_MAP = np.asarray([0, 1, 2, 3, 3, 3, 4], dtype=np.int8)

SADDLE_COLOR = "#00c92b"
PITCHFORK_COLOR = "#ffd000"
SUBCRITICAL_COLOR = "#e97817"
CLASSIFICATION_COLOR = "#5b5b5b"
COEX_BU_NU_COLOR = "#2f5d8f"
C_MIN_RIBBON_PT = 3.0
C_CURVE_DISPLAY_OFFSET_PT = 1.1

if np.isclose(B_CUTOFF, 0.56):
    PANEL_A_GRID = DATA_DIR / f"mf_42bin_four_state_B056{MODE_SUFFIX}_composition_grid.csv"
    PANEL_A_BOUNDARIES = DATA_DIR / f"mf_42bin_four_state_B056{MODE_SUFFIX}_continuous_boundaries.csv"
elif np.isclose(B_CUTOFF, 0.57):
    PANEL_A_GRID = DATA_DIR / "mf_42bin_four_state_B057_composition_grid.csv"
    PANEL_A_BOUNDARIES = DATA_DIR / "mf_42bin_four_state_B57_continuous_boundaries.csv"
else:
    raise ValueError("This figure currently supports B_CUTOFF=0.56 or 0.57.")

PANEL_B_CACHE = DATA_DIR / "alpha2_dc_42bin_full_stability_step0p01.pkl"
PANEL_B_CURVES = DATA_DIR / "alpha2_dc_42bin_true_pseudoarclength_curves.csv"
PANEL_B_MACRO = DATA_DIR / "alpha2_dc_42bin_macroscopic_boundaries.csv"
PANEL_B_PITCHFORK = DATA_DIR / "alpha2_dc_42bin_pitchfork_criticality_fine.csv"
PANEL_B_PC = DATA_DIR / "alpha2_dc_42bin_tricritical_points.csv"

PANEL_C_ATTRACTORS = DATA_DIR / f"alpha3_fs_42bin_stable_attractors_fs020_090{MODE_SUFFIX}.csv"
PANEL_C_CURVES = DATA_DIR / f"alpha3_fs_42bin_{B_TAG}_pseudoarclength_boundaries.csv"
PANEL_C_PITCHFORK_TYPE = DATA_DIR / "alpha3_fs_42bin_pitchfork_criticality.csv"
POINT_AUDIT = DATA_DIR / "special_points_42bin_audit.csv"

OUTPUT_STEM = f"figure_42bin_{STATE_TAG}_{B_TAG}{MODE_SUFFIX}_parameter_planes_ABC"
if SHOW_BF_PREVIEW:
    OUTPUT_STEM += "_BF_preview"
if SEPARATE_C_NEAR_LINES:
    OUTPUT_STEM += "_Cseparated"
if OUTPUT_SUFFIX:
    OUTPUT_STEM += f"_{OUTPUT_SUFFIX}"
OUTPUT_PNG = FIGURE_DIR / f"{OUTPUT_STEM}.png"
OUTPUT_PDF = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
OUTPUT_SVG = FIGURE_DIR / f"{OUTPUT_STEM}.svg"
OUTPUT_TIFF = FIGURE_DIR / f"{OUTPUT_STEM}.tiff"
OUTPUT_TRACE = FIGURE_DIR / f"{OUTPUT_STEM}_trace.json"
OUTPUT_CAPTION = FIGURE_DIR / f"{OUTPUT_STEM}_caption.txt"


def finite_sample_corrected_bimodality(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n != 40:
        raise ValueError(f"Expected 40 non-stubborn bins, received {n}.")
    if float(np.std(values)) <= 1.0e-12:
        return 0.0
    g1 = float(skew(values, bias=False))
    g2 = float(kurtosis(values, fisher=True, bias=False))
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = g2 + correction
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float((g1**2 + 1.0) / denominator)


def population_bimodality(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    m2 = float(np.mean(centered**2))
    if m2 <= 1.0e-18:
        return 0.0
    skewness = float(np.mean(centered**3) / m2**1.5)
    kurtosis_value = float(np.mean(centered**4) / m2**2)
    return float((skewness**2 + 1.0) / kurtosis_value)


def sign_reflection(solution: np.ndarray) -> np.ndarray:
    x = np.asarray(solution, dtype=float)
    n = (x.size - 2) // 2
    reflected = np.empty_like(x)
    reflected[0], reflected[1] = -x[1], -x[0]
    reflected[2 : 2 + n] = -x[2 + n :][::-1]
    reflected[2 + n :] = -x[2 : 2 + n][::-1]
    return reflected


bimodality_coefficient = (
    finite_sample_corrected_bimodality
    if BIMODALITY_MODE == "sample_corrected"
    else population_bimodality
)


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def split_visible_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [tuple(pair) for pair in changes.reshape(-1, 2)]


def continuous_resample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_span: float,
    y_span: float,
    count: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an ordered continuation curve by normalized arc length.

    PCHIP preserves the ordering and avoids the overshoot that an unconstrained
    parametric cubic spline can introduce near continuation folds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return x, y
    step = np.hypot(np.diff(x) / x_span, np.diff(y) / y_span)
    keep = np.r_[True, step > 1.0e-12]
    x, y = x[keep], y[keep]
    if x.size < 2:
        return x, y
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x) / x_span, np.diff(y) / y_span))]
    if not np.all(np.diff(arc) > 0.0):
        raise ValueError("Continuation arc length must be strictly increasing.")
    dense_arc = np.linspace(arc[0], arc[-1], max(count, x.size * 3))
    return PchipInterpolator(arc, x)(dense_arc), PchipInterpolator(arc, y)(dense_arc)


def state_change_mask(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    codes: np.ndarray,
    *,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    offset: float = 0.013,
) -> np.ndarray:
    """Detect where a continued critical curve separates two macro classes."""
    x_span = x_domain[1] - x_domain[0]
    y_span = y_domain[1] - y_domain[0]
    xn = (x - x_domain[0]) / x_span
    yn = (y - y_domain[0]) / y_span
    dx = np.gradient(xn)
    dy = np.gradient(yn)
    length = np.hypot(dx, dy)
    length[length < 1.0e-12] = 1.0
    nx = -dy / length
    ny = dx / length
    result = np.zeros(x.size, dtype=bool)
    for index in range(x.size):
        points = []
        for sign in (-1.0, 1.0):
            sample_x = x[index] + sign * offset * nx[index] * x_span
            sample_y = y[index] + sign * offset * ny[index] * y_span
            if not (
                x_domain[0] <= sample_x <= x_domain[1]
                and y_domain[0] <= sample_y <= y_domain[1]
            ):
                points = []
                break
            ix = int(np.argmin(np.abs(x_grid - sample_x)))
            iy = int(np.argmin(np.abs(y_grid - sample_y)))
            points.append(int(codes[iy, ix]))
        result[index] = len(points) == 2 and points[0] != points[1]
    # Suppress one-point flips caused by sampling exactly at a three-region
    # junction while retaining sustained state-changing segments.
    if result.size >= 11:
        score = np.convolve(result.astype(float), np.ones(11) / 11.0, mode="same")
        result = score >= 0.45
    return result


def composition_code(present: set[str]) -> int:
    if present in ({SB, BU}, {SB, AB, BU}, {AB, BU}):
        return CODE[COEX_BU]
    if present == {BU}:
        return CODE[BU]
    if present in ({SB, NU}, {SB, AB, NU}, {AB, NU}):
        return CODE[COEX_NU]
    if present == {SB, AB}:
        return CODE[COEX_BIMODAL]
    if present == {SB}:
        return CODE[SB]
    if present == {AB}:
        return CODE[AB]
    if present == {NU}:
        return CODE[NU]
    return -1


def resolve_unclassified(codes: np.ndarray) -> np.ndarray:
    unresolved = codes < 0
    if not np.any(unresolved):
        return codes
    if np.all(unresolved):
        raise RuntimeError("No classified cells are available for nearest-region filling.")
    nearest = distance_transform_edt(
        unresolved,
        return_distances=False,
        return_indices=True,
    )
    result = codes.copy()
    result[unresolved] = codes[tuple(nearest[:, unresolved])]
    return result


def smooth_categorical_grid(
    x: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    *,
    nx: int = 721,
    ny: int = 721,
    sigma: float = 1.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_dense = np.linspace(*x_domain, nx)
    y_dense = np.linspace(*y_domain, ny)
    xx, yy = np.meshgrid(x_dense, y_dense)
    query = np.column_stack((yy.ravel(), xx.ravel()))
    fields = []
    for code in range(len(STATE_CLASSES)):
        mask = codes == code
        if not np.any(mask):
            fields.append(np.full(xx.shape, -1.0e6))
            continue
        signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
        signed = gaussian_filter(signed, sigma=sigma, mode="nearest")
        interpolator = RegularGridInterpolator(
            (y, x),
            signed,
            method="linear",
            bounds_error=False,
            fill_value=-1.0e6,
        )
        fields.append(interpolator(query).reshape(xx.shape))
    dense = np.argmax(np.stack(fields), axis=0).astype(np.int8)
    return x_dense, y_dense, dense


def display_code_grid(codes: np.ndarray) -> np.ndarray:
    """Map the four-state taxonomy onto the requested display taxonomy."""
    codes = np.asarray(codes, dtype=np.int8)
    if STATE_MODE == "four":
        return codes
    if np.any((codes < 0) | (codes >= len(THREE_STATE_CODE_MAP))):
        raise ValueError("State-code grid contains an out-of-range value.")
    return THREE_STATE_CODE_MAP[codes]


def draw_background(ax, x: np.ndarray, y: np.ndarray, codes: np.ndarray) -> None:
    shown = display_code_grid(codes)
    if STATE_MODE == "four":
        classes = STATE_CLASSES
        colors = STATE_COLORS
    else:
        classes = THREE_STATE_CLASSES
        colors = THREE_STATE_COLORS
    cmap = ListedColormap([colors[label] for label in classes])
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, cmap.N)
    ax.pcolormesh(
        x,
        y,
        shown,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )


def load_special_points(panel: str) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    for row in read_csv(POINT_AUDIT):
        if row["panel"] != panel:
            continue
        points.append(
            {
                "label": row["label"],
                "type": row["type"],
                "marker": row["marker"],
                "x": float(row["x"]),
                "y": float(row["y"]),
            }
        )
    return points


POINT_LABEL_STYLES = {
    ("a", "SNP"): {"offset": (0.010, -0.025), "ha": "left", "va": "top"},
    ("a", "TP"): {"offset": (-0.008, -0.050), "ha": "center", "va": "top"},
    ("b", "PC1"): {"offset": (0.012, -0.024), "ha": "left", "va": "top"},
    ("b", "PC2"): {"offset": (0.014, -0.070), "ha": "left", "va": "top"},
    ("c", "TP"): {"offset": (0.018, -0.024), "ha": "left", "va": "top"},
    ("c", "SNP"): {"offset": (-0.018, 0.022), "ha": "right", "va": "bottom"},
}


def draw_special_points(ax, panel: str) -> None:
    """Draw verified codimension-two points and qualified PC markers."""
    marker_map = {"circle": "o", "diamond": "D", "triangle": "^"}
    for point in load_special_points(panel):
        label = str(point["label"])
        marker = marker_map[str(point["marker"])]
        x_value, y_value = float(point["x"]), float(point["y"])
        ax.plot(
            x_value,
            y_value,
            marker=marker,
            markersize=5.5 if marker == "D" else 5.2,
            markerfacecolor="white",
            markeredgecolor="0.15",
            markeredgewidth=1.05,
            zorder=12,
        )
        style = POINT_LABEL_STYLES[(panel, label)]
        dx, dy = style["offset"]
        display_label = (
            rf"$\mathrm{{PC}}_{{{label[-1]}}}$"
            if label in {"PC1", "PC2"}
            else label
        )
        ax.text(
            x_value + dx,
            y_value + dy,
            display_label,
            fontsize=7.4,
            fontweight="bold",
            ha=style["ha"],
            va=style["va"],
            zorder=13,
        )

    if SHOW_BF_PREVIEW and panel == "c":
        # Preview-only marker: the asymmetric fold reaches the nonsmooth
        # switching manifold x_22=0 here.  It is not a pitchfork/fold
        # intersection and is therefore kept separate from TP/SNP markers.
        bf_x, bf_y = 0.5146826465054444, 0.7313097798402317
        ax.plot(
            bf_x,
            bf_y,
            marker="s",
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor="0.15",
            markeredgewidth=1.05,
            zorder=14,
        )
        ax.text(
            bf_x - 0.018,
            bf_y + 0.027,
            "BF",
            fontsize=7.4,
            fontweight="bold",
            ha="right",
            va="bottom",
            zorder=15,
        )


def make_xy_interpolator(rows: list[dict[str, str]], x_key: str, y_key: str):
    x = np.asarray([float(row[x_key]) for row in rows], dtype=float)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    order = np.argsort(y)
    x, y = x[order], y[order]
    rounded = np.round(y, 11)
    unique_y = np.unique(rounded)
    unique_x = np.asarray([np.mean(x[rounded == value]) for value in unique_y])
    return {
        "x": unique_x,
        "y": unique_y,
        "f": PchipInterpolator(unique_y, unique_x, extrapolate=False),
    }


def evaluate_curve(curve, value: float) -> float | None:
    if curve["y"][0] <= value <= curve["y"][-1]:
        answer = float(curve["f"](value))
        if np.isfinite(answer):
            return answer
    return None


def outlined_dashed(
    ax,
    x,
    y,
    color: str,
    *,
    linewidth: float = 2.6,
    dashes=(4.2, 3.4),
    dash_offset: float = 0.0,
    capstyle: str = "butt",
    outline_extra: float = 1.0,
    zorder: float = 7,
    x_offset_pt: float = 0.0,
) -> None:
    transform = ax.transData
    if x_offset_pt:
        transform = transform + mtransforms.ScaledTranslation(
            x_offset_pt / 72.0,
            0.0,
            ax.figure.dpi_scale_trans,
        )
    line, = ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=(dash_offset, dashes),
        zorder=zorder,
        transform=transform,
    )
    line.set_dash_capstyle(capstyle)
    line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=linewidth + outline_extra,
                foreground="0.22",
                alpha=0.48,
            ),
            path_effects.Normal(),
        ]
    )


def load_panel_a():
    rows = read_csv(PANEL_A_GRID)
    alpha = np.asarray(sorted({float(row["alpha2"]) for row in rows}))
    fs = np.asarray(sorted({float(row["fs"]) for row in rows}))
    alpha_index = {round(value, 12): index for index, value in enumerate(alpha)}
    fs_index = {round(value, 12): index for index, value in enumerate(fs)}
    codes = np.full((fs.size, alpha.size), -1, dtype=np.int8)
    for row in rows:
        present = set(filter(None, row["composition"].split(" + ")))
        codes[
            fs_index[round(float(row["fs"]), 12)],
            alpha_index[round(float(row["alpha2"]), 12)],
        ] = composition_code(present)
    codes = resolve_unclassified(codes)

    boundary_rows = read_csv(PANEL_A_BOUNDARIES)
    grouped = defaultdict(list)
    for row in boundary_rows:
        grouped[row["boundary"]].append(row)
    curves = {
        name: make_xy_interpolator(values, "alpha2", "fs")
        for name, values in grouped.items()
    }

    x_dense, y_dense, dense = smooth_categorical_grid(
        alpha,
        fs,
        codes,
        (0.0, 0.30),
        (0.20, 0.90),
    )

    lower = curves["Saddle-node: yellow left boundary"]
    supercritical = curves["Supercritical pitchfork: yellow right boundary"]
    subcritical = curves["Subcritical pitchfork: yellow upper-left boundary"]
    bu_fold = curves["Saddle-node: biased-unimodal pair disappears"]
    ab_fold = curves["Saddle-node: last asymmetric-bimodal pair disappears"]
    b_curve = curves[f"Classification only: symmetric B={B_CUTOFF:.2f}"]
    tp_fs = min(bu_fold["y"])
    tp_alpha = evaluate_curve(bu_fold, tp_fs)

    for row_index, fs_value in enumerate(y_dense):
        if fs_value < tp_fs:
            right = evaluate_curve(supercritical, fs_value)
            if right is None:
                continue
            if fs_value <= max(lower["y"]):
                left = evaluate_curve(lower, fs_value)
            else:
                left = evaluate_curve(subcritical, fs_value)
            if left is None:
                left = 0.0
            dense[row_index, x_dense < left] = CODE[COEX_BU]
            dense[
                row_index,
                (x_dense >= left) & (x_dense < right),
            ] = CODE[BU]

            b_value = evaluate_curve(b_curve, fs_value)
            if b_value is not None and b_value > right:
                dense[
                    row_index,
                    (x_dense >= right) & (x_dense < b_value),
                ] = CODE[SB]
                dense[row_index, x_dense >= b_value] = CODE[NU]
            else:
                dense[row_index, x_dense >= right] = CODE[NU]
            continue

        bu_alpha = evaluate_curve(bu_fold, fs_value)
        ab_alpha = evaluate_curve(ab_fold, fs_value)
        b_alpha = evaluate_curve(b_curve, fs_value)
        if bu_alpha is None or ab_alpha is None:
            continue
        dense[row_index, x_dense < bu_alpha] = CODE[COEX_BU]
        dense[
            row_index,
            (x_dense >= bu_alpha) & (x_dense < ab_alpha),
        ] = CODE[COEX_BIMODAL]
        if b_alpha is None:
            dense[row_index, x_dense >= ab_alpha] = CODE[SB]
        else:
            dense[
                row_index,
                (x_dense >= ab_alpha) & (x_dense < b_alpha),
            ] = CODE[SB]
            dense[row_index, x_dense >= b_alpha] = CODE[NU]
    return x_dense, y_dense, dense, curves, float(tp_alpha), float(tp_fs)


def draw_panel_a(ax) -> None:
    x_dense, y_dense, dense, curves, tp_alpha, tp_fs = load_panel_a()
    draw_background(ax, x_dense, y_dense, dense)
    styles = {
        "Saddle-node: yellow left boundary": SADDLE_COLOR,
        "Saddle-node: biased-unimodal pair disappears": SADDLE_COLOR,
        "Saddle-node: last asymmetric-bimodal pair disappears": SADDLE_COLOR,
        "Supercritical pitchfork: yellow right boundary": PITCHFORK_COLOR,
        "Subcritical pitchfork: yellow upper-left boundary": SUBCRITICAL_COLOR,
    }
    saddle_junction = 0.5 * (
        float(np.max(curves["Saddle-node: yellow left boundary"]["y"]))
        + float(np.min(curves["Subcritical pitchfork: yellow upper-left boundary"]["y"]))
    )
    for name, color in styles.items():
        curve = curves[name]
        visible = (
            (curve["x"] >= 0.0)
            & (curve["x"] <= 0.30)
            & (curve["y"] >= 0.20)
            & (curve["y"] <= 0.90)
        )
        # Leave a visible gap at the codimension-two hand-off so the
        # green saddle-node and orange subcritical-pitchfork strokes remain
        # distinct despite their finite linewidth.  The two continued loci
        # meet at the endpoint; the gap is a drawing convention only.
        if name == "Saddle-node: yellow left boundary":
            visible &= curve["y"] <= saddle_junction - 0.006
        elif name == "Subcritical pitchfork: yellow upper-left boundary":
            visible &= curve["y"] >= saddle_junction + 0.006
        for start, stop in split_visible_runs(visible):
            if stop - start >= 2:
                outlined_dashed(
                    ax,
                    curve["x"][start:stop],
                    curve["y"][start:stop],
                    color,
                )

    b_curve = curves[f"Classification only: symmetric B={B_CUTOFF:.2f}"]
    supercritical = curves["Supercritical pitchfork: yellow right boundary"]
    exposed = []
    for alpha2, fs_value in zip(b_curve["x"], b_curve["y"]):
        right = evaluate_curve(supercritical, float(fs_value))
        exposed.append(
            0.0 <= alpha2 <= 0.30
            and 0.20 <= fs_value <= 0.90
            and (right is None or alpha2 >= right + 0.003)
        )
    exposed = np.asarray(exposed, dtype=bool)
    for start, stop in split_visible_runs(exposed):
        if stop - start >= 2:
            ax.plot(
                b_curve["x"][start:stop],
                b_curve["y"][start:stop],
                color=CLASSIFICATION_COLOR,
                linewidth=1.7,
                linestyle=(0, (2.0, 2.0)),
                zorder=8,
            )

    # Point markers are read from the independent numerical audit rather than
    # inferred from the plotted line geometry.
    draw_special_points(ax, "a")

    slice_y = 0.50
    marker_x = [0.03, 0.10, 0.30]
    ax.plot([0.0, 0.30], [slice_y, slice_y], color="0.20", linestyle=":", linewidth=1.15, zorder=9)
    ax.plot(
        marker_x,
        [slice_y] * 3,
        linestyle="none",
        marker="o",
        markersize=4.3,
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=0.9,
        clip_on=False,
        zorder=10,
    )
    for value, roman in zip(marker_x, ("I", "II", "III")):
        if roman == "III":
            label_x, horizontal = value - 0.004, "right"
        elif roman == "II":
            # Avoid the major vertical gridline at alpha_2 = 0.10.
            label_x, horizontal = value + 0.008, "left"
        else:
            label_x, horizontal = value, "center"
        # A slightly larger vertical offset prevents the slice and nearby
        # bifurcation curves from crossing the Roman-numeral annotations.
        label_y = slice_y - (0.058 if roman == "II" else 0.048)
        ax.text(label_x, label_y, roman, ha=horizontal, va="top", fontsize=7.0, fontweight="bold")
    ax.text(
        0.004,
        slice_y + 0.065,
        r"$f_s=0.50$",
        fontsize=8.0,
    )


def classify_root(root: np.ndarray, roots: list[np.ndarray]) -> str:
    b_value = bimodality_coefficient(np.asarray(root)[2:])
    reflected = sign_reflection(root)
    distinct = float(np.max(np.abs(reflected - root))) > MIRROR_TOL
    has_partner = distinct and any(
        float(np.max(np.abs(reflected - candidate))) <= MIRROR_TOL
        for candidate in roots
    )
    if b_value > B_CUTOFF:
        return AB if has_partner else SB
    return BU if has_partner else NU


def load_panel_b():
    with PANEL_B_CACHE.open("rb") as stream:
        scan = pickle.load(stream)
    alpha = np.asarray(scan["alpha_values"], dtype=float)
    dc = np.asarray(scan["dc_values"], dtype=float)
    codes = np.full((dc.size, alpha.size), -1, dtype=np.int8)
    root_counts = np.zeros((dc.size, alpha.size), dtype=np.int16)
    for row_index, row in enumerate(scan["solutions"]):
        for column_index, root_values in enumerate(row):
            roots = [np.asarray(root, dtype=float) for root in root_values]
            root_counts[row_index, column_index] = len(roots)
            labels = {classify_root(root, roots) for root in roots}
            codes[row_index, column_index] = composition_code(labels)
    codes = resolve_unclassified(codes)
    x_dense, y_dense, dense = smooth_categorical_grid(
        alpha,
        dc,
        codes,
        (0.0, 0.30),
        (0.0, 1.00),
        nx=721,
        ny=801,
        sigma=1.85,
    )
    return alpha, dc, codes, root_counts, x_dense, y_dense, dense


def panel_b_classification_curves(
    alpha: np.ndarray,
    dc: np.ndarray,
    codes: np.ndarray,
    root_counts: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connect B-only class changes that do not alter the stable-root count."""
    points_by_pair: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for row_index, dc_value in enumerate(dc):
        for column_index in range(alpha.size - 1):
            left = int(codes[row_index, column_index])
            right = int(codes[row_index, column_index + 1])
            if left == right:
                continue
            if root_counts[row_index, column_index] != root_counts[row_index, column_index + 1]:
                continue
            pair = tuple(sorted((left, right)))
            points_by_pair[pair].append(
                (float(dc_value), float(0.5 * (alpha[column_index] + alpha[column_index + 1])))
            )

    connected: list[tuple[np.ndarray, np.ndarray]] = []
    dc_step = float(np.median(np.diff(dc)))
    for points in points_by_pair.values():
        points.sort()
        tracks: list[list[tuple[float, float]]] = []
        for point in points:
            candidates = [
                (abs(track[-1][1] - point[1]), index)
                for index, track in enumerate(tracks)
                if 0.0 < point[0] - track[-1][0] <= 1.6 * dc_step
                and abs(track[-1][1] - point[1]) <= 0.035
            ]
            if candidates:
                tracks[min(candidates)[1]].append(point)
            else:
                tracks.append([point])
        for track in tracks:
            if len(track) < 4:
                continue
            values = np.asarray(track)
            unique_dc, unique_index = np.unique(values[:, 0], return_index=True)
            alpha_values = values[unique_index, 1]
            if unique_dc.size < 4:
                continue
            dense_dc = np.linspace(unique_dc[0], unique_dc[-1], max(160, 8 * unique_dc.size))
            dense_alpha = PchipInterpolator(unique_dc, alpha_values)(dense_dc)
            connected.append((dense_alpha, dense_dc))
    return connected


def load_grouped_curves(path: Path, group_key: str, order_key: str):
    grouped = defaultdict(list)
    for row in read_csv(path):
        grouped[row[group_key]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row[order_key]))
    return grouped


def smooth_macro_points(points: list[dict[str, str]], boundary: str):
    dc = np.asarray([float(point["dc"]) for point in points], dtype=float)
    alpha = np.asarray([float(point["alpha2"]) for point in points], dtype=float)
    order = np.argsort(dc)
    dc, alpha = dc[order], alpha[order]
    unique_dc, unique_index = np.unique(dc, return_index=True)
    dc, alpha = unique_dc, alpha[unique_index]
    if boundary == "bimodal_loss" and alpha.size >= 7:
        alpha = UnivariateSpline(dc, alpha, k=3, s=4.0e-3)(dc)
    elif alpha.size >= 7:
        window = min(11, alpha.size if alpha.size % 2 else alpha.size - 1)
        alpha = savgol_filter(alpha, window_length=window, polyorder=3, mode="interp")
    dense_dc = np.linspace(dc.min(), dc.max(), max(360, dc.size * 8))
    dense_alpha = np.interp(dense_dc, dc, alpha)
    return dense_alpha, dense_dc


def draw_panel_b(ax) -> None:
    alpha, dc, codes, root_counts, x_dense, y_dense, dense = load_panel_b()

    macro = defaultdict(list)
    for row in read_csv(PANEL_B_MACRO):
        macro[row["boundary"]].append(row)
    green_x, green_y = smooth_macro_points(macro["bimodal_loss"], "bimodal_loss")
    green_x, green_y = continuous_resample(
        green_x, green_y, x_span=0.30, y_span=1.00, count=900
    )
    pitch_x, pitch_y = smooth_macro_points(macro["biased_loss"], "biased_loss")
    pitch_x, pitch_y = continuous_resample(
        pitch_x, pitch_y, x_span=0.30, y_span=1.00, count=1100
    )
    criticality = read_csv(PANEL_B_PITCHFORK)
    reference_dc = np.asarray([float(row["dc"]) for row in criticality])
    reference_cubic = np.asarray([float(row["normal_form_cubic"]) for row in criticality])
    subcritical = np.interp(pitch_y, reference_dc, reference_cubic) > 0.0

    # Replace the coarse categorical red--yellow edge by the continuously
    # continued saddle-node and pitchfork loci.  Below the lower
    # criticality-change point the local four-state topology is retained from
    # the full fixed-point scan; above it, the coexistence and biased-unimodal
    # regions are bounded exactly by the two continued curves.
    pc_rows = read_csv(PANEL_B_PC)
    lower_pc_dc = min(float(row["dc"]) for row in pc_rows)
    for row_index, dc_value in enumerate(y_dense):
        if not (
            lower_pc_dc <= dc_value <= float(np.max(green_y))
            and float(np.min(green_y)) <= dc_value
        ):
            continue
        green_alpha = float(np.interp(dc_value, green_y, green_x))
        if not (0.0 <= green_alpha <= 0.30):
            continue
        dense[row_index, x_dense < green_alpha] = CODE[COEX_BU]
        if float(np.min(pitch_y)) <= dc_value <= float(np.max(pitch_y)):
            pitch_alpha = float(np.interp(dc_value, pitch_y, pitch_x))
            if pitch_alpha > green_alpha:
                dense[
                    row_index,
                    (x_dense >= green_alpha) & (x_dense < pitch_alpha),
                ] = CODE[BU]

    draw_background(ax, x_dense, y_dense, dense)
    outlined_dashed(ax, green_x, green_y, SADDLE_COLOR, linewidth=2.6, zorder=7)

    start = 0
    for index in range(1, len(pitch_x) + 1):
        if index == len(pitch_x) or subcritical[index] != subcritical[start]:
            stop = min(index + 1, len(pitch_x))
            color = SUBCRITICAL_COLOR if subcritical[start] else PITCHFORK_COLOR
            outlined_dashed(
                ax,
                pitch_x[start:stop],
                pitch_y[start:stop],
                color,
                linewidth=2.6,
                zorder=7.2,
            )
            start = index

    for class_x, class_y in panel_b_classification_curves(
        alpha, dc, display_code_grid(codes), root_counts
    ):
        visible = (
            (class_x >= 0.0)
            & (class_x <= 0.30)
            & (class_y >= 0.0)
            & (class_y <= 1.00)
        )
        for start, stop in split_visible_runs(visible):
            if stop - start >= 2:
                ax.plot(
                    class_x[start:stop],
                    class_y[start:stop],
                    color=CLASSIFICATION_COLOR,
                    linewidth=1.45,
                    linestyle=(0, (2.0, 2.0)),
                    zorder=8,
                )
    draw_special_points(ax, "b")


def load_panel_c():
    grouped = defaultdict(dict)
    for row in read_csv(PANEL_C_ATTRACTORS):
        key = (round(float(row["alpha3"]), 12), round(float(row["fs"]), 12))
        grouped[key].setdefault(
            int(row["attractor"]),
            (
                float(row["B"]),
                float(row["abs_S"]),
                row.get("state", ""),
            ),
        )
    alpha = np.round(np.arange(0.0, 0.7001, 0.01), 12)
    fs = np.round(np.arange(0.20, 0.9001, 0.01), 12)
    codes = np.full((fs.size, alpha.size), -1, dtype=np.int8)
    root_counts = np.zeros((fs.size, alpha.size), dtype=np.int16)
    b_self = np.full((fs.size, alpha.size), np.nan, dtype=float)
    b_asymmetric = np.full((fs.size, alpha.size), np.nan, dtype=float)
    for row_index, fs_value in enumerate(fs):
        for column_index, alpha3 in enumerate(alpha):
            labels = set()
            values = grouped[(float(alpha3), float(fs_value))].values()
            root_counts[row_index, column_index] = len(values)
            self_values = []
            asymmetric_values = []
            for b_value, abs_s, state in values:
                if state:
                    labels.add(state)
                elif b_value > B_CUTOFF:
                    labels.add(AB if abs_s > S_TOL else SB)
                else:
                    labels.add(BU if abs_s > S_TOL else NU)
                classified_state = state or (
                    (AB if abs_s > S_TOL else SB)
                    if b_value > B_CUTOFF
                    else (BU if abs_s > S_TOL else NU)
                )
                if classified_state in {SB, NU}:
                    self_values.append(b_value)
                else:
                    asymmetric_values.append(b_value)
            if self_values:
                b_self[row_index, column_index] = min(
                    self_values, key=lambda value: abs(value - B_CUTOFF)
                )
            if asymmetric_values:
                b_asymmetric[row_index, column_index] = min(
                    asymmetric_values, key=lambda value: abs(value - B_CUTOFF)
                )
            codes[row_index, column_index] = composition_code(labels)
    codes = resolve_unclassified(codes)
    x_dense, y_dense, dense = smooth_categorical_grid(
        alpha,
        fs,
        codes,
        (0.0, 0.70),
        (0.20, 0.90),
        nx=1401,
        ny=1401,
        sigma=2.20,
    )
    curves = defaultdict(list)
    for row in read_csv(PANEL_C_CURVES):
        curves[row["event"]].append(row)
    for values in curves.values():
        values.sort(key=lambda row: int(row["order"]))

    asymmetric_left = np.asarray(
        [
            (float(row["fs"]), float(row["alpha3"]))
            for row in curves["Asymmetric saddle-node"]
            if 0.20 <= float(row["fs"]) <= 0.90
            and 0.0 <= float(row["alpha3"]) <= 0.70
        ]
    )
    asymmetric_right = np.asarray(
        [
            (float(row["fs"]), float(row["alpha3"]))
            for row in curves["Asymmetric saddle-node (right)"]
            if 0.20 <= float(row["fs"]) <= 0.90
            and 0.0 <= float(row["alpha3"]) <= 0.90
        ]
    )
    if asymmetric_left.size and asymmetric_right.size:
        asymmetric_left = asymmetric_left[np.argsort(asymmetric_left[:, 0])]
        left_fs, left_index = np.unique(asymmetric_left[:, 0], return_index=True)
        left_alpha = asymmetric_left[left_index, 1]
        cusp_fs = float(left_fs[0])
        cusp_alpha = float(left_alpha[0])

        # The right fold also contains a lower continuation segment.  Retain
        # only the branch leaving the codimension-two endpoint toward larger
        # alpha3; together with the left fold it encloses the coexistence
        # tongue and removes the artificial horizontal colour discontinuity.
        asymmetric_right = asymmetric_right[
            (asymmetric_right[:, 0] >= cusp_fs - 0.01)
            & (asymmetric_right[:, 1] >= cusp_alpha - 0.01)
        ]
        asymmetric_right = asymmetric_right[np.argsort(asymmetric_right[:, 0])]
        right_fs, right_index = np.unique(asymmetric_right[:, 0], return_index=True)
        right_alpha = asymmetric_right[right_index, 1]

        for row_index, fs_value in enumerate(y_dense):
            if left_fs[0] <= fs_value <= left_fs[-1]:
                left_boundary = float(np.interp(fs_value, left_fs, left_alpha))
                if fs_value <= right_fs[-1]:
                    right_boundary = float(np.interp(fs_value, right_fs, right_alpha))
                else:
                    right_boundary = 0.71
                dense[row_index, x_dense < left_boundary] = CODE[COEX_BU]
                dense[
                    row_index,
                    (x_dense >= left_boundary) & (x_dense < right_boundary),
                ] = CODE[COEX_BIMODAL]
    return (
        alpha,
        fs,
        codes,
        root_counts,
        b_self,
        b_asymmetric,
        x_dense,
        y_dense,
        dense,
        curves,
    )


def draw_panel_c(ax) -> None:
    (
        alpha,
        fs,
        codes,
        root_counts,
        b_self,
        b_asymmetric,
        x_dense,
        y_dense,
        dense,
        curves,
    ) = load_panel_c()

    # Rebuild panel (c)'s colour field directly from the computed stable-
    # attractor grid.  ``load_panel_c`` also returns an older presentation
    # field in which the upper saddle-node endpoint was artificially joined
    # to another boundary; that post-processing created a spurious narrow BP
    # strip between the BP--BU coexistence and BU regions.  The replacement
    # panel keeps the numerical state grid descriptive and lets the overlaid
    # continuation curves carry the bifurcation information.
    x_dense, y_dense, dense = smooth_categorical_grid(
        alpha,
        fs,
        codes,
        (0.0, 0.70),
        (0.20, 0.90),
        nx=1401,
        ny=1401,
        sigma=0.75,
    )
    shown_dense = display_code_grid(dense).copy()
    if STATE_MODE == "three":
        # Keep the corrected panel visually focused on the three macroscopic
        # states.  Above the horizontal B threshold the small secondary
        # subdivisions are merged into BP; below it the genuine, very narrow
        # BU--NU coexistence tongue is drawn explicitly from its orange and
        # green continuation boundaries below.
        classification_y = 0.622
        yy = y_dense[:, None]
        shown_dense[
            (yy > classification_y)
            & (shown_dense != THREE_STATE_CLASSES.index(COEX_BU))
            & (shown_dense != THREE_STATE_CLASSES.index(BU))
        ] = THREE_STATE_CLASSES.index(BIMODAL)
        shown_dense[
            (yy <= classification_y)
            & (shown_dense == THREE_STATE_CLASSES.index(COEX_NU))
        ] = THREE_STATE_CLASSES.index(NU)

        cmap = ListedColormap(
            [THREE_STATE_COLORS[label] for label in THREE_STATE_CLASSES]
        )
        norm = BoundaryNorm(
            np.arange(len(THREE_STATE_CLASSES) + 1) - 0.5,
            cmap.N,
        )
        ax.pcolormesh(
            x_dense,
            y_dense,
            shown_dense,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            rasterized=True,
            zorder=0,
        )
    else:
        draw_background(ax, x_dense, y_dense, dense)
    criticality_rows = read_csv(PANEL_C_PITCHFORK_TYPE)
    pitchfork_cubic = {
        int(row["order"]): float(row["normal_form_cubic"])
        for row in criticality_rows
    }
    stability_margin = 2.0e-6
    dynamic_points: list[np.ndarray] = []
    all_dynamic_points: list[np.ndarray] = []

    pc_points = [
        point
        for point in load_special_points("c")
        if str(point["label"]) in {"TP", "PC"}
        and float(point["y"]) < 0.40
    ]
    pc_x = float(pc_points[0]["x"]) if pc_points else float("nan")
    pc_y = float(pc_points[0]["y"]) if pc_points else float("nan")
    coexistence_geometry: dict[str, np.ndarray] | None = None

    if SEPARATE_C_NEAR_LINES:
        # Above the lower tricritical pitchfork (TP), the subcritical
        # pitchfork and the right asymmetric fold
        # bound a genuine hysteretic coexistence tongue. Fill only the true
        # interval between the numerical continuation curves. A modest centre
        # ribbon and symmetric display offsets keep the sub-line-width tongue
        # visible; the underlying coordinates are unchanged. No fold or
        # coexistence region is drawn below TP.
        pitch_rows = curves["Pitchfork"]
        pitch_x = np.asarray([float(row["alpha3"]) for row in pitch_rows])
        pitch_y = np.asarray([float(row["fs"]) for row in pitch_rows])
        pitch_noncritical = np.asarray(
            [float(row["largest_noncritical_multiplier_modulus"]) for row in pitch_rows]
        )
        pitch_cubic_values = np.asarray(
            [pitchfork_cubic.get(int(row["order"]), np.nan) for row in pitch_rows]
        )
        pitch_mask = (
            (pitch_x >= 0.0)
            & (pitch_x <= 0.70)
            & (pitch_y >= 0.20)
            & (pitch_y <= 0.90)
            & (pitch_noncritical < 1.0 - stability_margin)
            & (pitch_cubic_values > 0.0)
        )

        # Retain the right-hand, monotonically rising subcritical arc from PC
        # to its maximum-fs turning point.  The later return arc belongs to the
        # upper/left part of the pitchfork curve and is not the boundary paired
        # with this right fold.
        if np.isfinite(pc_x) and np.isfinite(pc_y):
            pitch_pc_index = int(
                np.argmin(
                    ((pitch_x - pc_x) / 0.70) ** 2
                    + ((pitch_y - pc_y) / 0.70) ** 2
                )
            )
            pitch_turn_index = pitch_pc_index + int(
                np.argmax(pitch_y[pitch_pc_index:])
            )
            pitch_mask[:pitch_pc_index] = False
            pitch_mask[pitch_turn_index + 1 :] = False

        fold_rows = curves["Asymmetric saddle-node (right)"]
        fold_x = np.asarray([float(row["alpha3"]) for row in fold_rows])
        fold_y = np.asarray([float(row["fs"]) for row in fold_rows])
        fold_noncritical = np.asarray(
            [float(row["largest_noncritical_multiplier_modulus"]) for row in fold_rows]
        )
        fold_mask = (
            (fold_x >= 0.0)
            & (fold_x <= 0.70)
            & (fold_y >= 0.20)
            & (fold_y <= 0.90)
            & (fold_noncritical < 1.0 - stability_margin)
        )
        if np.isfinite(pc_x) and np.isfinite(pc_y):
            fold_pc_index = int(
                np.argmin(
                    ((fold_x - pc_x) / 0.70) ** 2
                    + ((fold_y - pc_y) / 0.70) ** 2
                )
            )
            fold_mask[:fold_pc_index] = False

        px, py = pitch_x[pitch_mask], pitch_y[pitch_mask]
        fx, fy = fold_x[fold_mask], fold_y[fold_mask]
        if px.size >= 2 and fx.size >= 2:
            p_order = np.argsort(py)
            f_order = np.argsort(fy)
            py, px = py[p_order], px[p_order]
            fy, fx = fy[f_order], fx[f_order]
            py_unique, py_index = np.unique(py, return_index=True)
            fy_unique, fy_index = np.unique(fy, return_index=True)
            py, px = py_unique, px[py_index]
            fy, fx = fy_unique, fx[fy_index]
            if not (np.all(np.diff(py) > 0.0) and np.all(np.diff(fy) > 0.0)):
                raise RuntimeError("Panel-c coexistence interpolation requires increasing fs.")
            common_low = max(float(py.min()), float(fy.min()), pc_y, 0.20)
            common_high = min(float(py.max()), float(fy.max()), 0.90)
            if common_high > common_low:
                coexist_y = np.linspace(common_low, common_high, 500)
                coexist_pitch = PchipInterpolator(py, px)(coexist_y)
                coexist_fold = PchipInterpolator(fy, fx)(coexist_y)
                coexist_left = np.minimum(coexist_pitch, coexist_fold)
                coexist_right = np.maximum(coexist_pitch, coexist_fold)
                coexistence_geometry = {
                    "fs": coexist_y,
                    "pitchfork": coexist_pitch,
                    "fold": coexist_fold,
                    "left": coexist_left,
                    "right": coexist_right,
                }

    if coexistence_geometry is not None:
        coexist_y = coexistence_geometry["fs"]
        coexist_left = coexistence_geometry["left"]
        coexist_right = coexistence_geometry["right"]
        coexist_center = 0.5 * (coexist_left + coexist_right)
        ax.fill_betweenx(
            coexist_y,
            coexist_left,
            coexist_right,
            facecolor=COEX_BU_NU_COLOR,
            edgecolor="none",
            alpha=0.90,
            zorder=5.20,
        )
        # The true interval is narrower than a printed line over most of the
        # lower tongue. Add a small centre ribbon for visibility; the later
        # symmetric line offsets affect display only, not numerical geometry.
        narrow = (coexist_right - coexist_left) < 0.008
        for start, stop in split_visible_runs(narrow):
            if stop - start < 2:
                continue
            ax.plot(
                coexist_center[start:stop],
                coexist_y[start:stop],
                color=COEX_BU_NU_COLOR,
                alpha=0.90,
                linewidth=C_MIN_RIBBON_PT,
                solid_capstyle="round",
                zorder=5.25,
            )

    for event in (
        "Saddle-node",
        "Asymmetric saddle-node",
        "Pitchfork",
        # Draw the right asymmetric fold last, but only where it becomes an
        # independently resolved branch above the tricritical pitchfork.
        # Below TP it numerically collapses onto the supercritical
        # pitchfork and is not evidence for a second bifurcation.
        "Asymmetric saddle-node (right)",
    ):
        values = curves[event]
        x = np.asarray([float(row["alpha3"]) for row in values])
        y = np.asarray([float(row["fs"]) for row in values])
        noncritical = np.asarray(
            [float(row["largest_noncritical_multiplier_modulus"]) for row in values]
        )
        physical_visible = (
            (x >= 0.0)
            & (x <= 0.70)
            & (y >= 0.20)
            & (y <= 0.90)
            & (noncritical < 1.0 - stability_margin)
        )

        if event == "Asymmetric saddle-node (right)":
            # Below the tricritical pitchfork this full-state continuation
            # collapses numerically onto the supercritical pitchfork and does
            # not establish a second bifurcation.  The displayed fold therefore
            # starts at TP in both the ordinary and separated renderings.
            if pc_points:
                pc_index = int(
                    np.argmin(
                        ((x - pc_x) / 0.70) ** 2
                        + ((y - pc_y) / 0.70) ** 2
                    )
                )
                physical_visible[:pc_index] = False

        for start, stop in split_visible_runs(physical_visible):
            if stop - start < 2:
                continue
            ref_x, ref_y = continuous_resample(
                x[start:stop],
                y[start:stop],
                x_span=0.70,
                y_span=0.70,
                count=1100,
            )
            all_dynamic_points.append(
                np.column_stack((ref_x / 0.70, (ref_y - 0.20) / 0.70))
            )

        stable_visible = physical_visible.copy()

        if event == "Pitchfork":
            cubic = np.asarray(
                [pitchfork_cubic.get(int(row["order"]), np.nan) for row in values]
            )
            styled_masks = (
                (
                    stable_visible & (cubic < 0.0),
                    PITCHFORK_COLOR,
                    {},
                ),
                (
                    stable_visible & (cubic > 0.0),
                    SUBCRITICAL_COLOR,
                    {
                        "dashes": (3.0, 2.5),
                        "linewidth": 2.35,
                        "outline_extra": 0.75,
                        "zorder": 8,
                        "x_offset_pt": (
                            -C_CURVE_DISPLAY_OFFSET_PT
                            if SEPARATE_C_NEAR_LINES
                            else 0.0
                        ),
                    },
                ),
            )
        elif event == "Asymmetric saddle-node (right)":
            styled_masks = (
                (
                    stable_visible,
                    SADDLE_COLOR,
                    {
                        "dashes": (3.0, 2.5),
                        "dash_offset": 0.0,
                        "linewidth": 1.35,
                        "outline_extra": 0.30,
                        "zorder": 9,
                        "x_offset_pt": (
                            C_CURVE_DISPLAY_OFFSET_PT
                            if SEPARATE_C_NEAR_LINES
                            else 0.0
                        ),
                    },
                ),
            )
        else:
            styled_masks = ((stable_visible, SADDLE_COLOR, {}),)

        for visible, color, line_kwargs in styled_masks:
            for start, stop in split_visible_runs(visible):
                if stop - start < 2:
                    continue
                smooth_x, smooth_y = continuous_resample(
                    x[start:stop],
                    y[start:stop],
                    x_span=0.70,
                    y_span=0.70,
                    count=1100,
                )
                outlined_dashed(
                    ax,
                    smooth_x,
                    smooth_y,
                    color,
                    **line_kwargs,
                )
                dynamic_points.append(
                    np.column_stack((smooth_x / 0.70, (smooth_y - 0.20) / 0.70))
                )

    # Reconstruct corrected-B contours from the newly computed branch
    # observables.  Both orientations are captured (including the nearly
    # horizontal self-symmetric contour near fs≈0.63), unlike row-only edge
    # tracking.  Masked asymmetric regions prevent extrapolation beyond the
    # existence range of the asymmetric branches.
    for field in (b_self, b_asymmetric):
        finite = np.isfinite(field)
        if np.count_nonzero(finite) < 4:
            continue
        numerator = gaussian_filter(np.where(finite, field, 0.0), sigma=0.65)
        support = gaussian_filter(finite.astype(float), sigma=0.65)
        smoothed = np.divide(
            numerator,
            support,
            out=np.full_like(numerator, np.nan),
            where=support > 0.55,
        )
        contour = ax.contour(
            alpha,
            fs,
            np.ma.masked_invalid(smoothed),
            levels=[B_CUTOFF],
            colors=["none"],
            linewidths=0.0,
        )
        segments = contour.allsegs[0]
        contour.remove()
        dynamic = (
            np.vstack(all_dynamic_points)
            if all_dynamic_points
            else np.empty((0, 2))
        )
        for segment in segments:
            if segment.shape[0] < 2:
                continue
            normalized = np.column_stack(
                (segment[:, 0] / 0.70, (segment[:, 1] - 0.20) / 0.70)
            )
            if dynamic.size:
                min_distance = np.full(segment.shape[0], np.inf)
                for start in range(0, segment.shape[0], 250):
                    stop = min(start + 250, segment.shape[0])
                    delta = normalized[start:stop, None, :] - dynamic[None, :, :]
                    min_distance[start:stop] = np.sqrt(
                        np.min(np.sum(delta**2, axis=2), axis=1)
                    )
                exposed = min_distance > 0.032
            else:
                exposed = np.ones(segment.shape[0], dtype=bool)
            exposed &= state_change_mask(
                segment[:, 0],
                segment[:, 1],
                x_dense,
                y_dense,
                shown_dense,
                x_domain=(0.0, 0.70),
                y_domain=(0.20, 0.90),
                offset=0.009,
            )
            for start, stop in split_visible_runs(exposed):
                if stop - start >= 2:
                    ax.plot(
                        segment[start:stop, 0],
                        segment[start:stop, 1],
                        color=CLASSIFICATION_COLOR,
                        linewidth=1.45,
                        linestyle=(0, (2.0, 2.0)),
                        zorder=8,
                    )

    # Reuse the fill's interpolated coordinates, then apply small equal and
    # opposite display offsets so the dark-blue tongue remains visible. The
    # underlying continuation coordinates are unchanged.
    if coexistence_geometry is not None:
        outlined_dashed(
            ax,
            coexistence_geometry["pitchfork"],
            coexistence_geometry["fs"],
            SUBCRITICAL_COLOR,
            linewidth=2.35,
            dashes=(3.0, 2.5),
            outline_extra=0.75,
            zorder=10,
            x_offset_pt=-C_CURVE_DISPLAY_OFFSET_PT,
        )
        outlined_dashed(
            ax,
            coexistence_geometry["fold"],
            coexistence_geometry["fs"],
            SADDLE_COLOR,
            linewidth=1.35,
            dashes=(3.0, 2.5),
            outline_extra=0.30,
            zorder=10,
            x_offset_pt=C_CURVE_DISPLAY_OFFSET_PT,
        )
    draw_special_points(ax, "c")


def draw_panel_c_replacement(ax) -> None:
    """Draw the final corrected panel (c) without the legacy colour rewrite.

    This is a direct panel-sized rendering of the verified standalone panel-c
    figure.  The state background comes only from the stable-attractor grid;
    the BU--NU tongue is then filled between its two continued dynamical
    boundaries and those two boundaries are redrawn on top.
    """
    (
        alpha,
        fs,
        codes,
        _root_counts,
        _b_self,
        b_asymmetric,
        _legacy_x,
        _legacy_y,
        _legacy_dense,
        curves,
    ) = load_panel_c()

    x_dense, y_dense, dense = smooth_categorical_grid(
        alpha,
        fs,
        codes,
        (0.0, 0.70),
        (0.20, 0.90),
        nx=1401,
        ny=1401,
        sigma=0.75,
    )
    shown = display_code_grid(dense).copy()
    classification_y = 0.622
    if STATE_MODE == "three":
        yy = y_dense[:, None]
        shown[
            (yy > classification_y)
            & (shown != THREE_STATE_CLASSES.index(COEX_BU))
            & (shown != THREE_STATE_CLASSES.index(BU))
        ] = THREE_STATE_CLASSES.index(BIMODAL)
        shown[
            (yy <= classification_y)
            & (shown == THREE_STATE_CLASSES.index(COEX_NU))
        ] = THREE_STATE_CLASSES.index(NU)
        classes, colors = THREE_STATE_CLASSES, THREE_STATE_COLORS
    else:
        classes, colors = STATE_CLASSES, STATE_COLORS
    cmap = ListedColormap([colors[label] for label in classes])
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, cmap.N)
    ax.pcolormesh(
        x_dense,
        y_dense,
        shown,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )

    def curve_arrays(event: str):
        rows = curves[event]
        return (
            np.asarray([int(row["order"]) for row in rows]),
            np.asarray([float(row["alpha3"]) for row in rows]),
            np.asarray([float(row["fs"]) for row in rows]),
            np.asarray(
                [float(row["largest_noncritical_multiplier_modulus"]) for row in rows]
            ),
        )

    def x_of_y(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
        order = np.argsort(y)
        yy = y[order]
        xx = x[order]
        yy, keep = np.unique(yy, return_index=True)
        xx = xx[keep]
        if not np.all(np.diff(yy) > 0.0):
            raise RuntimeError("Panel-c interpolation requires increasing f_s.")
        return PchipInterpolator(yy, xx)(query)

    stability_margin = 2.0e-6
    pitchfork_cubic = {
        int(row["order"]): float(row["normal_form_cubic"])
        for row in read_csv(PANEL_C_PITCHFORK_TYPE)
    }
    pitch_order, pitch_x, pitch_y, pitch_noncritical = curve_arrays("Pitchfork")
    pitch_cubic = np.asarray(
        [pitchfork_cubic.get(int(order), np.nan) for order in pitch_order]
    )
    pitch_physical = (
        (pitch_x >= 0.0)
        & (pitch_x <= 0.70)
        & (pitch_y >= 0.20)
        & (pitch_y <= 0.90)
        & (pitch_noncritical < 1.0 - stability_margin)
    )

    _, fold_x, fold_y, fold_noncritical = curve_arrays(
        "Asymmetric saddle-node (right)"
    )
    fold_physical = (
        (fold_x >= 0.0)
        & (fold_x <= 0.70)
        & (fold_y >= 0.20)
        & (fold_y <= 0.90)
        & (fold_noncritical < 1.0 - stability_margin)
    )

    tp = next(
        point
        for point in load_special_points("c")
        if str(point["label"]) == "TP"
    )
    tp_x, tp_y = float(tp["x"]), float(tp["y"])
    pitch_tp = int(
        np.argmin(((pitch_x - tp_x) / 0.70) ** 2 + ((pitch_y - tp_y) / 0.70) ** 2)
    )
    pitch_turn = pitch_tp + int(np.argmax(pitch_y[pitch_tp:]))
    pitch_tongue = pitch_physical & (pitch_cubic > 0.0)
    pitch_tongue[:pitch_tp] = False
    pitch_tongue[pitch_turn + 1 :] = False

    fold_tp = int(
        np.argmin(((fold_x - tp_x) / 0.70) ** 2 + ((fold_y - tp_y) / 0.70) ** 2)
    )
    fold_tongue = fold_physical.copy()
    fold_tongue[:fold_tp] = False

    tongue_pitch_x = pitch_x[pitch_tongue]
    tongue_pitch_y = pitch_y[pitch_tongue]
    tongue_fold_x = fold_x[fold_tongue]
    tongue_fold_y = fold_y[fold_tongue]
    tongue_low = max(
        float(tongue_pitch_y.min()),
        float(tongue_fold_y.min()),
        tp_y,
    )
    tongue_high = min(
        float(tongue_pitch_y.max()),
        float(tongue_fold_y.max()),
        classification_y,
    )
    tongue_y = np.linspace(tongue_low, tongue_high, 650)
    tongue_pitch = x_of_y(tongue_pitch_x, tongue_pitch_y, tongue_y)
    tongue_fold = x_of_y(tongue_fold_x, tongue_fold_y, tongue_y)

    # Preserve the numerical tongue wherever it is resolvable.  Where its
    # width falls below the publication-scale minimum, expand both edges
    # symmetrically.  Both boundary curves are drawn once as plain thin dashed
    # lines on the two displayed edges; no centre ribbon or outlined stroke is
    # added inside the dark-blue region.
    axes_width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    minimum_halfwidth_data = 0.5 * C_MIN_RIBBON_PT * 0.70 / axes_width_pt
    tongue_center_x = 0.5 * (tongue_pitch + tongue_fold)
    tongue_center_y = tongue_y.copy()
    tangent_x = np.gradient(tongue_center_x)
    tangent_y = np.gradient(tongue_center_y)
    tangent_norm = np.hypot(tangent_x, tangent_y)
    normal_x = -tangent_y / tangent_norm
    normal_y = tangent_x / tangent_norm
    # Orient the normal toward the orange/yellow side of the tongue.
    if np.nanmedian((tongue_pitch - tongue_fold) * normal_x) < 0.0:
        normal_x *= -1.0
        normal_y *= -1.0
    physical_halfwidth = 0.5 * np.abs(
        (tongue_pitch - tongue_fold) * normal_x
    )
    display_halfwidth = np.maximum(
        physical_halfwidth, minimum_halfwidth_data
    )
    tongue_pitch_display = tongue_center_x + normal_x * display_halfwidth
    tongue_pitch_display_y = tongue_center_y + normal_y * display_halfwidth
    tongue_fold_display = tongue_center_x - normal_x * display_halfwidth
    tongue_fold_display_y = tongue_center_y - normal_y * display_halfwidth
    ax.fill(
        np.concatenate((tongue_pitch_display, tongue_fold_display[::-1])),
        np.concatenate((tongue_pitch_display_y, tongue_fold_display_y[::-1])),
        facecolor=COEX_BU_NU_COLOR,
        edgecolor="none",
        alpha=0.95,
        zorder=5.20,
    )

    # Stable saddle-node arcs other than the right-hand tongue boundary.
    green_reference = []
    for event in ("Saddle-node", "Asymmetric saddle-node"):
        _, x, y, noncritical = curve_arrays(event)
        visible = (
            (x >= 0.0)
            & (x <= 0.70)
            & (y >= 0.20)
            & (y <= 0.90)
            & (noncritical < 1.0 - stability_margin)
        )
        green_reference.append(np.column_stack((x[visible], y[visible])))
        for start, stop in split_visible_runs(visible):
            if stop - start >= 2:
                sx, sy = continuous_resample(
                    x[start:stop], y[start:stop], x_span=0.70, y_span=0.70
                )
                outlined_dashed(ax, sx, sy, SADDLE_COLOR, linewidth=2.6, zorder=7)

    # Supercritical and subcritical parts of the pitchfork continuation.
    tongue_pitch_overlap = (
        pitch_tongue
        & (pitch_y >= tongue_low)
        & (pitch_y <= tongue_high)
    )
    for visible, color, curve_width in (
        (pitch_physical & (pitch_cubic < 0.0), PITCHFORK_COLOR, 2.6),
        (
            pitch_physical
            & (pitch_cubic > 0.0)
            & ~tongue_pitch_overlap,
            SUBCRITICAL_COLOR,
            1.65,
        ),
    ):
        for start, stop in split_visible_runs(visible):
            if stop - start >= 2:
                sx, sy = continuous_resample(
                    pitch_x[start:stop],
                    pitch_y[start:stop],
                    x_span=0.70,
                    y_span=0.70,
                )
                outlined_dashed(
                    ax, sx, sy, color, linewidth=curve_width, zorder=8
                )

    # Outside the highlighted tongue, retain only the continuation above its
    # upper endpoint.  The lower part is drawn exactly once below as the thin
    # green display edge; drawing the full fold here as well would fill the
    # dash gaps and create a spurious thick green line inside the blue band.
    right_visible = fold_physical & (fold_y > tongue_high + 1.0e-10)
    for start, stop in split_visible_runs(right_visible):
        if stop - start >= 2:
            sx, sy = continuous_resample(
                fold_x[start:stop], fold_y[start:stop], x_span=0.70, y_span=0.70
            )
            outlined_dashed(
                ax,
                sx,
                sy,
                SADDLE_COLOR,
                linewidth=1.35,
                dashes=(3.0, 2.5),
                outline_extra=0.30,
                zorder=9,
            )

    # Draw the two displayed edges once with the reference styling: a modest
    # orange dashed edge on the left and a slightly thinner green dashed edge
    # on the right, with the blue fill visible between them.
    for edge_x, edge_y, edge_color, edge_width in (
        (
            tongue_pitch_display,
            tongue_pitch_display_y,
            SUBCRITICAL_COLOR,
            1.20,
        ),
        (
            tongue_fold_display,
            tongue_fold_display_y,
            SADDLE_COLOR,
            0.80,
        ),
    ):
        edge_line, = ax.plot(
            edge_x,
            edge_y,
            color=edge_color,
            linewidth=edge_width,
            linestyle=(0.0, (3.0, 2.5)),
            zorder=10,
        )
        edge_line.set_dash_capstyle("butt")

    # Independent middle segment of the asymmetric-branch B=0.56 contour.
    # It is clipped away from the orange and green dynamical curves.
    alpha_grid, fs_grid = np.meshgrid(alpha, fs)
    local_b_asymmetric = np.ma.masked_invalid(b_asymmetric.copy())
    local_b_asymmetric = np.ma.masked_where(
        (alpha_grid < 0.48)
        | (alpha_grid > 0.565)
        | (fs_grid < 0.64)
        | (fs_grid > 0.76),
        local_b_asymmetric,
    )
    hidden = ax.contour(
        alpha,
        fs,
        local_b_asymmetric,
        levels=[B_CUTOFF],
        colors="none",
        linewidths=0.0,
    )
    classification_segments = [segment.copy() for segment in hidden.allsegs[0]]
    hidden.remove()
    green_reference = np.vstack(green_reference)
    orange_reference = np.column_stack(
        (pitch_x[pitch_physical & (pitch_cubic > 0.0)],
         pitch_y[pitch_physical & (pitch_cubic > 0.0)])
    )

    def distance_to_curve(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
        delta = points[:, None, :] - reference[None, :, :]
        return np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))

    for segment in classification_segments:
        if segment.shape[0] < 2:
            continue
        independent = (
            (distance_to_curve(segment, green_reference) > 0.0075)
            & (distance_to_curve(segment, orange_reference) > 0.0075)
        )
        for start, stop in split_visible_runs(independent):
            if stop - start >= 2:
                ax.plot(
                    segment[start:stop, 0],
                    segment[start:stop, 1],
                    color=CLASSIFICATION_COLOR,
                    linewidth=1.45,
                    linestyle=(0, (2.0, 2.0)),
                    zorder=11,
                )

    class_start = float(
        max(
            x_of_y(tongue_pitch_x, tongue_pitch_y, np.asarray([classification_y]))[0],
            x_of_y(tongue_fold_x, tongue_fold_y, np.asarray([classification_y]))[0],
        )
    )
    ax.plot(
        [class_start, 0.70],
        [classification_y, classification_y],
        color=CLASSIFICATION_COLOR,
        linewidth=1.45,
        linestyle=(0, (2.0, 2.0)),
        zorder=11,
    )
    draw_special_points(ax, "c")


def style_axis(
    ax,
    panel: str,
    x_label: str,
    y_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
    subtitle: str,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{value:g}" for value in xticks])
    ax.set_yticklabels([f"{value:g}" for value in yticks])
    ax.set_xlabel(x_label, fontsize=11.0, labelpad=4)
    ax.set_ylabel(y_label, fontsize=11.0, labelpad=5)
    ax.tick_params(axis="both", labelsize=8.0, width=0.9, length=3)
    ax.set_box_aspect(1)
    ax.grid(linestyle=":", color="black", alpha=0.17, linewidth=0.72, zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    ax.text(
        -0.17,
        1.075,
        panel,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_title(subtitle, fontsize=8.0, pad=7, y=1.01)


def save_trace() -> None:
    audited_points = {
        panel: load_special_points(panel) for panel in ("a", "b", "c")
    }
    trace = {
        "figure": OUTPUT_PNG.name,
        "classification": {
            "state_mode": STATE_MODE,
            "B_cutoff": B_CUTOFF,
            "bimodality_mode": BIMODALITY_MODE,
            "mirror_tolerance": MIRROR_TOL,
            "symmetric_bimodal": f"B > {B_CUTOFF:.2f} and self-mirror",
            "asymmetric_bimodal": f"B > {B_CUTOFF:.2f} and a distinct mirror partner exists",
            "biased_unimodal": f"B <= {B_CUTOFF:.2f} and a distinct mirror partner exists",
            "neutral_unimodal": f"B <= {B_CUTOFF:.2f} and self-mirror",
        },
        "panels": {
            "a": {
                "axes": ["alpha2", "fs"],
                "fixed": {"dc": 0.5, "alpha3": 0.0},
                "inputs": [PANEL_A_GRID.name, PANEL_A_BOUNDARIES.name],
                "slice_markers": {"fs": 0.5, "alpha2": [0.03, 0.10, 0.30]},
                "special_points": audited_points["a"],
            },
            "b": {
                "axes": ["alpha2", "dc"],
                "fixed": {"fs": 0.5, "alpha3": 0.0},
                "inputs": [
                    PANEL_B_CACHE.name,
                    PANEL_B_MACRO.name,
                    PANEL_B_PITCHFORK.name,
                ],
                "special_points": audited_points["b"],
            },
        "c": {
                "axes": ["alpha3", "fs"],
                "fixed": {"dc": 0.5, "alpha2": 0.0},
                "inputs": [
                    PANEL_C_ATTRACTORS.name,
                    PANEL_C_CURVES.name,
                    PANEL_C_PITCHFORK_TYPE.name,
                ],
                "special_points": audited_points["c"],
                "bifurcation_audit": {
                    "pitchfork_criticality_check": (
                        "The normal-form cubic coefficient was recomputed at "
                        "four finite-difference steps (1e-3 to 1.25e-4); the "
                        "representative values agree to better than 1e-6 "
                        "relative precision."
                    ),
                    "near_pitchfork_right_arc": (
                        (
                            "Only above the panel-c TP, the right asymmetric "
                            "fold and the "
                            "subcritical pitchfork delimit a hysteretic "
                            "coexistence tongue. Where the numerical separation "
                            f"is below {C_MIN_RIBBON_PT:.1f} pt, both display "
                            "edges are expanded symmetrically to that minimum "
                            "width. One thin unoutlined orange curve and one "
                            "thin unoutlined green curve are drawn on the two "
                            "display edges; no centre stroke is added. "
                            "Below TP, only the supercritical pitchfork is "
                            "displayed."
                        )
                        if SEPARATE_C_NEAR_LINES
                        else
                        (
                            "The full-state asymmetric fold approaches the "
                            "pitchfork closely near the lower TP but remains a "
                            "distinct continuation branch. It is drawn "
                            "continuously from TP onward with a thinner green "
                            "dash pattern."
                        )
                    ),
                },
            },
        },
        "note": f"Grey dotted curves are B={B_CUTOFF:.2f} classification boundaries, not dynamical bifurcations.",
        "point_audit_source": POINT_AUDIT.name,
    }
    OUTPUT_TRACE.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    required = (
        PANEL_A_GRID,
        PANEL_A_BOUNDARIES,
        PANEL_B_CACHE,
        PANEL_B_MACRO,
        PANEL_B_PITCHFORK,
        PANEL_B_PC,
        PANEL_C_ATTRACTORS,
        PANEL_C_CURVES,
        PANEL_C_PITCHFORK_TYPE,
        POINT_AUDIT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), dpi=400)
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c_replacement(axes[2])

    style_axis(
        axes[0],
        "(a)",
        r"$\alpha_2$",
        r"$f_s$",
        (0.0, 0.30),
        (0.20, 0.90),
        [0.0, 0.1, 0.2, 0.3],
        [0.2, 0.4, 0.6, 0.9],
        r"$d_c=0.5,\ \alpha_3=0$",
    )
    style_axis(
        axes[1],
        "(b)",
        r"$\alpha_2$",
        r"$d_c$",
        (0.0, 0.30),
        (0.0, 1.00),
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        r"$f_s=0.5,\ \alpha_3=0$",
    )
    style_axis(
        axes[2],
        "(c)",
        r"$\alpha_3$",
        r"$f_s$",
        (0.0, 0.70),
        (0.20, 0.90),
        [0.0, 0.2, 0.4, 0.6, 0.7],
        [0.2, 0.4, 0.6, 0.9],
        r"$d_c=0.5,\ \alpha_2=0$",
    )

    boundary_handles = [
        Line2D([0], [0], color=SADDLE_COLOR, linewidth=1.6, linestyle=(0, (2.4, 1.7)), label="Saddle-node"),
        Line2D([0], [0], color=PITCHFORK_COLOR, linewidth=1.6, linestyle=(0, (2.4, 1.7)), label="Supercritical PF"),
        Line2D([0], [0], color=SUBCRITICAL_COLOR, linewidth=1.6, linestyle=(0, (2.4, 1.7)), label="Subcritical PF"),
        Line2D([0], [0], color=CLASSIFICATION_COLOR, linewidth=1.6, linestyle=(0, (2.0, 2.0)), label=f"$B={B_CUTOFF:.2f}$"),
    ]

    if STATE_MODE == "four":
        state_classes, state_colors, state_labels = (
            STATE_CLASSES,
            STATE_COLORS,
            STATE_LABELS,
        )
    else:
        state_classes, state_colors, state_labels = (
            THREE_STATE_CLASSES,
            THREE_STATE_COLORS,
            THREE_STATE_LABELS,
        )
    state_handles = [
        Patch(
            facecolor=state_colors[label],
            edgecolor="black",
            linewidth=0.8,
            label=state_labels[label],
        )
        for label in state_classes
    ]
    if SEPARATE_C_NEAR_LINES:
        state_handles.append(
            Patch(
                facecolor=COEX_BU_NU_COLOR,
                edgecolor="black",
                linewidth=0.8,
                label="BU–NU coex." if STATE_MODE == "three" else "Biased–neutral unimodal coexistence",
            )
        )
    line_legend = fig.legend(
        handles=boundary_handles,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.105),
        ncol=4,
        fontsize=8.0,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.25,
        columnspacing=0.70,
        borderpad=0.0,
    )
    state_legend = fig.legend(
        handles=state_handles,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.045),
        ncol=len(state_handles),
        fontsize=8.0,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.25,
        columnspacing=0.85,
        borderpad=0.0,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.245,
        top=0.93,
        wspace=0.30,
    )
    fig.canvas.draw()

    # The two rows are separate legends so each row has genuinely uniform
    # inter-entry spacing. Draw one shared frame around their rendered union.
    renderer = fig.canvas.get_renderer()
    legend_bbox = mtransforms.Bbox.union(
        [
            line_legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            ),
            state_legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            ),
        ]
    )
    # Shift both rows by the same amount so their shared frame is centred on
    # the full figure, while preserving the left alignment within the frame.
    legend_shift_x = 0.5 - 0.5 * (legend_bbox.x0 + legend_bbox.x1)
    line_legend.set_bbox_to_anchor(
        (0.075 + legend_shift_x, 0.105), transform=fig.transFigure
    )
    state_legend.set_bbox_to_anchor(
        (0.075 + legend_shift_x, 0.045), transform=fig.transFigure
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = mtransforms.Bbox.union(
        [
            line_legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            ),
            state_legend.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            ),
        ]
    )
    frame_pad_x = 0.008
    frame_pad_y = 0.008
    legend_frame = FancyBboxPatch(
        (legend_bbox.x0 - frame_pad_x, legend_bbox.y0 - frame_pad_y),
        legend_bbox.width + 2.0 * frame_pad_x,
        legend_bbox.height + 2.0 * frame_pad_y,
        boxstyle="round,pad=0.003,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="0.65",
        linewidth=0.8,
        alpha=0.97,
        clip_on=False,
        zorder=1,
    )
    fig.add_artist(legend_frame)
    line_legend.set_zorder(3)
    state_legend.set_zorder(3)
    fig.canvas.draw()
    fig.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUTPUT_TIFF, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    taxonomy_text = (
        "Stable attractors are classified as symmetric bimodal, asymmetric "
        "bimodal, biased unimodal, or neutral unimodal; coexistence colours "
        "indicate parameter regions supporting more than one attractor type. "
        if STATE_MODE == "four"
        else
        "BP, BU, and NU denote the bimodal polarized, biased unimodal, and "
        "neutral unimodal states, respectively. The bimodal class merges the "
        "symmetric- and asymmetric-bimodal "
        "subtypes; coexistence colours indicate parameter regions supporting "
        "more than one of these three macroscopic states. "
    )
    caption = (
        f"{'Four' if STATE_MODE == 'four' else 'Three'}-state phase maps of the 42-bin mean-field model using the "
        "finite-sample-corrected bimodality coefficient. "
        "(a) The (alpha2, fs) plane at dc=0.5 and alpha3=0; the dotted fs=0.50 "
        "slice marks cases I–III at alpha2=0.03, 0.10, and 0.30. The diamond "
        "marks the saddle-node–pitchfork codimension-two point (SNP). "
        "(b) The (alpha2, dc) plane at fs=0.5 and alpha3=0. "
        "(c) The (alpha3, fs) plane at dc=0.5 and alpha2=0. "
        + taxonomy_text
        + "Green, yellow, and orange curves denote stable-involving continued "
        "saddle-node, supercritical-pitchfork, and subcritical-pitchfork "
        "bifurcations, respectively. "
        "White circles mark verified tricritical pitchfork points (TP); white "
        "diamonds mark saddle-node–pitchfork codimension-two points (SNP), "
        "verified by simultaneous even- and odd-sector criticality at the same "
        "fixed point. White triangles mark the retained pitchfork criticality-"
        "change points (PC) in panel (b). In panel (c), the lower point is a "
        "verified TP because the reduced cubic coefficient vanishes while the "
        "quintic coefficient remains nonzero. "
        + (
            "Above this TP, the subcritical pitchfork and right saddle-node fold "
            "bound a biased–neutral unimodal coexistence tongue. The dark-blue "
            "region is filled between the displayed orange and green boundaries. "
            f"Where their numerical separation is below {C_MIN_RIBBON_PT:.1f} pt, "
            "both edges are expanded symmetrically to that minimum width. "
            "Thin unoutlined orange and green curves mark the two displayed "
            "edges, with no additional centre stroke. "
            "Below TP, only the "
            "supercritical pitchfork is shown. "
            if SEPARATE_C_NEAR_LINES
            else
            "a green fold is drawn only where it is independently resolvable "
            "from the pitchfork at final figure size. "
        )
        + f"Grey dotted curves mark the B={B_CUTOFF:.2f} classification threshold "
        "and are not bifurcations."
    )
    OUTPUT_CAPTION.write_text(caption + "\n", encoding="utf-8")
    save_trace()
    for path in (
        OUTPUT_PNG,
        OUTPUT_PDF,
        OUTPUT_SVG,
        OUTPUT_TIFF,
        OUTPUT_CAPTION,
        OUTPUT_TRACE,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
