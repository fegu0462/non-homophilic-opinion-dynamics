"""Compute and plot the three 42-bin mean-field parameter planes.

Panels
------
(a) (alpha2, fs), at dc=0.5 and alpha3=0;
(b) (alpha2, dc), at fs=0.5 and alpha3=0;
(c) (alpha3, fs), at dc=0.5 and alpha2=0.

For immediate reproduction, the archived numerical scan and pseudo-arclength
continuation data in ``data`` can be plotted directly.  The
same module also contains the complete from-scratch route: the 42-dimensional
mean-field equation, multistart fixed-point scan, analytic stability test,
state classification, and pseudo-arclength continuation.
"""


from __future__ import annotations


import csv


import argparse


import json


import os


import pickle


from collections import defaultdict


from dataclasses import dataclass


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


from scipy.optimize import least_squares


from scipy.ndimage import distance_transform_edt, gaussian_filter


from scipy.signal import savgol_filter


from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = ROOT / "data"


FIGURE_DIR = ROOT / "figures"


FIGURE_DIR.mkdir(parents=True, exist_ok=True)


B_CUTOFF = float(os.environ.get("B_CUTOFF", "0.56"))


B_TAG = f"B{int(round(100 * B_CUTOFF)):03d}"


BIMODALITY_MODE = os.environ.get("BIMODALITY_MODE", "sample_corrected").strip().lower()


if BIMODALITY_MODE not in {"population", "sample_corrected"}:
    raise ValueError(f"Unsupported BIMODALITY_MODE={BIMODALITY_MODE!r}")


MODE_SUFFIX = "_sample_corrected" if BIMODALITY_MODE == "sample_corrected" else ""


STATE_MODE = os.environ.get("STATE_MODE", "three").strip().lower()


if STATE_MODE not in {"four", "three"}:
    raise ValueError(f"Unsupported STATE_MODE={STATE_MODE!r}")


STATE_TAG = "four_state" if STATE_MODE == "four" else "three_state"


SHOW_BF_PREVIEW = os.environ.get("SHOW_BF_PREVIEW", "0") == "1"


SEPARATE_C_NEAR_LINES = os.environ.get("SEPARATE_C_NEAR_LINES", "1") == "1"


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


POINT_AUDIT = ROOT / "data" / "special_points_42bin_audit.csv"


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


def required_data_files() -> list[Path]:
    """Return the numerical files used by the three panels."""
    return [
        PANEL_A_GRID,
        PANEL_A_BOUNDARIES,
        PANEL_B_CACHE,
        PANEL_B_CURVES,
        PANEL_B_MACRO,
        PANEL_B_PITCHFORK,
        PANEL_B_PC,
        PANEL_C_ATTRACTORS,
        PANEL_C_CURVES,
        PANEL_C_PITCHFORK_TYPE,
        POINT_AUDIT,
    ]


def check_data() -> None:
    """Raise a clear error if a required 42-bin data file is missing."""
    missing = [str(path) for path in required_data_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing 42-bin data files:\n" + "\n".join(missing))


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


# =============================================================================
# From-scratch 42-bin calculation
# =============================================================================

# The plotting functions above reproduce the manuscript panels immediately from
# the archived numerical data.  The functions below are the independent
# calculation route: they start from the 42-dimensional mean-field equation,
# find stable fixed points, evaluate the analytic Jacobian, classify the
# attractors, and (optionally) continue critical curves by pseudo-arclength.


@dataclass(frozen=True)
class MF42Settings:
    bins_per_side: int = 20
    gamma: float = 4.0
    sigma_stubborn: float = 0.8
    sigma_nonstubborn: float = 0.1
    stubborn_positive_share: float = 0.5
    b_cutoff: float = 0.56
    solution_tolerance: float = 3.0e-5
    residual_tolerance: float = 2.0e-8
    stability_margin: float = 2.0e-7
    max_iterations: int = 60_000


@dataclass(frozen=True)
class MF42Parameters:
    alpha2: float
    alpha3: float
    fs: float
    dc: float


@dataclass(frozen=True)
class MF42Plane:
    name: str
    x_parameter: str
    y_parameter: str
    x_values: np.ndarray
    y_values: np.ndarray
    fixed: dict[str, float]

    def parameters(self, x_value: float, y_value: float) -> MF42Parameters:
        values = dict(self.fixed)
        values[self.x_parameter] = float(x_value)
        values[self.y_parameter] = float(y_value)
        return MF42Parameters(**values)


@dataclass
class MF42Population:
    eta: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.eta.size)


MF42_DEFAULTS = MF42Settings()
MF42_GENERATED_DIR = ROOT / "data" / "generated_42bin"
MF42_GENERATED_FIGURE_DIR = FIGURE_DIR / "generated_42bin"


def manuscript_planes_42bin(quick: bool = False) -> list[MF42Plane]:
    """Return the three parameter planes used in the paper.

    ``quick=True`` is a small end-to-end test.  ``quick=False`` uses the full
    scan grids; it is computationally expensive because every point is a
    multistart 42-dimensional fixed-point calculation.
    """
    if quick:
        alpha2_a = np.linspace(0.0, 0.30, 5)
        fs_a = np.linspace(0.20, 0.90, 5)
        alpha2_b = np.linspace(0.0, 0.30, 5)
        dc_b = np.linspace(0.0, 1.0, 5)
        alpha3_c = np.linspace(0.0, 0.70, 5)
        fs_c = np.linspace(0.20, 0.90, 5)
    else:
        # Panel (a) uses the same 81 x 181 grid as the archived scan.
        alpha2_a = np.round(np.arange(0.0, 0.4001, 0.005), 12)
        fs_a = np.round(np.arange(0.0, 0.9001, 0.005), 12)
        alpha2_b = np.round(np.arange(0.0, 0.3001, 0.01), 12)
        dc_b = np.round(np.arange(0.0, 1.0001, 0.01), 12)
        alpha3_c = np.round(np.arange(0.0, 0.7001, 0.01), 12)
        fs_c = np.round(np.arange(0.20, 0.9001, 0.01), 12)

    return [
        MF42Plane(
            "panel_a_alpha2_fs",
            "alpha2",
            "fs",
            alpha2_a,
            fs_a,
            {"alpha2": 0.0, "alpha3": 0.0, "fs": 0.5, "dc": 0.5},
        ),
        MF42Plane(
            "panel_b_alpha2_dc",
            "alpha2",
            "dc",
            alpha2_b,
            dc_b,
            {"alpha2": 0.0, "alpha3": 0.0, "fs": 0.5, "dc": 0.5},
        ),
        MF42Plane(
            "panel_c_alpha3_fs",
            "alpha3",
            "fs",
            alpha3_c,
            fs_c,
            {"alpha2": 0.0, "alpha3": 0.0, "fs": 0.5, "dc": 0.5},
        ),
    ]


def make_population_42(
    fs: float, settings: MF42Settings = MF42_DEFAULTS
) -> MF42Population:
    """Build the two stubborn groups and 40 non-stubborn eta bins."""
    n = settings.bins_per_side
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    eta = np.concatenate(([1.0, -1.0], eta_positive, eta_negative))
    sigma = np.concatenate(
        ([settings.sigma_stubborn, settings.sigma_stubborn],
         np.full(2 * n, settings.sigma_nonstubborn))
    )
    weights = np.concatenate(
        ([fs * settings.stubborn_positive_share,
          fs * (1.0 - settings.stubborn_positive_share)],
         np.full(2 * n, (1.0 - fs) / (2.0 * n)))
    )
    return MF42Population(eta=eta, sigma=sigma, weights=weights)


def mean_field_map_42(
    state: np.ndarray,
    parameters: MF42Parameters,
    population: MF42Population | None = None,
    settings: MF42Settings = MF42_DEFAULTS,
    return_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate the 42-bin mean-field map and its analytic state Jacobian."""
    x = np.asarray(state, dtype=float)
    pop = population or make_population_42(parameters.fs, settings)
    if x.shape != (pop.dimension,):
        raise ValueError(f"Expected a {pop.dimension}-component state, got {x.shape}.")

    difference = x[:, None] - x[None, :]
    distance = np.abs(difference)
    exponent = np.clip(settings.gamma * (distance - parameters.dc), -100.0, 100.0)
    logistic = 1.0 / (1.0 + np.exp(exponent))
    confidence = (1.0 + np.exp(-settings.gamma * parameters.dc)) * logistic

    alpha1 = 1.0 - parameters.alpha2 - parameters.alpha3
    if alpha1 < -1.0e-12:
        raise ValueError("alpha2 + alpha3 must not exceed one.")
    source_extremity = np.abs(x)[None, :]
    receiver_extremity = np.abs(x)[:, None]
    connection = (
        alpha1 * confidence
        + parameters.alpha2 * source_extremity
        + parameters.alpha3 * receiver_extremity
    )
    denominator = connection @ pop.weights
    if np.any(denominator <= 1.0e-14):
        raise RuntimeError("The normalized interaction denominator reached zero.")
    numerator = connection @ (pop.weights * x)
    neighbour_mean = numerator / denominator
    update = pop.sigma * pop.eta + (1.0 - pop.sigma) * neighbour_mean
    if not return_jacobian:
        return update, None

    sign_difference = np.sign(difference)
    dconfidence = (
        -settings.gamma
        * (1.0 + np.exp(-settings.gamma * parameters.dc))
        * logistic
        * (1.0 - logistic)
    )
    dconnection_receiver = (
        alpha1 * dconfidence * sign_difference
        + parameters.alpha3 * np.sign(x)[:, None]
    )
    dconnection_source = (
        -alpha1 * dconfidence * sign_difference
        + parameters.alpha2 * np.sign(x)[None, :]
    )
    ddenominator = dconnection_source * pop.weights[None, :]
    dnumerator = pop.weights[None, :] * (
        dconnection_source * x[None, :] + connection
    )
    diagonal = np.arange(x.size)
    ddenominator[diagonal, diagonal] += dconnection_receiver @ pop.weights
    dnumerator[diagonal, diagonal] += dconnection_receiver @ (pop.weights * x)
    jacobian = (1.0 - pop.sigma)[:, None] * (
        dnumerator * denominator[:, None]
        - numerator[:, None] * ddenominator
    ) / denominator[:, None] ** 2
    return update, jacobian


def reflect_state_42(state: np.ndarray) -> np.ndarray:
    """Apply the exact opinion-sign/intrinsic-bias reflection symmetry."""
    x = np.asarray(state, dtype=float)
    n = (x.size - 2) // 2
    reflected = np.empty_like(x)
    reflected[0], reflected[1] = -x[1], -x[0]
    reflected[2 : 2 + n] = -x[2 + n :][::-1]
    reflected[2 + n :] = -x[2 : 2 + n][::-1]
    return reflected


def _append_distinct_42(
    collection: list[np.ndarray], candidate: np.ndarray, tolerance: float
) -> bool:
    candidate = np.asarray(candidate, dtype=float)
    if any(np.max(np.abs(candidate - old)) <= tolerance for old in collection):
        return False
    collection.append(candidate.copy())
    return True


def seed_bank_42(settings: MF42Settings = MF42_DEFAULTS) -> list[np.ndarray]:
    """Deterministic seeds for bimodal, biased, neutral, and detached-bin states."""
    n = settings.bins_per_side
    seeds: list[np.ndarray] = []

    def split(amplitude: float, shift: float = 0.0) -> np.ndarray:
        return np.concatenate(
            ([1.0, -1.0], np.full(n, amplitude + shift),
             np.full(n, -amplitude + shift))
        )

    for amplitude in (0.02, 0.20, 0.50, 0.80, 0.95):
        _append_distinct_42(seeds, split(amplitude), 1.0e-12)
    for amplitude in (0.35, 0.65, 0.90):
        for shift in (-0.04, -0.012, 0.012, 0.04):
            _append_distinct_42(seeds, split(amplitude, shift), 1.0e-12)
    for value in (-0.85, -0.55, -0.25, 0.0, 0.25, 0.55, 0.85):
        seed = np.concatenate(([1.0, -1.0], np.full(2 * n, value)))
        _append_distinct_42(seeds, seed, 1.0e-12)
    eta_positive = (np.arange(n) + 0.5) / n
    eta_negative = -1.0 + (np.arange(n) + 0.5) / n
    eta_ns = np.concatenate((eta_positive, eta_negative))
    for scale in (0.35, 0.70, 1.0):
        seed = np.concatenate(([1.0, -1.0], scale * eta_ns))
        _append_distinct_42(seeds, seed, 1.0e-12)
    for amplitude in (0.50, 0.85):
        for number in (1, 2, 4, 8):
            positive_flip = split(amplitude)
            positive_flip[2 : 2 + number] = -amplitude
            negative_flip = split(amplitude)
            negative_flip[-number:] = amplitude
            _append_distinct_42(seeds, positive_flip, 1.0e-12)
            _append_distinct_42(seeds, negative_flip, 1.0e-12)
    return seeds


def _iterate_seed_42(
    seed: np.ndarray,
    parameters: MF42Parameters,
    population: MF42Population,
    settings: MF42Settings,
) -> np.ndarray | None:
    x = np.asarray(seed, dtype=float).copy()
    for _ in range(settings.max_iterations):
        update, _ = mean_field_map_42(
            x, parameters, population, settings, return_jacobian=False
        )
        error = float(np.max(np.abs(update - x)))
        x = update
        if error <= settings.residual_tolerance:
            return x
    return None


def bimodality_and_mean_42(values: np.ndarray) -> tuple[float, float]:
    """Finite-sample-corrected bimodality coefficient and mean opinion."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if float(np.std(values)) <= 1.0e-12:
        return 0.0, mean
    g1 = float(skew(values, bias=False))
    g2 = float(kurtosis(values, fisher=True, bias=False))
    n = values.size
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = g2 + correction
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan"), mean
    return float((g1**2 + 1.0) / denominator), mean


def stable_attractors_42(
    parameters: MF42Parameters,
    continuation_seeds: list[np.ndarray] | None = None,
    settings: MF42Settings = MF42_DEFAULTS,
) -> list[dict[str, object]]:
    """Find distinct stable fixed points by deterministic multistart iteration."""
    pop = make_population_42(parameters.fs, settings)
    seeds: list[np.ndarray] = []
    for seed in (continuation_seeds or []) + seed_bank_42(settings):
        _append_distinct_42(seeds, seed, 1.0e-10)

    candidates: list[np.ndarray] = []
    for seed in seeds:
        solution = _iterate_seed_42(seed, parameters, pop, settings)
        if solution is not None:
            _append_distinct_42(candidates, solution, settings.solution_tolerance)

    if np.isclose(settings.stubborn_positive_share, 0.5):
        for solution in list(candidates):
            reflected = reflect_state_42(solution)
            update, _ = mean_field_map_42(
                reflected, parameters, pop, settings, return_jacobian=False
            )
            if np.max(np.abs(update - reflected)) <= 5.0 * settings.residual_tolerance:
                _append_distinct_42(candidates, reflected, settings.solution_tolerance)

    stable: list[dict[str, object]] = []
    for solution in candidates:
        update, jacobian = mean_field_map_42(solution, parameters, pop, settings)
        residual = float(np.max(np.abs(update - solution)))
        eigenvalues = np.linalg.eigvals(jacobian)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        if residual > 5.0 * settings.residual_tolerance:
            continue
        if spectral_radius >= 1.0 - settings.stability_margin:
            continue
        b_value, s_value = bimodality_and_mean_42(solution[2:])
        asymmetric = (
            float(np.max(np.abs(reflect_state_42(solution) - solution))) > 2.0e-5
        )
        if b_value > settings.b_cutoff:
            label = "Asymmetric bimodal" if asymmetric else "Symmetric bimodal"
        else:
            label = "Biased unimodal" if asymmetric else "Neutral unimodal"
        stable.append(
            {
                "state": solution,
                "B": b_value,
                "S": s_value,
                "label": label,
                "residual": residual,
                "spectral_radius": spectral_radius,
            }
        )
    stable.sort(key=lambda item: (abs(float(item["S"])), float(item["B"])))
    return stable


def macroscopic_state_42(attractors: list[dict[str, object]]) -> str:
    """Merge the four calculation labels into the three-state figure labels."""
    labels = {str(item["label"]) for item in attractors}
    has_bp = bool(labels & {"Symmetric bimodal", "Asymmetric bimodal"})
    has_bu = "Biased unimodal" in labels
    has_nu = "Neutral unimodal" in labels
    if has_bp and has_bu:
        return "BP-BU coexistence"
    if has_bp and has_nu:
        return "BP-NU coexistence"
    if has_bu and has_nu:
        return "BU-NU coexistence"
    if has_bp:
        return "Bimodal polarized"
    if has_bu:
        return "Biased unimodal"
    if has_nu:
        return "Neutral unimodal"
    return "Undetected"


def scan_plane_42(
    plane: MF42Plane,
    output_dir: Path | str = MF42_GENERATED_DIR,
    settings: MF42Settings = MF42_DEFAULTS,
) -> dict[str, object]:
    """Scan one plane from the mean-field equation and save all attractors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points: dict[tuple[int, int], list[dict[str, object]]] = {}
    grid_rows: list[dict[str, object]] = []
    attractor_rows: list[dict[str, object]] = []
    previous_row: dict[int, list[np.ndarray]] = {}

    for iy, y_value in enumerate(plane.y_values):
        previous_in_row: list[np.ndarray] = []
        current_row: dict[int, list[np.ndarray]] = {}
        for ix, x_value in enumerate(plane.x_values):
            parameters = plane.parameters(x_value, y_value)
            vertical = previous_row.get(ix, [])
            attractors = stable_attractors_42(
                parameters, previous_in_row + vertical, settings
            )
            points[(iy, ix)] = attractors
            states = [np.asarray(item["state"]) for item in attractors]
            previous_in_row = states
            current_row[ix] = states
            macro = macroscopic_state_42(attractors)
            grid_rows.append(
                {
                    "x": x_value,
                    "y": y_value,
                    plane.x_parameter: x_value,
                    plane.y_parameter: y_value,
                    "state": macro,
                    "stable_count": len(attractors),
                }
            )
            for index, item in enumerate(attractors):
                attractor_rows.append(
                    {
                        "x": x_value,
                        "y": y_value,
                        plane.x_parameter: x_value,
                        plane.y_parameter: y_value,
                        "attractor": index,
                        "label": item["label"],
                        "B": item["B"],
                        "S": item["S"],
                        "spectral_radius": item["spectral_radius"],
                        "residual": item["residual"],
                        "state_vector": json.dumps(
                            np.asarray(item["state"], dtype=float).tolist()
                        ),
                    }
                )
        previous_row = current_row
        print(
            f"{plane.name}: completed row {iy + 1}/{len(plane.y_values)} "
            f"({plane.y_parameter}={y_value:.4g})"
        )

    grid_path = output_dir / f"{plane.name}_grid.csv"
    attractor_path = output_dir / f"{plane.name}_attractors.csv"
    _write_dict_rows_42(grid_path, grid_rows)
    _write_dict_rows_42(attractor_path, attractor_rows)
    return {
        "plane": plane,
        "points": points,
        "grid_rows": grid_rows,
        "attractor_rows": attractor_rows,
        "grid_path": grid_path,
        "attractor_path": attractor_path,
    }


def _write_dict_rows_42(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows were generated for {path.name}.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _critical_multiplier_42(
    state: np.ndarray,
    parameters: MF42Parameters,
    settings: MF42Settings,
) -> tuple[float, np.ndarray]:
    pop = make_population_42(parameters.fs, settings)
    _, jacobian = mean_field_map_42(state, parameters, pop, settings)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    return float(np.real(eigenvalues[index]) - 1.0), np.real(eigenvectors[:, index])


def _pack_critical_42(state: np.ndarray, x_value: float, y_value: float) -> np.ndarray:
    scale = np.sqrt(state.size)
    return np.concatenate((state / scale, [x_value, y_value]))


def _unpack_critical_42(vector: np.ndarray) -> tuple[np.ndarray, float, float]:
    dimension = vector.size - 2
    scale = np.sqrt(dimension)
    return scale * vector[:dimension], float(vector[-2]), float(vector[-1])


def _critical_residual_42(
    vector: np.ndarray,
    plane: MF42Plane,
    settings: MF42Settings,
) -> np.ndarray:
    state, x_value, y_value = _unpack_critical_42(vector)
    parameters = plane.parameters(x_value, y_value)
    pop = make_population_42(parameters.fs, settings)
    update, _ = mean_field_map_42(state, parameters, pop, settings)
    critical, _ = _critical_multiplier_42(state, parameters, settings)
    return np.concatenate((update - state, [critical]))


def _finite_jacobian_42(function, vector: np.ndarray) -> np.ndarray:
    base = function(vector)
    answer = np.empty((base.size, vector.size))
    for column in range(vector.size):
        step = 2.0e-6 * max(1.0, abs(float(vector[column])))
        plus = vector.copy()
        minus = vector.copy()
        plus[column] += step
        minus[column] -= step
        answer[:, column] = (function(plus) - function(minus)) / (2.0 * step)
    return answer


def _critical_event_42(
    state: np.ndarray,
    parameters: MF42Parameters,
    settings: MF42Settings,
) -> str:
    symmetric = np.max(np.abs(reflect_state_42(state) - state)) < 2.0e-4
    _, eigenvector = _critical_multiplier_42(state, parameters, settings)
    reflected_vector = reflect_state_42(eigenvector)
    odd_error = np.linalg.norm(reflected_vector + eigenvector)
    even_error = np.linalg.norm(reflected_vector - eigenvector)
    return "Pitchfork" if symmetric and odd_error < even_error else "Saddle-node"


def _refine_critical_seed_42(
    state: np.ndarray,
    x_value: float,
    y_value: float,
    plane: MF42Plane,
    settings: MF42Settings,
) -> np.ndarray | None:
    initial = _pack_critical_42(state, x_value, y_value)[:-1]

    def residual(vector: np.ndarray) -> np.ndarray:
        full = np.concatenate((vector, [y_value]))
        return _critical_residual_42(full, plane, settings)

    answer = least_squares(
        residual,
        initial,
        xtol=2.0e-9,
        ftol=2.0e-9,
        gtol=2.0e-9,
        max_nfev=250,
    )
    full = np.concatenate((answer.x, [y_value]))
    if np.max(np.abs(residual(answer.x))) > 2.0e-5:
        return None
    _, refined_x, _ = _unpack_critical_42(full)
    if not plane.x_values[0] - 0.03 <= refined_x <= plane.x_values[-1] + 0.03:
        return None
    return full


def _candidate_critical_seeds_42(
    scan: dict[str, object], settings: MF42Settings
) -> list[np.ndarray]:
    plane = scan["plane"]
    points = scan["points"]
    candidates: list[np.ndarray] = []
    for iy, y_value in enumerate(plane.y_values):
        for ix in range(len(plane.x_values) - 1):
            left = points[(iy, ix)]
            right = points[(iy, ix + 1)]
            if macroscopic_state_42(left) == macroscopic_state_42(right):
                continue
            choices: list[tuple[dict[str, object], float]] = []
            choices.extend((item, float(plane.x_values[ix])) for item in left)
            choices.extend((item, float(plane.x_values[ix + 1])) for item in right)
            if not choices:
                continue
            item, x_guess = max(
                choices, key=lambda value: float(value[0]["spectral_radius"])
            )
            refined = _refine_critical_seed_42(
                np.asarray(item["state"]), x_guess, float(y_value), plane, settings
            )
            if refined is None:
                continue
            _, x_refined, y_refined = _unpack_critical_42(refined)
            if any(
                np.hypot(x_refined - _unpack_critical_42(old)[1],
                         y_refined - _unpack_critical_42(old)[2]) < 0.025
                for old in candidates
            ):
                continue
            candidates.append(refined)
    return candidates


def _trace_direction_42(
    seed: np.ndarray,
    direction: float,
    plane: MF42Plane,
    settings: MF42Settings,
    step: float,
    maximum_points: int,
) -> list[np.ndarray]:
    function = lambda vector: _critical_residual_42(vector, plane, settings)
    curve = [seed.copy()]
    previous_tangent: np.ndarray | None = None
    current = seed.copy()
    x_min, x_max = float(plane.x_values[0]), float(plane.x_values[-1])
    y_min, y_max = float(plane.y_values[0]), float(plane.y_values[-1])
    for _ in range(maximum_points - 1):
        jacobian = _finite_jacobian_42(function, current)
        _, _, right = np.linalg.svd(jacobian, full_matrices=True)
        tangent = right[-1]
        if previous_tangent is not None and np.dot(tangent, previous_tangent) < 0.0:
            tangent = -tangent
        elif previous_tangent is None:
            tangent *= direction
        tangent /= np.linalg.norm(tangent)
        predictor = current + step * tangent

        def augmented(vector: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (function(vector), [np.dot(vector - predictor, tangent)])
            )

        answer = least_squares(
            augmented,
            predictor,
            xtol=5.0e-9,
            ftol=5.0e-9,
            gtol=5.0e-9,
            max_nfev=160,
        )
        if np.max(np.abs(augmented(answer.x))) > 4.0e-5:
            break
        current = answer.x
        _, x_value, y_value = _unpack_critical_42(current)
        if not (x_min - 0.03 <= x_value <= x_max + 0.03):
            break
        if not (y_min - 0.03 <= y_value <= y_max + 0.03):
            break
        curve.append(current.copy())
        previous_tangent = tangent
    return curve


def continue_boundaries_42(
    scan: dict[str, object],
    output_dir: Path | str = MF42_GENERATED_DIR,
    settings: MF42Settings = MF42_DEFAULTS,
    maximum_curves: int = 12,
) -> Path:
    """Locate critical seeds and trace them with pseudo-arclength continuation."""
    plane = scan["plane"]
    seeds = _candidate_critical_seeds_42(scan, settings)[:maximum_curves]
    rows: list[dict[str, object]] = []
    step = 0.006 if plane.name != "panel_b_alpha2_dc" else 0.008
    for curve_id, seed in enumerate(seeds):
        backward = _trace_direction_42(seed, -1.0, plane, settings, step, 260)
        forward = _trace_direction_42(seed, 1.0, plane, settings, step, 260)
        curve = list(reversed(backward[1:])) + forward
        for order, vector in enumerate(curve):
            state, x_value, y_value = _unpack_critical_42(vector)
            parameters = plane.parameters(x_value, y_value)
            critical, _ = _critical_multiplier_42(state, parameters, settings)
            b_value, s_value = bimodality_and_mean_42(state[2:])
            rows.append(
                {
                    "curve_id": curve_id,
                    "order": order,
                    plane.x_parameter: x_value,
                    plane.y_parameter: y_value,
                    "event": _critical_event_42(state, parameters, settings),
                    "critical_multiplier_minus_one": critical,
                    "B": b_value,
                    "S": s_value,
                    "state_vector": json.dumps(state.tolist()),
                }
            )
        print(f"{plane.name}: continued critical curve {curve_id + 1}/{len(seeds)}")
    path = Path(output_dir) / f"{plane.name}_boundaries.csv"
    if rows:
        _write_dict_rows_42(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("curve_id,order,event\n")
    return path


def plot_generated_plane_42(
    plane: MF42Plane,
    data_dir: Path | str = MF42_GENERATED_DIR,
    figure_dir: Path | str = MF42_GENERATED_FIGURE_DIR,
) -> Path:
    """Plot a freshly calculated state grid and its continued boundaries."""
    data_dir = Path(data_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{plane.name}_grid.csv").open() as handle:
        grid_rows = list(csv.DictReader(handle))
    state_order = [
        "BP-BU coexistence",
        "Biased unimodal",
        "BP-NU coexistence",
        "Bimodal polarized",
        "Neutral unimodal",
        "BU-NU coexistence",
        "Undetected",
    ]
    colors = ["#ef9a9a", "#ffe08a", "#c9b9e8", "#9f89d0", "#9ec9ee",
              "#2f5d8f", "#eeeeee"]
    code = {name: index for index, name in enumerate(state_order)}
    grid = np.full((len(plane.y_values), len(plane.x_values)), code["Undetected"])
    x_index = {round(float(v), 12): i for i, v in enumerate(plane.x_values)}
    y_index = {round(float(v), 12): i for i, v in enumerate(plane.y_values)}
    for row in grid_rows:
        ix = x_index[round(float(row[plane.x_parameter]), 12)]
        iy = y_index[round(float(row[plane.y_parameter]), 12)]
        grid[iy, ix] = code[row["state"]]

    fig, ax = plt.subplots(figsize=(5.2, 4.7), dpi=180)
    cmap = ListedColormap(colors)
    ax.pcolormesh(
        plane.x_values,
        plane.y_values,
        grid,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        shading="nearest",
    )
    boundary_path = data_dir / f"{plane.name}_boundaries.csv"
    if boundary_path.is_file():
        with boundary_path.open() as handle:
            boundary_rows = list(csv.DictReader(handle))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in boundary_rows:
            if row.get("event"):
                grouped[(row["curve_id"], row["event"])].append(row)
        for (_, event), rows in grouped.items():
            rows.sort(key=lambda row: int(row["order"]))
            color = SADDLE_COLOR if event == "Saddle-node" else PITCHFORK_COLOR
            ax.plot(
                [float(row[plane.x_parameter]) for row in rows],
                [float(row[plane.y_parameter]) for row in rows],
                color=color,
                linewidth=1.7,
                linestyle="--",
            )
    ax.set_xlabel({"alpha2": r"$\alpha_2$", "alpha3": r"$\alpha_3$"}[plane.x_parameter])
    ax.set_ylabel({"fs": r"$f_s$", "dc": r"$d_c$"}[plane.y_parameter])
    ax.set_xlim(float(plane.x_values[0]), float(plane.x_values[-1]))
    ax.set_ylim(float(plane.y_values[0]), float(plane.y_values[-1]))
    ax.grid(linestyle=":", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=colors[i], edgecolor="black", label=name)
               for i, name in enumerate(state_order[:-1])]
    ax.legend(handles=handles, fontsize=7.1, ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    path = figure_dir / f"{plane.name}.png"
    fig.savefig(path, dpi=350, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def recompute_all_42bin(
    output_dir: Path | str = MF42_GENERATED_DIR,
    figure_dir: Path | str = MF42_GENERATED_FIGURE_DIR,
    quick: bool = False,
    trace_boundaries: bool = True,
    settings: MF42Settings = MF42_DEFAULTS,
) -> list[dict[str, Path]]:
    """Recompute all three 42-bin parameter planes from the model equations."""
    results: list[dict[str, Path]] = []
    for plane in manuscript_planes_42bin(quick=quick):
        scan = scan_plane_42(plane, output_dir, settings)
        boundary_path = Path(output_dir) / f"{plane.name}_boundaries.csv"
        if trace_boundaries:
            boundary_path = continue_boundaries_42(
                scan,
                output_dir,
                settings,
                maximum_curves=2 if quick else 12,
            )
        figure_path = plot_generated_plane_42(plane, output_dir, figure_dir)
        results.append(
            {
                "grid": Path(scan["grid_path"]),
                "attractors": Path(scan["attractor_path"]),
                "boundaries": boundary_path,
                "figure": figure_path,
            }
        )
    return results


def _command_line_42bin() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="42-bin mean-field calculation")
    parser.add_argument(
        "--recompute", action="store_true",
        help="calculate the three parameter planes from the equations",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="use a 5 x 5 test grid instead of the manuscript grids",
    )
    parser.add_argument(
        "--no-continuation", action="store_true",
        help="skip pseudo-arclength continuation",
    )
    args = parser.parse_args()
    if args.recompute:
        recompute_all_42bin(
            quick=args.quick,
            trace_boundaries=not args.no_continuation,
        )
    else:
        check_data()
        print("All archived 42-bin manuscript data are present.")


if __name__ == "__main__":
    _command_line_42bin()
