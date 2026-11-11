"""42-bin alpha3--fs stable-attractor scan using corrected sample BC."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["BIMODALITY_MODE"] = "sample_corrected"
os.environ["B_CUTOFF"] = "0.56"

import numpy as np

import compute_plot_42bin_alpha3_fs as source


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "alpha3_fs_42bin_stable_attractors_fs020_090_sample_corrected.csv"


def main() -> None:
    source.FS_MIN = 0.20
    source.FS_MAX = 0.90
    source.FS_VALUES = np.round(np.arange(0.20, 0.9001, 0.01), 12)
    source.ATTRACTOR_CSV = OUTPUT
    source.run_stable_scan(max(1, min(8, os.cpu_count() or 1)))


if __name__ == "__main__":
    main()
