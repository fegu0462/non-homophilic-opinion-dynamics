# Code and data for the opinion-dynamics study

This repository contains three model modules and five notebooks.

## Core modules

- `src/monte_carlo.py`: microscopic Monte Carlo dynamics and all initial conditions.
- `src/mean_field_4bin.py`: four-bin fixed points, stability, bifurcation detection, and plotting.
- `src/mean_field_42bin.py`: validation and plotting of the 42-bin stability-scan and continuation data.

## Notebooks

1. `01_trajectory_states_K40.ipynb`: three representative Monte Carlo states.
2. `02_coexisting_states_K40_T50.ipynb`: coexisting states from different initial conditions.
3. `03_check_42bin_mean_field_data.ipynb`: check all supplied 42-bin numerical data.
4. `04_plot_42bin_parameter_planes.ipynb`: generate the three 42-bin parameter-plane panels.
5. `05_compute_plot_4bin_bifurcation_slices.ipynb`: recompute and plot both four-bin slices.

## Installation

Python 3.9 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

Open the repository folder in JupyterLab or VS Code, select the same Python
environment, and run a notebook from top to bottom. Generated figures are
written to `figures/`.

The 42-bin parameter planes use the numerical results in
`data/recomputed_exact/`. The four-bin notebook recomputes its fixed points
and bifurcations directly from the equations.
