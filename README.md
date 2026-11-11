# Non-homophilic interactions and phase transitions in opinion dynamics

This repository contains the code and pre-computed data used in the manuscript  
**“Non-homophilic interactions and phase transitions in opinion dynamics”**  
by Feng Gu and Eduardo G. Altmann.

The repository provides Monte Carlo simulations of the coevolving opinion model,
four-bin and 42-bin mean-field calculations, and the code used to generate the
main trajectory and bifurcation figures in the manuscript.

## Repository structure

- `notebooks/`: Jupyter notebooks used to reproduce the figures.
- `src/`: Python source code for the Monte Carlo and mean-field models.
- `exact_data_generation/`: scripts used to generate the 42-bin mean-field data.
- `data/recomputed_exact/`: pre-computed data used by the plotting notebooks.
- `figures/`: output directory for generated figures.

## Notebooks

### Monte Carlo simulations

- `01_trajectory_states_K40_final.ipynb`  
  Generates representative trajectories and terminal opinion distributions for
  the bimodal polarized, biased unimodal, and neutral unimodal states.

- `02_three_states_K40_T50_complete.ipynb`  
  Generates trajectories and terminal distributions for three coexisting states
  reached from different initial conditions: bimodal polarized, biased unimodal,
  and asymmetric bimodal polarized.

### Mean-field calculations

- `03_mean_field_42bin_exact_data_generation.ipynb`  
  Recomputes the data for the three two-parameter bifurcation diagrams of the
  42-bin mean-field model. This calculation can be computationally intensive.

- `04_mean_field_42bin_plot_three_panels.ipynb`  
  Reads the pre-computed 42-bin data and generates the three two-parameter
  bifurcation diagrams. This notebook can be run without recomputing the data.

- `05_mean_field_4bin_two_slices_full_reproduction.ipynb`  
  Recomputes and plots the two one-parameter bifurcation diagrams of the
  four-bin mean-field model.

## Installation

The calculations were tested with Python 3.9.20.

To create a suitable environment, run

```bash
conda create -n opinion-dynamics python=3.9.20
conda activate opinion-dynamics
pip install -r requirements.txt
