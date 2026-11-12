# Selected code and data for Mean-filed approximation and Monte Carlo simulations

This repository contains the code and data used in our study of Non-homophilic interactions and phase transitions in opinion dynamics.

The source Python code is provided in `src/`. The five Jupyter notebooks in
`notebooks/` reproduce the Monte Carlo trajectories, the four-bin mean-field
bifurcation slices, and the 42-bin mean-field parameter-plane diagrams.
Precomputed data used for the 42-bin figures are provided in `data/`; the same
data can also be recomputed from the model equations using
`src/mean_field_42bin.py`. Generated figures are written to `figures/`.

Python 3.9 or later is recommended. Install the required packages with

```bash
python -m pip install -r requirements.txt
```

Then open the repository in JupyterLab or VS Code and run the notebooks from
top to bottom. To perform a quick test of the complete 42-bin calculation, run

```bash
python src/mean_field_42bin.py --recompute --quick
```

To run the full 42-bin calculation, use

```bash
python src/mean_field_42bin.py --recompute
```
