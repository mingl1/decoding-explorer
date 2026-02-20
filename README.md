# Decoding Explorer

## Root Layout

- `main.py`, `align_arrays.py`, `image_processing.py`, `utils.py`: core runtime modules.
- `model/`, `view/`, `viewmodel/`: MVVM application code.
- `tests/`: automated tests.
- `assets/`: UI and packaged application assets.
- `notebooks/`: exploratory and workflow notebooks.
- `data/raw/`: input/reference datasets.
- `data/results/`: generated outputs and result tables.
- `tmp/`: local scratch/debug/cache artifacts.

## Keep Root Clean

- Put new notebooks in `notebooks/`.
- Put newly produced CSV/TIF outputs in `data/results/`.
- Put ad-hoc debug files in `tmp/`.
