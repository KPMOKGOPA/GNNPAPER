# GNNpaper Project

## Project overview

This repository contains data extraction, preprocessing, graph generation, and graph neural network training code for reaction path/IRC analysis.

## Main components

- `data_prop/data_extraction.py`
  - Parses Gaussian IRC log files
  - Extracts forward/reverse/TS geometries, forces, distance matrices, energies, and reaction coordinates
  - Saves structured data to an HDF5 file

- `data_prop/data_prep.py`
  - Reads the HDF5 dataset (`all_logs.h5`)
  - Computes geometry distance matrices and related reaction metrics
  - Builds a DataFrame of reactant/TS/product structures and forces

- `data_prop/graphs_generation.py`
  - Builds PyTorch Geometric graphs from geometry and force arrays
  - Contains `ReactionGraphDataset` used by training code

- `models/models.py`
  - Defines the GNN architecture (`ReactionGNN`)
  - Implements training and evaluation functions
  - Imports `ReactionGraphDataset` from `data_prop/graphs_generation.py`

- `models/run_models.py`
  - Entry-point script for training and evaluating the model
  - Imports model and training utilities from `models/models.py`

- `models/analysis.py`
  - Evaluation and plotting helper functions for model results

## How to use

1. Prepare the HDF5 data using `data_prop/data_extraction.py`.
2. Process the data and build a DataFrame in `data_prop/data_prep.py`.
3. Generate graphs with `data_prop/graphs_generation.py`.
4. Train the model by editing `models/run_models.py` to load `df_main` and then run:

```bash
python models/run_models.py
```

## Notes

- `models/models.py` now imports the shared dataset class from `data_prop/graphs_generation.py`.
- `models/run_models.py` is structured as a main entry point with a placeholder for `df_main`.
- `models/analysis.py` includes required plotting and torch imports.

## Dependencies

Common dependencies include:

- Python 3.8+
- `numpy`
- `pandas`
- `h5py`
- `scipy`
- `torch`
- `torch-geometric`
- `scikit-learn`
- `matplotlib`
# GNNPAPER
