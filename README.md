# FITS: Flow-based Interpretable Time-Series

FITS is a research codebase for **flow-based, interpretable time-series forecasting**. It
includes multiple forecasting model families (FITS/FITSJ, CSDI, DiffusionTS, FMTS, and
VAR baselines), dataset helpers, and training/evaluation utilities built around PyTorch.

## Features

- **Dataset utilities** for popular forecasting benchmarks (air quality, PhysioNet,
  solar energy, and ETT).
- **Model zoo** covering FITS/FITSJ, diffusion-based approaches, and baselines like
  VAR.
- **Training and evaluation helpers** with EMA support, learning-rate scheduling, and
  checkpoint organization.
- **Notebook examples** for running experiments and plotting results.

## Project structure

```
src/fits/
  config.py              # paths + global seeding helpers
  dataframes/            # datasets, dataloaders, download utilities
  modelling/             # model implementations and training framework
  notebooks/             # example Jupyter notebooks
```

## Installation

This repository targets Python 3.13 and expects a CUDA-capable GPU (training will raise
an error if CUDA is unavailable).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Dataset utilities live in `fits.dataframes.download`. The following helpers download
and unpack benchmark data to `data/datasets/`:

- `DownloadDatasetAirQuality()`
- `DownloadDatasetPhysio()`
- `DownloadDatasetSolar()`
- `DownloadDatasetETTh()`

Example:

```python
from fits.dataframes.download import DownloadDatasetAirQuality

DownloadDatasetAirQuality()
```

## Training example

Below is a minimal sketch showing how to create loaders and train a model.

```python
from fits.dataframes.dataloader import ForecastingDataLoader
from fits.dataframes.dataset import DatasetAirQuality
from fits.modelling.FITS.model import FITS, FITSConfig
from fits.modelling.framework import Train

train_loader, valid_loader, test_loader = ForecastingDataLoader(
    DatasetAirQuality,
    batch_size=128,
    seq_len=48,
    horizon=6,
)

model = FITS(FITSConfig())
Train(model, train_loader, valid_loader, epochs=50)
```

## Notes

- Model training currently assumes CUDA availability (`torch.cuda.is_available()`).
- Data and model outputs are stored under `data/` in the repository root.
