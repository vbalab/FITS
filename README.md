# FITS: Flow-based Interpretable Time-Series

FITS is a modelling framework for **flow-based, interpretable vector time-series forecasting**.

Research codebase includes multiple forecasting model families (**FITS/FITSJ** as primary models, and also _CSDI, DiffusionTS, FMTS, VAR_ as baselines), dataset helpers, and training/evaluation utilities.

## Features

- **Dataset utilities** for popular forecasting benchmarks (solar energy, ETT,
  electricity, exchange rate, and weather).
- **Model zoo** covering FITS/FITSJ, diffusion-based approaches, and baselines like
  VAR.
- **Training and evaluation helpers** with EMA support, learning-rate scheduling, and
  checkpoint organization.
- **Notebook examples** for running experiments and plotting results.

## Project structure

```txt
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

- `DownloadDatasetSolar()`
- `DownloadDatasetETTh()`
- `DownloadDatasetElectricity()`
- `DownloadDatasetExchange()`
- `DownloadDatasetWeather()`

Example:

```python
from fits.dataframes.download import DownloadDatasetElectricity

DownloadDatasetElectricity()
```

## Training example

Below is a minimal sketch showing how to create loaders and train a model.

```python
from fits.dataframes.dataloader import ForecastingDataLoader
from fits.dataframes.dataset import DatasetElectricity
from fits.modelling.FITS.model import FITS, FITSConfig
from fits.modelling.framework import Train

train_loader, valid_loader, test_loader = ForecastingDataLoader(
    DatasetElectricity,
    batch_size=128,
    seq_len=96,
    horizon=24,
)

model = FITS(FITSConfig())
Train(model, train_loader, valid_loader, epochs=50)
```

## Notes

- Model training currently assumes CUDA availability (`torch.cuda.is_available()`).
- Data and model outputs are stored under `data/` in the repository root.
