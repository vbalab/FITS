# CLAUDE.md

## Project overview

FITS (Flow-based Interpretable Time-Series) is a research codebase for flow-based, interpretable vector time-series forecasting. It implements multiple model families (FITS/FITSJ as primary, plus CSDI, DiffusionTS, FMTS, VAR baselines), dataset helpers for standard benchmarks, and a unified training/evaluation framework. This is an academic project targeting Python 3.13 with CUDA-capable GPUs.

## Repository layout

```
src/fits/
  config.py                    # Global paths (DATA_PATH, MODELS_PATH, etc.) and SeedEverything()
  dataframes/
    dataset.py                 # Dataset classes: DatasetSolar, DatasetETTh, DatasetElectricity, DatasetExchange, DatasetWeather
    dataloader.py              # ForecastingDataLoader() factory -> (train, valid, test) loaders
    download.py                # DownloadDataset*() helpers that fetch benchmarks to data/datasets/
  modelling/
    framework.py               # ForecastingModel ABC, ModelConfig, ForecastedData, Train(), Evaluate(), EMA
    comparison.py              # Visualization: PCA, t-SNE, KDE density, forecast sample plots
    FITS/                      # Primary flow-based model (FITSModel, FITSConfig)
    FITSJ/                     # FITS with trend/seasonality/jump decomposition
    CSDI/                      # Conditional Score-based Diffusion (adapter pattern)
    DiffusionTS/               # Transformer-based diffusion (adapter pattern)
    FMTS/                      # Fourier-based interpretable diffusion (adapter pattern)
    VAR/                       # Seasonal Vector AutoRegression baseline
  notebooks/
    etth.ipynb                 # ETTh benchmark experiments
    solar.ipynb                # Solar energy benchmark experiments
```

Data and outputs are stored under `data/` (gitignored):
- `data/datasets/` -- downloaded benchmark CSVs/files
- `data/models/training/` -- checkpoints organized by timestamp
- `data/models/evaluation/` -- pickled evaluation results

## Setup and dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, pandas, torch, requests, tqdm
pip install -r requirements-dev.txt    # jupyter, black, ruff, mypy, type stubs
```

Requires **Python 3.13** and a **CUDA GPU** (ForecastingModel raises RuntimeError if CUDA is unavailable).

## Code quality commands

```bash
black src                    # Format (line-length 88, target py313)
ruff check src --fix         # Lint (E, F, B, I, UP, N, S, C4, DTZ, T20, Q, PL)
mypy src                     # Type check (strict mode, pydantic plugin)
```

There is no test suite or Makefile. Validation is done through Jupyter notebooks and mypy strict mode.

## Code conventions

- **Naming**: PascalCase for classes and public functions (e.g. `Train()`, `Evaluate()`, `SeedEverything()`), `_underscore_prefix` for private methods, `UPPER_SNAKE` for module-level constants
- **Type hints**: Full annotations everywhere; use modern union syntax (`int | None`, not `Optional[int]`)
- **Config**: Dataclasses inheriting from `ModelConfig` -- no loose dicts
- **Imports**: stdlib, then third-party, then local (`fits.*`); isort with combine-as-imports and split-on-trailing-comma
- **Formatting**: Black with default 88-char lines; ruff handles the rest
- **Linting ignores**: E501 (line length), S101 (assert), N802 (uppercase function names -- intentional project convention)

## Architecture patterns

- **Abstract base**: All models extend `ForecastingModel(nn.Module, ABC)` and implement `forward()` and `evaluate()`
- **Adapter pattern**: Diffusion models (CSDI, DiffusionTS, FMTS) wrap third-party implementations via `*Adapter` classes that conform to the `ForecastingModel` interface
- **Data contract**: `ForecastingData` dataclass flows through the pipeline -- tensors shaped `[B, L, K]` (batch, time, features) with companion masks
- **Evaluation output**: `ForecastedData` with Monte-Carlo samples shaped `[B, nsample, L, K]`
- **Training loop**: `Train(model, train_loader, valid_loader, epochs)` handles optimizer setup, LR scheduling (warmup + multistep), gradient clipping, optional EMA, and checkpoint saving
- **Evaluation loop**: `Evaluate(model, test_loader)` returns pickled `ForecastedData` with quantile-based CRPS metrics
- **Normalization**: Min-max per feature, computed on training split, applied globally

## Key data shapes

- Input tensors: `[B, L, K]` where B=batch, L=sequence length, K=features
- Masks: binary `[B, L, K]` (1=observed, 0=missing)
- Time points: `[B, L]` float
- Forecast samples: `[B, nsample, L, K]`

## Adding a new model

1. Create a directory under `src/fits/modelling/YourModel/`
2. Define a config dataclass extending `ModelConfig`
3. Implement a class extending `ForecastingModel` with `forward()` and `evaluate()` methods
4. The model must accept `ForecastingData` batches and return loss from `forward()`, `ForecastedData` from `evaluate()`

## Adding a new dataset

1. Subclass `ForecastingDataset` in `src/fits/dataframes/dataset.py`
2. Implement normalization, masking, and train/valid/test splitting
3. Add a download helper in `download.py` if the data is remote
4. Add the path to `DatasetsPaths` enum in `config.py`

## Common pitfalls

- Training **requires CUDA** -- CPU-only environments will fail at model init
- Dataset files must be downloaded first via `DownloadDataset*()` before creating dataset objects
- The `data/` directory is gitignored; checkpoints and datasets are local-only
- `print()` statements are flagged by ruff (T20 rule) -- use logging or suppress with `# noqa: T20` only when intentional (e.g. `SeedEverything`)
- Notebooks are the primary way to run experiments; there is no CLI entrypoint
