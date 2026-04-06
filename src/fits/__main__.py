"""Train and evaluate all models on all datasets.

Usage:
    python -m fits
"""

import logging

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")  # non-interactive backend — safe for scripts

from fits.dataframes.dataset import (
    DatasetElectricity,
    DatasetETTh,
    DatasetExchange,
    DatasetWeather,
)
from fits.dataframes.download import (
    DownloadDatasetElectricity,
    DownloadDatasetETTh,
    DownloadDatasetExchange,
    DownloadDatasetWeather,
)
from fits.modelling.framework import TrainAndEvaluate
from fits.modelling.models.CSDI import CSDIAdapter, CSDIConfig
from fits.modelling.models.DiffusionTS import (
    DiffusionTSAdapter,
    DiffusionTSConfig,
)
from fits.modelling.models.FALDA import FALDAAdapter, FALDAConfig
from fits.modelling.models.iTransformer import (
    ITransformerAdapter,
    ITransformerConfig,
)
from fits.modelling.models.mrDiff import MrDiffAdapter, MrDiffConfig
from fits.modelling.models.NsDiff import NsDiffAdapter, NsDiffConfig
from fits.modelling.models.PatchTST import PatchTSTAdapter, PatchTSTConfig
from fits.modelling.models.SSSD import SSSDAdapter, SSSDConfig
from fits.modelling.models.TimeGrad import TimeGradAdapter, TimeGradConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datasets:  name → (dataset_cls, feature_size, extra_dataset_kwargs)
#
# For datasets with an n_features parameter the value here controls both
# how many columns the dataset loads AND the feature_size passed to models.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Each entry: dataset_cls, feature_size, extra_kwargs, download_fn
# ---------------------------------------------------------------------------
DATASETS = {
    "etth":        (DatasetETTh,        7,  {},               DownloadDatasetETTh),
    "electricity": (DatasetElectricity, 32, {"n_features": 32}, DownloadDatasetElectricity),  # noqa: E501
    "exchange":    (DatasetExchange,    8,  {},               DownloadDatasetExchange),
    "weather":     (DatasetWeather,     21, {},               DownloadDatasetWeather),
}

# ---------------------------------------------------------------------------
# Models:  factory that accepts feature_size and returns a fresh model instance
# ---------------------------------------------------------------------------
MODEL_FACTORIES = {
    "csdi":          lambda k: CSDIAdapter(CSDIConfig(feature_size=k)),
    "diffusion_ts":  lambda k: DiffusionTSAdapter(DiffusionTSConfig(feature_size=k)),
    "timegrad":      lambda k: TimeGradAdapter(TimeGradConfig(feature_size=k)),
    "patchtst":      lambda k: PatchTSTAdapter(PatchTSTConfig(feature_size=k)),
    "itransformer":  lambda k: ITransformerAdapter(ITransformerConfig(feature_size=k)),
    "sssd":          lambda k: SSSDAdapter(SSSDConfig(feature_size=k)),
    "mrdiff":        lambda k: MrDiffAdapter(MrDiffConfig(feature_size=k)),
    "nsdiff":        lambda k: NsDiffAdapter(NsDiffConfig(feature_size=k)),
    "falda":         lambda k: FALDAAdapter(FALDAConfig(feature_size=k)),
}


def _ensure_datasets() -> None:
    """Download any datasets whose files are not yet present."""
    from fits.dataframes.dataset import ModelMode  # noqa: PLC0415

    for name, (dataset_cls, _, ds_kwargs, download_fn) in DATASETS.items():
        try:
            dataset_cls(mode=ModelMode.train, **ds_kwargs)
        except FileNotFoundError:
            log.info("Downloading %s dataset...", name)
            download_fn()
            log.info("  %s downloaded.", name)


def _run_all(
    epochs: int = 500,
    batch_size: int = 128,
    nsample: int = 10,
) -> None:
    _ensure_datasets()

    combos = [
        (dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory)
        for dataset_name, (dataset_cls, feature_size, ds_kwargs, _) in DATASETS.items()
        for model_name, factory in MODEL_FACTORIES.items()
    ]

    for dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory in tqdm(
        combos, desc="runs", unit="run"
    ):
        folder = f"{model_name}_{dataset_name}"
        tqdm.write(f"▶ {model_name} on {dataset_name}")

        model = factory(feature_size)
        TrainAndEvaluate(
            model=model,
            dataset_cls=dataset_cls,
            batch_size=batch_size,
            epochs=epochs,
            dataset_kwargs=ds_kwargs,
            nsample=nsample,
            verbose=False,
            folder_name=folder,
        )
        tqdm.write(f"  ✓ done → {folder}")


if __name__ == "__main__":
    _run_all()
