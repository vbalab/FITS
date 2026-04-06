"""Train and evaluate all models on all datasets.

Usage:
    python -m fits
"""


import os

GPUs = [
    "GPU-e83bd31b-fcb9-b8de-f617-2d717619413b",
    "GPU-5a9b7750-9f85-49a5-3aae-fe07b1b7661d",
    "GPU-fe2d8dfd-06f2-a5c4-a7fd-4a5f23947005",
    "GPU-0c320096-21ee-4060-8731-826ca2febfab",
    "GPU-baef952c-6609-aace-3b78-e4e07788d5de",
    "GPU-3979d65b-c238-4e9c-0c1c-1aa3f05c56a1",
    "GPU-6c76a2c5-5375-aa06-11d4-0fddfac30e91",
]
os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUs[0]}"


# -----

import logging  # noqa: E402

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
# Models:  factory that accepts (feature_size, first_differences) and returns
# a fresh model instance.
# ---------------------------------------------------------------------------
MODEL_FACTORIES = {
    "csdi":          lambda k, fd: CSDIAdapter(CSDIConfig(feature_size=k, first_differences=fd)),
    "diffusion_ts":  lambda k, fd: DiffusionTSAdapter(DiffusionTSConfig(feature_size=k, first_differences=fd)),
    "timegrad":      lambda k, fd: TimeGradAdapter(TimeGradConfig(feature_size=k, first_differences=fd)),
    "patchtst":      lambda k, fd: PatchTSTAdapter(PatchTSTConfig(feature_size=k, first_differences=fd)),
    "itransformer":  lambda k, fd: ITransformerAdapter(ITransformerConfig(feature_size=k, first_differences=fd)),
    "sssd":          lambda k, fd: SSSDAdapter(SSSDConfig(feature_size=k, first_differences=fd)),
    "mrdiff":        lambda k, fd: MrDiffAdapter(MrDiffConfig(feature_size=k, first_differences=fd)),
    "nsdiff":        lambda k, fd: NsDiffAdapter(NsDiffConfig(feature_size=k, first_differences=fd)),
    "falda":         lambda k, fd: FALDAAdapter(FALDAConfig(feature_size=k, first_differences=fd)),
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


def _make_combos() -> list:
    return [
        (dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory)
        for dataset_name, (dataset_cls, feature_size, ds_kwargs, _) in DATASETS.items()
        for model_name, factory in MODEL_FACTORIES.items()
    ]


def _run_combos(combos: list, first_differences: bool, epochs: int, batch_size: int, nsample: int) -> None:
    fd_suffix = "_fd" if first_differences else ""
    for dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory in tqdm(
        combos, desc=f"runs (fd={first_differences})", unit="run"
    ):
        folder = f"{model_name}_{dataset_name}{fd_suffix}"
        tqdm.write(f"▶ {model_name} on {dataset_name} (first_differences={first_differences})")

        model = factory(feature_size, first_differences)
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


def _run_all(
    epochs: int = 200,
    batch_size: int = 128,
    nsample: int = 10,
) -> None:
    _ensure_datasets()

    combos = _make_combos()
    _run_combos(combos, first_differences=False, epochs=epochs, batch_size=batch_size, nsample=nsample)

    combos_fd = _make_combos()
    _run_combos(combos_fd, first_differences=True, epochs=epochs, batch_size=batch_size, nsample=nsample)


if __name__ == "__main__":
    _run_all()
