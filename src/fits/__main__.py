"""Train and evaluate all models on all datasets.

Usage:
    python -m fits
"""

import logging
import traceback

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")  # non-interactive backend — safe for scripts

from fits.dataframes.dataset import (
    DatasetElectricity,
    DatasetETTh,
    DatasetExchange,
    DatasetWeather,
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
DATASETS = {
    "etth":        (DatasetETTh,        7,  {}),
    "electricity": (DatasetElectricity, 32, {"n_features": 32}),
    "exchange":    (DatasetExchange,    8,  {}),
    "weather":     (DatasetWeather,     21, {}),
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


def _run_all(
    epochs: int = 500,
    batch_size: int = 128,
    nsample: int = 10,
) -> None:
    combos = [
        (dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory)
        for dataset_name, (dataset_cls, feature_size, ds_kwargs) in DATASETS.items()
        for model_name, factory in MODEL_FACTORIES.items()
    ]

    for dataset_name, dataset_cls, feature_size, ds_kwargs, model_name, factory in tqdm(
        combos, desc="runs", unit="run"
    ):
        folder = f"{model_name}_{dataset_name}"
        tqdm.write(f"▶ {model_name} on {dataset_name}")

        try:
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
        except Exception:  # noqa: BLE001
            tqdm.write(f"  ✗ FAILED: {model_name} on {dataset_name}")
            traceback.print_exc()


if __name__ == "__main__":
    _run_all()
