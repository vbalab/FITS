from dataclasses import dataclass

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.FITS.transformer import ...
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class FITSConfig(ModelConfig):
    ...

class FITSModel(ForecastingModel):
    ...
