import torch
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from fits.config import DatasetPaths


class ModelMode(Enum):
    train = 0
    valid = 1
    test = 2


@dataclass
class ForecastingData:
    observed_data: torch.Tensor  # [T, K] float
    observed_mask: torch.Tensor  # [T, K] int
    time_points: torch.Tensor  # [T, K] float
    feature_ids: torch.Tensor  # [T, K] float


class ForecastingDataset(Dataset, ABC):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int,  # T
        horizon: int,  # lasn H observations to forecast
        dim: int,  # K
    ) -> None:
        assert horizon < seq_len, f"Horizon={horizon} >= seq_len={seq_len}"

        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.dim = dim

    @abstractmethod
    def __getitem__(self, index: int) -> ForecastingData: ...

    @abstractmethod
    def __len__(self) -> int: ...


class DatasetAirQuality(ForecastingDataset):
    def __init__(self, mode: ModelMode, seq_len: int = 48, horizon: int = 6) -> None:
        super().__init__(mode=mode, seq_len=seq_len, horizon=horizon, dim=36)

        df = pd.read_csv(
            DatasetPaths.pm25.value,
            index_col="datetime",
            parse_dates=True,
        )

        ...
