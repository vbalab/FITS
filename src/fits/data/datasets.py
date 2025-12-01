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
        horizon: int,  # last H observations to forecast
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

        dataset_path = DatasetPaths.pm25.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Air Quality dataset not found. Run fits.data.download.DownloadDatasetAirQuality() first."
            )

        df = pd.read_csv(dataset_path, index_col="datetime", parse_dates=True)
        df = df.sort_index()

        values = df.to_numpy(dtype=np.float32)
        mask = ~np.isnan(values)
        values = np.nan_to_num(values, nan=0.0)

        self.data = torch.from_numpy(values)
        self.mask = torch.from_numpy(mask.astype(np.float32))

        num_sequences = len(df) - self.seq_len + 1
        assert num_sequences > 0, "Not enough time steps for the requested sequence length"

        train_end = int(num_sequences * 0.7)
        valid_end = int(num_sequences * 0.85)

        if mode is ModelMode.train:
            self.start = 0
            self.end = train_end
        elif mode is ModelMode.valid:
            self.start = train_end
            self.end = valid_end
        else:
            self.start = valid_end
            self.end = num_sequences

        self.time_points = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1).repeat(1, self.dim)
        self.feature_ids = torch.arange(self.dim, dtype=torch.float32).unsqueeze(0).repeat(self.seq_len, 1)

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        window_data = self.data[start_idx:end_idx]
        window_mask = self.mask[start_idx:end_idx]

        observed_mask = window_mask.clone()
        if self.horizon > 0:
            observed_mask[-self.horizon :] = 0

        observed_data = window_data.clone()
        observed_data = observed_data * observed_mask

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
        )

    def __len__(self) -> int:
        return self.end - self.start
