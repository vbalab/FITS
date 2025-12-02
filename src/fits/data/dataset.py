import torch
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from fits.config import DatasetsPaths


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
    conditioning_mask: torch.Tensor | None = None  # [T, K] int


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor

    def as_device(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> "NormalizationStats":
        return NormalizationStats(
            mean=self.mean.to(device=device, dtype=dtype),
            std=self.std.to(device=device, dtype=dtype),
        )


class ForecastingDataset(Dataset[ForecastingData], ABC):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int,  # T
        horizon: int,  # last H observations to forecast
        dim: int,  # K
        train_share: float = 0.7,
        validation_share: float = 0.15,
    ) -> None:
        assert horizon < seq_len, f"Horizon={horizon} >= seq_len={seq_len}"
        assert train_share + validation_share <= 1, "Incorrect train/validation shares"

        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.dim = dim
        self.train_share = train_share
        self.validation_share = validation_share

    @abstractmethod
    def __getitem__(self, index: int) -> ForecastingData: ...

    @abstractmethod
    def __len__(self) -> int: ...


class DatasetAirQuality(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 48,
        horizon: int = 6,
        train_share: float = 0.7,
        validation_share: float = 0.15,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            seq_len=seq_len,
            horizon=horizon,
            dim=36,
            train_share=train_share,
            validation_share=validation_share,
        )
        dataset_path = DatasetsPaths.pm25.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Air Quality dataset not found. Run fits.data.download.DownloadDatasetAirQuality() first."
            )

        df = pd.read_csv(dataset_path, index_col="datetime", parse_dates=True)
        df = df.sort_index()

        values = df.to_numpy(dtype=np.float32)
        mask = ~np.isnan(values)
        np.nan_to_num(values, nan=0.0, copy=False)

        self.data = torch.from_numpy(values)
        self.mask = torch.from_numpy(mask.astype(np.float32))

        num_sequences = len(df) - self.seq_len + 1
        assert (
            num_sequences > 0
        ), "Not enough time steps for the requested sequence length"

        train_end = int(num_sequences * train_share)
        valid_end = train_end + int(num_sequences * validation_share)

        if mode is ModelMode.train:
            self.start = 0
            self.end = train_end
        elif mode is ModelMode.valid:
            self.start = train_end
            self.end = valid_end
        else:
            self.start = valid_end
            self.end = num_sequences

        self.time_points = (
            torch.arange(self.seq_len, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim)
        )
        self.feature_ids = (
            torch.arange(self.dim, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )

        if normalization_stats is None and normalization:
            normalization_stats = self._compute_training_normalization(
                values=values,
                mask=mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        window_mask = self.mask[start_idx:end_idx]
        conditioning_mask = window_mask.clone()
        conditioning_mask[-self.horizon :] = 0

        window_data = self.data[start_idx:end_idx]
        window_data = self._normalize(window_data)
        observed_data = window_data

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=window_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            conditioning_mask=conditioning_mask,
        )

    def __len__(self) -> int:
        return self.end - self.start

    @staticmethod
    def _compute_training_normalization(
        values: np.ndarray,
        mask: np.ndarray,
        train_sequences: int,
        seq_len: int,
    ) -> NormalizationStats:
        last_index = train_sequences + seq_len - 1

        observed_values = values[:last_index]
        observed_mask = mask[:last_index]

        feature_sum = (observed_values * observed_mask).sum(axis=0)
        feature_count = observed_mask.sum(axis=0)

        mean = feature_sum / np.clip(feature_count, a_min=1, a_max=None)
        variance = (((observed_values - mean) ** 2) * observed_mask).sum(
            axis=0
        ) / np.clip(feature_count, a_min=1, a_max=None)

        std = np.sqrt(variance)
        std[std == 0] = 1.0

        return NormalizationStats(
            mean=torch.from_numpy(mean.astype(np.float32)),
            std=torch.from_numpy(std.astype(np.float32)),
        )

    def _normalize(self, window_data: torch.Tensor) -> torch.Tensor:
        if self.normalization_stats is None:
            return window_data

        stats = self.normalization_stats
        if stats.mean.device != window_data.device:
            stats = stats.as_device(window_data.device, window_data.dtype)

        return (window_data - stats.mean) / stats.std
