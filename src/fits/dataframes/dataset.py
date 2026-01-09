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
    observed_data: torch.Tensor  # [L, K] float
    observed_mask: torch.Tensor  # [L, K] int
    # 0 - not observed
    # 1 - observed (ground truth)
    forecast_mask: torch.Tensor  # [L, K] int
    # 0 - context (history)
    # 1 - horizon to forecast (to be generated)
    time_points: torch.Tensor  # [L, K] float
    feature_ids: torch.Tensor  # [L, K] float


@dataclass
class NormalizationStats:
    min: torch.Tensor
    max: torch.Tensor

    def as_device(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> "NormalizationStats":
        return NormalizationStats(
            min=self.min.to(device=device, dtype=dtype),
            max=self.max.to(device=device, dtype=dtype),
        )


class ForecastingDataset(Dataset[ForecastingData], ABC):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int,  # T
        horizon: int,  # last H observations to forecast
        dim: int,  # K
        train_share: float = 0.75,
        validation_share: float = 0.1,
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
        train_share: float = 0.75,
        validation_share: float = 0.1,
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

        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        observed_data = self.data[start_idx:end_idx]
        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
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

        feature_count = observed_mask.sum(axis=0)

        masked_min = np.where(observed_mask, observed_values, np.inf)
        masked_max = np.where(observed_mask, observed_values, -np.inf)

        feature_min = masked_min.min(axis=0)
        feature_max = masked_max.max(axis=0)

        empty_features = feature_count == 0
        feature_min[empty_features] = 0.0
        feature_max[empty_features] = 0.0

        return NormalizationStats(
            min=torch.from_numpy(feature_min.astype(np.float32)),
            max=torch.from_numpy(feature_max.astype(np.float32)),
        )

    def _normalize(self, window_data: torch.Tensor) -> torch.Tensor:
        if self.normalization_stats is None:
            return window_data

        stats = self.normalization_stats
        if stats.min.device != window_data.device:
            stats = stats.as_device(window_data.device, window_data.dtype)

        center = (stats.max + stats.min) / 2
        scale = (stats.max - stats.min) / 2
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        return (window_data - center) / scale


class DatasetSolar(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.95,
        validation_share: float = 0.02,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        n_features: int = 128,  # max=137, but default=128 features to match published setups
    ) -> None:
        dataset_path = DatasetsPaths.solar.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Solar dataset not found. Run fits.data.download.DownloadDatasetSolar() first."
            )

        df = pd.read_csv(dataset_path, header=None)
        if df.shape[1] > 0 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        df = df.iloc[:, (137-n_features):]
        values = df.to_numpy(dtype=np.float32)
        mask = ~np.isnan(values)
        np.nan_to_num(values, nan=0.0, copy=False)

        dim = values.shape[1]

        super().__init__(
            mode=mode,
            seq_len=seq_len,
            horizon=horizon,
            dim=dim,
            train_share=train_share,
            validation_share=validation_share,
        )

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

        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        observed_data = self.data[start_idx:end_idx]
        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
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

        feature_count = observed_mask.sum(axis=0)

        masked_min = np.where(observed_mask, observed_values, np.inf)
        masked_max = np.where(observed_mask, observed_values, -np.inf)

        feature_min = masked_min.min(axis=0)
        feature_max = masked_max.max(axis=0)

        empty_features = feature_count == 0
        feature_min[empty_features] = 0.0
        feature_max[empty_features] = 0.0

        return NormalizationStats(
            min=torch.from_numpy(feature_min.astype(np.float32)),
            max=torch.from_numpy(feature_max.astype(np.float32)),
        )

    def _normalize(self, window_data: torch.Tensor) -> torch.Tensor:
        if self.normalization_stats is None:
            return window_data

        stats = self.normalization_stats
        if stats.min.device != window_data.device:
            stats = stats.as_device(window_data.device, window_data.dtype)

        center = (stats.max + stats.min) / 2
        scale = (stats.max - stats.min) / 2
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        return (window_data - center) / scale
