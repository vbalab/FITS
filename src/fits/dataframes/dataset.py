from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
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
    base_level: torch.Tensor | None = (
        None  # [K] float — raw first value (first-differences only)
    )


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
        first_differences: bool = False,
    ) -> None:
        assert horizon < seq_len, f"Horizon={horizon} >= seq_len={seq_len}"
        assert train_share + validation_share <= 1, "Incorrect train/validation shares"

        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.dim = dim
        self.train_share = train_share
        self.validation_share = validation_share
        self.first_differences = first_differences

    @abstractmethod
    def __getitem__(self, index: int) -> ForecastingData: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @staticmethod
    def _apply_first_differences(
        values: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute first differences and adjust the observation mask.

        Returns ``(diff_values, diff_mask)`` where::

            diff_values[0]  = 0
            diff_values[t]  = values[t] - values[t-1]   for t >= 1

            diff_mask[0]    = mask[0]
            diff_mask[t]    = mask[t] & mask[t-1]        for t >= 1
        """
        diff_values = np.zeros_like(values)
        diff_values[1:] = values[1:] - values[:-1]

        diff_mask = np.zeros_like(mask)
        diff_mask[0] = mask[0]
        diff_mask[1:] = mask[1:] * mask[:-1]

        return diff_values, diff_mask


class DatasetSolar(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.9,
        validation_share: float = 0.05,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        n_features: int = 128,  # max=137, but default=128 features to match published setups
        first_differences: bool = False,
    ) -> None:
        dataset_path = DatasetsPaths.solar.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Solar dataset not found. Run fits.data.download.DownloadDatasetSolar() first."
            )

        df = pd.read_csv(dataset_path, header=None)
        if df.shape[1] > 0 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        df = df.iloc[:, (137 - n_features) :]
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
            first_differences=first_differences,
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
            norm_values, norm_mask = values, mask
            if first_differences:
                norm_values, norm_mask = self._apply_first_differences(values, mask)
            normalization_stats = self._compute_training_normalization(
                values=norm_values,
                mask=norm_mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        observed_data = self.data[start_idx:end_idx]
        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        base_level: torch.Tensor | None = None
        if self.first_differences:
            base_level = observed_data[0].clone()
            diffs = torch.zeros_like(observed_data)
            diffs[1:] = observed_data[1:] - observed_data[:-1]
            observed_data = diffs

            diff_mask = torch.zeros_like(observed_mask)
            diff_mask[0] = observed_mask[0]
            diff_mask[1:] = observed_mask[1:] * observed_mask[:-1]
            observed_mask = diff_mask

        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            base_level=base_level,
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


class DatasetETTh(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.8,
        validation_share: float = 0.1,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        first_differences: bool = False,
    ) -> None:
        dataset_path = DatasetsPaths.etth.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "ETTh1 dataset not found. Run fits.data.download.DownloadDatasetETTh() first."
            )

        df = pd.read_csv(dataset_path, parse_dates=["date"])
        df = df.sort_values("date")

        values = df.drop(columns=["date"]).to_numpy(dtype=np.float32)
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
            first_differences=first_differences,
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
            norm_values, norm_mask = values, mask
            if first_differences:
                norm_values, norm_mask = self._apply_first_differences(values, mask)
            normalization_stats = self._compute_training_normalization(
                values=norm_values,
                mask=norm_mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        observed_data = self.data[start_idx:end_idx]
        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        base_level: torch.Tensor | None = None
        if self.first_differences:
            base_level = observed_data[0].clone()
            diffs = torch.zeros_like(observed_data)
            diffs[1:] = observed_data[1:] - observed_data[:-1]
            observed_data = diffs

            diff_mask = torch.zeros_like(observed_mask)
            diff_mask[0] = observed_mask[0]
            diff_mask[1:] = observed_mask[1:] * observed_mask[:-1]
            observed_mask = diff_mask

        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            base_level=base_level,
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


class DatasetElectricity(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.8,
        validation_share: float = 0.1,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        n_features: int = 321,  # max=321; use fewer for faster experiments
        first_differences: bool = False,
    ) -> None:
        dataset_path = DatasetsPaths.electricity.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Electricity dataset not found. "
                "Run fits.dataframes.download.DownloadDatasetElectricity() first."
            )

        df = pd.read_csv(dataset_path, header=None)
        if df.shape[1] > 0 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        df = df.iloc[:, :n_features]
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
            first_differences=first_differences,
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
            norm_values, norm_mask = values, mask
            if first_differences:
                norm_values, norm_mask = self._apply_first_differences(values, mask)
            normalization_stats = self._compute_training_normalization(
                values=norm_values,
                mask=norm_mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        observed_data = self.data[start_idx:end_idx]
        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        base_level: torch.Tensor | None = None
        if self.first_differences:
            base_level = observed_data[0].clone()
            diffs = torch.zeros_like(observed_data)
            diffs[1:] = observed_data[1:] - observed_data[:-1]
            observed_data = diffs

            diff_mask = torch.zeros_like(observed_mask)
            diff_mask[0] = observed_mask[0]
            diff_mask[1:] = observed_mask[1:] * observed_mask[:-1]
            observed_mask = diff_mask

        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            base_level=base_level,
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


class DatasetExchange(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.8,
        validation_share: float = 0.1,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        n_features: int = 8,  # max=8
        first_differences: bool = False,
    ) -> None:
        dataset_path = DatasetsPaths.exchange.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Exchange Rate dataset not found. "
                "Run fits.dataframes.download.DownloadDatasetExchange() first."
            )

        df = pd.read_csv(dataset_path, header=None)
        if df.shape[1] > 0 and df.iloc[:, -1].isna().all():
            df = df.iloc[:, :-1]

        df = df.iloc[:, :n_features]
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
            first_differences=first_differences,
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
            norm_values, norm_mask = values, mask
            if first_differences:
                norm_values, norm_mask = self._apply_first_differences(values, mask)
            normalization_stats = self._compute_training_normalization(
                values=norm_values,
                mask=norm_mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        observed_data = self.data[start_idx:end_idx]
        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        base_level: torch.Tensor | None = None
        if self.first_differences:
            base_level = observed_data[0].clone()
            diffs = torch.zeros_like(observed_data)
            diffs[1:] = observed_data[1:] - observed_data[:-1]
            observed_data = diffs

            diff_mask = torch.zeros_like(observed_mask)
            diff_mask[0] = observed_mask[0]
            diff_mask[1:] = observed_mask[1:] * observed_mask[:-1]
            observed_mask = diff_mask

        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            base_level=base_level,
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


class DatasetWeather(ForecastingDataset):
    def __init__(
        self,
        mode: ModelMode,
        seq_len: int = 96,
        horizon: int = 24,
        train_share: float = 0.8,
        validation_share: float = 0.1,
        normalization: bool = True,
        normalization_stats: NormalizationStats | None = None,
        n_features: int = 21,  # max=21
        first_differences: bool = False,
    ) -> None:
        dataset_path = DatasetsPaths.weather.value

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Weather dataset not found. "
                "Run fits.dataframes.download.DownloadDatasetWeather() first."
            )

        df = pd.read_csv(dataset_path, parse_dates=["date"])
        df = df.sort_values("date")

        values = (
            df.drop(columns=["date"]).iloc[:, :n_features].to_numpy(dtype=np.float32)
        )
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
            first_differences=first_differences,
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
            norm_values, norm_mask = values, mask
            if first_differences:
                norm_values, norm_mask = self._apply_first_differences(values, mask)
            normalization_stats = self._compute_training_normalization(
                values=norm_values,
                mask=norm_mask,
                train_sequences=train_end,
                seq_len=self.seq_len,
            )

        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int) -> ForecastingData:
        start_idx = self.start + index
        end_idx = start_idx + self.seq_len

        observed_data = self.data[start_idx:end_idx]
        observed_mask = self.mask[start_idx:end_idx]

        forecast_mask = torch.zeros_like(observed_mask)
        forecast_mask[-self.horizon :] = observed_mask[-self.horizon :]

        base_level: torch.Tensor | None = None
        if self.first_differences:
            base_level = observed_data[0].clone()
            diffs = torch.zeros_like(observed_data)
            diffs[1:] = observed_data[1:] - observed_data[:-1]
            observed_data = diffs

            diff_mask = torch.zeros_like(observed_mask)
            diff_mask[0] = observed_mask[0]
            diff_mask[1:] = observed_mask[1:] * observed_mask[:-1]
            observed_mask = diff_mask

        observed_data = self._normalize(observed_data)

        return ForecastingData(
            observed_data=observed_data,
            observed_mask=observed_mask,
            forecast_mask=forecast_mask,
            time_points=self.time_points,
            feature_ids=self.feature_ids,
            base_level=base_level,
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
