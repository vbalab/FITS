from typing import Any

import torch
from torch.utils.data import DataLoader

from fits.dataframes.dataset import ForecastingData, ForecastingDataset, ModelMode


def _collate_forecasting_data(batch: list[ForecastingData]) -> ForecastingData:
    """Stack :class:`ForecastingData` samples into a single batch."""

    observed_data = torch.stack([sample.observed_data for sample in batch], dim=0)
    observed_mask = torch.stack([sample.observed_mask for sample in batch], dim=0)
    forecast_mask = torch.stack([sample.forecast_mask for sample in batch], dim=0)
    time_points = torch.stack([sample.time_points for sample in batch], dim=0)
    feature_ids = torch.stack([sample.feature_ids for sample in batch], dim=0)

    return ForecastingData(
        observed_data=observed_data,
        observed_mask=observed_mask,
        forecast_mask=forecast_mask,
        time_points=time_points,
        feature_ids=feature_ids,
    )


def ForecastingDataLoader(
    dataset_cls: type[ForecastingDataset],
    batch_size: int = 128,
    num_workers=0,
    **dataset_kwargs: Any,
) -> tuple[
    DataLoader[ForecastingData],
    DataLoader[ForecastingData],
    DataLoader[ForecastingData],
]:
    train_ds = dataset_cls(mode=ModelMode.train, **dataset_kwargs)
    valid_ds = dataset_cls(mode=ModelMode.valid, **dataset_kwargs)
    test_ds = dataset_cls(mode=ModelMode.test, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=_collate_forecasting_data,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=_collate_forecasting_data,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  #!
        collate_fn=_collate_forecasting_data,
    )

    return train_loader, valid_loader, test_loader
