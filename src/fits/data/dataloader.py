from typing import Any

from torch.utils.data import DataLoader

from fits.data.dataset import ForecastingData, ForecastingDataset, ModelMode


def InitDataLoader(
    dataset_cls: type[ForecastingDataset],
    batch_size: int = 128,
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
        num_workers=1,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,  #!
    )

    return train_loader, valid_loader, test_loader
