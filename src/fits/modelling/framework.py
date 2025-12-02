import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from IPython.display import clear_output

from fits.config import MODELS_PATH
from fits.data.dataset import ForecastingData, NormalizationStats


@dataclass
class ForecastedData:
    forecasted_data: torch.Tensor  # [B, nsample, K, L] float
    observed_data: torch.Tensor  # [B, K, L] float
    observed_mask: torch.Tensor  # [B, K, L] int
    forecast_mask: torch.Tensor  # [B, K, L] int
    time_points: torch.Tensor  # [B, L] float


@dataclass
class ModelConfig:
    """Base model configuration used by :class:`ForecastingModel`.

    The configuration is intentionally typed to avoid passing around loose
    dictionaries. Concrete models should subclass this dataclass when they need
    additional parameters.
    """

    name: Optional[str] = None
    device: Optional[torch.device | str] = None


class ForecastingModel(nn.Module, ABC):
    """Base class for all forecasting models in the project."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = self._resolve_device(config.device)

    @property
    def model_name(self) -> str:
        return self.config.name or self.__class__.__name__

    def _resolve_device(self, device: Optional[torch.device | str]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        # Keep a reference to the current device for adapters using it internally.
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self

    @abstractmethod
    def forward(self, batch: ForecastingData, is_train: int = 1):
        """Compute the training loss for a batch."""

    @abstractmethod
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        """Run model-specific evaluation and return generated samples."""


def Train(
    model: ForecastingModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    lr: float = 1.0e-3,
    epochs: int = 200,
    valid_epoch_interval: int = 10,
    verbose: bool = True,
):
    """Generic training loop for :class:`ForecastingModel` implementations."""

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = MODELS_PATH / f"{model.model_name}_{current_time}"
    folder_name.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # first 0.10 * epoechs: 1e-3
    # then  0.15 * epoechs: 1e-3 * 0.1 = 1e-4
    # then  0.75 * epoechs: 1e-4 * 0.1 = 1e-5
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    shed = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[p1, p2], gamma=0.1)

    metrics: dict[str, list[float]] = {"train_loss": [], "test_loss": []}

    best_valid_loss = float("inf")
    for epoch_no in range(epochs):
        epoch_loss = 0.0
        model.train()

        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                opt.zero_grad()

                loss = model(train_batch)
                epoch_loss += loss.item()

                loss.backward()
                opt.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": epoch_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            shed.step()

        metrics["train_loss"].append((epoch_no + 1, epoch_loss))

        if (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            valid_loss = 0.0
            valid_batches = 0

            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        valid_loss += loss.item()
                        valid_batches = batch_no

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": valid_loss / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            metrics["test_loss"].append((epoch_no + 1, valid_loss))

            avg_valid_loss = valid_loss / max(valid_batches, 1)
            if best_valid_loss > avg_valid_loss:
                best_valid_loss = avg_valid_loss
                print(
                    "\n best loss is updated to ",
                    avg_valid_loss,
                    "at",
                    epoch_no,
                )

        if verbose:
            clear_output(True)
            plt.figure(figsize=(12, 4))

            for i, (name, history) in enumerate(sorted(metrics.items())):
                plt.subplot(1, len(metrics), i + 1)
                plt.title(name)
                plt.plot(*zip(*history))
                plt.grid()

            plt.savefig(folder_name / "training.png")
            plt.show()

        torch.save(model.state_dict(), folder_name / "model.pth")


def CalcQuantileLoss(target, forecast, q: float, eval_points) -> torch.Tensor:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def CalcQuantileCRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = torch.sum(torch.abs(target * eval_points))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = CalcQuantileLoss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def CalcQuantileCRPSSum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = torch.sum(torch.abs(target * eval_points))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1), quantiles[i], dim=1)
        q_loss = CalcQuantileLoss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


@torch.no_grad()
def Evaluate(
    model: ForecastingModel,
    test_loader: DataLoader,
    nsample: int = 100,
    normalization: NormalizationStats | None = None,
    scaler: float | torch.Tensor = 1,
    mean_scaler: float | torch.Tensor = 0,
    foldername: str = "",
):
    """Evaluate a :class:`ForecastingModel` and persist generated outputs.

    The function keeps the same side effects as the legacy implementation (pickle
    dumps for generated outputs and final metrics) while improving readability
    and bookkeeping.
    """

    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    evalpoints_total = 0.0

    all_target: list[torch.Tensor] = []
    all_observed_point: list[torch.Tensor] = []
    all_observed_time: list[torch.Tensor] = []
    all_evalpoint: list[torch.Tensor] = []
    all_generated_samples: list[torch.Tensor] = []

    output_dir = Path(foldername) if foldername else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalization is not None:
        scaler = normalization.std
        mean_scaler = normalization.mean

    scaler_tensor = torch.as_tensor(scaler)
    mean_tensor = torch.as_tensor(mean_scaler)

    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            forecasted_data = model.evaluate(test_batch, nsample)

            samples = forecasted_data.forecasted_data
            c_target = forecasted_data.observed_data
            eval_points = forecasted_data.forecast_mask
            observed_points = forecasted_data.observed_mask
            observed_time = forecasted_data.time_points

            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)

            scaler_tensor = torch.as_tensor(scaler, device=samples.device)
            mean_tensor = torch.as_tensor(mean_scaler, device=samples.device)
            scaler = scaler_tensor
            mean_scaler = mean_tensor

            samples_median = samples.median(dim=1)
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (
                scaler_tensor**2
            )
            mae_current = (
                torch.abs((samples_median.values - c_target) * eval_points)
            ) * scaler_tensor

            mse_total += mse_current.sum().item()
            mae_total += mae_current.sum().item()
            evalpoints_total += eval_points.sum().item()

            it.set_postfix(
                ordered_dict={
                    "rmse_total": np.sqrt(mse_total / evalpoints_total),
                    "mae_total": mae_total / evalpoints_total,
                    "batch_no": batch_no,
                },
                refresh=True,
            )

    with open(output_dir / f"generated_outputs_nsample{nsample}.pk", "wb") as f:
        pickle.dump(
            [
                torch.cat(all_generated_samples, dim=0),
                torch.cat(all_target, dim=0),
                torch.cat(all_evalpoint, dim=0),
                torch.cat(all_observed_point, dim=0),
                torch.cat(all_observed_time, dim=0),
                scaler_tensor.cpu(),
                mean_tensor.cpu(),
            ],
            f,
        )

    crps = CalcQuantileCRPS(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
    )
    crps_sum = CalcQuantileCRPSSum(
        all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
    )

    rmse = np.sqrt(mse_total / evalpoints_total)
    mae = mae_total / evalpoints_total
    with open(output_dir / f"result_nsample{nsample}.pk", "wb") as f:
        pickle.dump([rmse, mae, crps], f)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", crps)
    print("CRPS_sum:", crps_sum)
