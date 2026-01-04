import pickle
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

from fits.config import TRAINING_PATH, EVALUATION_PATH
from fits.dataframes.dataset import ForecastingData, NormalizationStats


@dataclass
class ForecastedData:
    # `nsample` - number of Monte-Carlo forecast samples
    forecasted_data: torch.Tensor  # [B, nsample, L, K] float
    forecast_mask: torch.Tensor  # [B, L, K] int
    # 0 - context (history)
    # 1 - horizon to forecast (to be generated)
    observed_data: torch.Tensor  # [B, L, K] float
    observed_mask: torch.Tensor  # [B, L, K] int
    # 0 - not observed
    # 1 - observed (ground truth)
    time_points: torch.Tensor  # [B, L] float


@dataclass
class ModelConfig:
    """Base model configuration used by :class:`ForecastingModel`.

    The configuration is intentionally typed to avoid passing around loose
    dictionaries. Concrete models should subclass this dataclass when they need
    additional parameters.
    """

    name: Optional[str] = None


class ForecastingModel(nn.Module, ABC):
    """Base class for all forecasting models in the project."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        self.device = torch.device("cuda")

    @property
    def model_name(self) -> str:
        return self.config.name or self.__class__.__name__

    @abstractmethod
    def forward(self, batch: ForecastingData):
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
    valid_epoch_interval: int = 20,
    verbose: bool = True,
):
    """Generic training loop for :class:`ForecastingModel` implementations."""

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = TRAINING_PATH / f"{model.model_name}_{current_time}"
    folder_name.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # first 0.10 * epoechs: 1e-3
    # then  0.15 * epoechs: 1e-3 * 0.1 = 1e-4
    # then  0.75 * epoechs: 1e-4 * 0.1 = 1e-5
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    shed = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[p1, p2], gamma=0.1)

    metrics: dict[str, list[tuple[int, float]]] = {"train_loss": [], "test_loss": []}

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
                        loss = model(valid_batch)
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

    # Ensure the evaluation mask participates in floating point operations.
    # Some models (e.g., DiffusionTSAdapter) surface boolean masks, causing
    # ``mean`` to fail because it cannot infer a floating output dtype from
    # booleans.  Cast explicitly to float before reducing.
    eval_points = eval_points.float().mean(-1)
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
    normalization: NormalizationStats,
    nsample: int = 10,
):
    """Evaluate a :class:`ForecastingModel` and persist generated outputs.

    The function keeps the same side effects as the legacy implementation (pickle
    dumps for generated outputs and final metrics) while improving readability
    and bookkeeping.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = EVALUATION_PATH / f"{model.model_name}_{current_time}"
    folder_name.mkdir(parents=True, exist_ok=True)

    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    evalpoints_total = 0.0

    scaler_tensor = torch.as_tensor(normalization.std, device=model.device)
    mean_tensor = torch.as_tensor(normalization.mean, device=model.device)

    all_forecasted_data: list[torch.Tensor] = []
    all_forecast_mask: list[torch.Tensor] = []
    all_observed_data: list[torch.Tensor] = []
    all_observed_mask: list[torch.Tensor] = []
    all_time_points: list[torch.Tensor] = []

    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            f: ForecastedData = model.evaluate(test_batch, nsample)

            all_forecasted_data.append(f.forecasted_data)
            all_forecast_mask.append(f.forecast_mask)
            all_observed_data.append(f.observed_data)
            all_observed_mask.append(f.observed_mask)
            all_time_points.append(f.time_points)

            forecasted_data_median = f.forecasted_data.median(dim=1).values

            mse_current = (
                ((forecasted_data_median - f.observed_data) * f.forecast_mask) ** 2
            ) * (scaler_tensor**2)
            mae_current = (
                torch.abs((forecasted_data_median - f.observed_data) * f.forecast_mask)
            ) * scaler_tensor

            mse_total += mse_current.sum().item()
            mae_total += mae_current.sum().item()
            evalpoints_total += f.forecast_mask.sum().item()

            it.set_postfix(
                ordered_dict={
                    "rmse_total": np.sqrt(mse_total / evalpoints_total),
                    "mae_total": mae_total / evalpoints_total,
                    "batch_no": batch_no,
                },
                refresh=True,
            )

    all_forecasted_data_tensor = torch.cat(all_forecasted_data, dim=0)
    all_forecast_mask_tensor = torch.cat(all_forecast_mask, dim=0)
    all_observed_data_tensor = torch.cat(all_observed_data, dim=0)
    all_observed_mask_tensor = torch.cat(all_observed_mask, dim=0)
    all_time_points_tensor = torch.cat(all_time_points, dim=0)

    with open(folder_name / f"generated_outputs_nsample{nsample}.pk", "wb") as file:
        pickle.dump(
            [
                all_forecasted_data_tensor,
                all_forecast_mask_tensor,
                all_observed_data_tensor,
                all_observed_mask_tensor,
                all_time_points_tensor,
                scaler_tensor.cpu(),
                mean_tensor.cpu(),
            ],
            file,
        )

    crps = CalcQuantileCRPS(
        all_observed_data_tensor,
        all_forecasted_data_tensor,
        all_forecast_mask_tensor,
        mean_tensor,
        scaler_tensor,
    )
    crps_sum = CalcQuantileCRPSSum(
        all_observed_data_tensor,
        all_forecasted_data_tensor,
        all_forecast_mask_tensor,
        mean_tensor,
        scaler_tensor,
    )

    rmse = np.sqrt(mse_total / evalpoints_total)
    mae = mae_total / evalpoints_total

    metrics = [rmse, mae, crps, crps_sum]
    metrics = [float(metric) for metric in metrics]
    with open(folder_name / f"result_nsample{nsample}.pk", "wb") as file:
        pickle.dump(metrics, file)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", crps)
    print("CRPS_sum:", crps_sum)
