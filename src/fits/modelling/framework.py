import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fits.config import EVALUATION_PATH, TRAINING_PATH
from fits.dataframes.dataloader import ForecastingDataLoader
from fits.dataframes.dataset import (
    ForecastingData,
    ForecastingDataset,
    NormalizationStats,
)


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
    base_level: torch.Tensor | None = (
        None  # [B, K] — raw first value (first-differences only)
    )


@dataclass
class ModelConfig:
    """Base model configuration used by :class:`ForecastingModel`.

    The configuration is intentionally typed to avoid passing around loose
    dictionaries.
    """

    name: str | None = None


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
    def forward(self, batch: ForecastingData) -> torch.Tensor:
        """Compute the training loss for a batch."""

    @abstractmethod
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        """Run model-specific evaluation and return generated samples."""


@dataclass
class EMA:
    """Exponential Moving Average of model parameters."""

    decay: float = 0.999
    device: torch.device | None = None  # e.g. torch.device("cpu") to store EMA on CPU

    def __post_init__(self):
        self.shadow = {}
        self._backup = {}

    @torch.no_grad()
    def register(self, model: torch.nn.Module) -> None:
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (
                    p.detach().clone().to(self.device)
                    if self.device
                    else p.detach().clone()
                )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert (
                name in self.shadow
            ), "EMA.register(model) must be called before EMA.update(model)."
            new = p.detach()
            if self.device:
                new = new.to(self.device)
            self.shadow[name].mul_(d).add_(new, alpha=(1.0 - d))

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        """Swap model params to EMA params (stores current params for later restore)."""
        self._backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup[name] = p.detach().clone()
            ema_p = self.shadow[name]
            if ema_p.device != p.device:
                ema_p = ema_p.to(p.device)
            p.copy_(ema_p)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore params saved by apply()."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self._backup[name])
        self._backup = {}


def Train(
    model: ForecastingModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    lr: float = 1.0e-3,
    epochs: int = 200,
    valid_epoch_interval: int = 10,
    verbose: bool = True,
    warmup_epochs: int = 10,
    warmup_start_factor: float = 0.1,  # initial lr = lr * warmup_start_factor
    grad_clip_norm: float | None = 0.3,
    weight_decay: float = 1.0e-5,
    use_ema: bool = False,
    ema_decay: float = 0.995,  # if use_ema=True
    ema_eval: bool = True,  # if use_ema=True
    ema_save: bool = True,  # if use_ema=True
    folder_name: str | None = None,
):
    if not folder_name:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{model.model_name}_{current_time}"

    folder_path = TRAINING_PATH / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # first 75% epoechs: 1e-3
    # then  15% epoechs: 1e-3 * 0.1 = 1e-4
    # then  10% epoechs: 1e-4 * 0.1 = 1e-5
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)

    scheduler: (
        torch.optim.lr_scheduler.LambdaLR | torch.optim.lr_scheduler.MultiStepLR | None
    ) = None
    if warmup_epochs > 0:

        def lr_lambda(epoch: int) -> float:
            warm = min(1.0, (epoch + 1) / float(warmup_epochs))
            # approximate MultiStepLR behavior with epoch-based factor:
            # after passing milestones, apply gamma^k
            k = 0
            if (epoch + 1) > p1:
                k += 1
            if (epoch + 1) > p2:
                k += 1
            return (warmup_start_factor + (1.0 - warmup_start_factor) * warm) * (0.1**k)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[p1, p2], gamma=0.1
        )

    ema: EMA | None = None
    if use_ema:
        ema = EMA(decay=ema_decay)
        ema.register(model)

    metrics: dict[str, list[tuple[int, float]]] = {"train_loss": [], "test_loss": []}
    best_valid_loss = float("inf")
    best_valid_epoch = -1

    for epoch_no in range(epochs):
        epoch_loss = 0.0
        model.train()

        with tqdm(train_loader, leave=False) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                opt.zero_grad()

                loss = model(train_batch)
                epoch_loss += loss.item()

                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()

                if ema:
                    ema.update(model)

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": epoch_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

        scheduler.step()
        metrics["train_loss"].append((epoch_no + 1, epoch_loss / batch_no))

        if (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()

            if ema and ema_eval:
                ema.apply(model)

            valid_loss = 0.0
            valid_batches = 0

            with torch.no_grad():
                with tqdm(valid_loader, leave=False) as it:
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

            metrics["test_loss"].append(
                (epoch_no + 1, valid_loss / max(valid_batches, 1))
            )

            avg_valid_loss = valid_loss / max(valid_batches, 1)
            if best_valid_loss > avg_valid_loss:
                torch.save(model.state_dict(), folder_path / "best_model.pth")

                best_valid_loss = avg_valid_loss
                best_valid_epoch = epoch_no

            if ema and ema_eval:
                ema.restore(model)

        with open(folder_path / "metrics.json", "w") as _f:
            json.dump(metrics, _f)

        # Always save the plot; only display it interactively when verbose=True
        if verbose:
            clear_output(True)
        fig = plt.figure(figsize=(12, 4))
        for i, (name, history) in enumerate(sorted(metrics.items())):
            plt.subplot(1, len(metrics), i + 1)
            plt.title(name)
            plt.plot(*zip(*history, strict=False))
            plt.grid()
        plt.gca().text(
            0.02,
            0.98,
            f"lr = {opt.param_groups[0]['lr']:.2e}",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
        )
        plt.savefig(folder_path / "training.png")
        if verbose:
            plt.show()
        plt.close(fig)

        if ema and ema_save:
            ema.apply(model)

        torch.save(model.state_dict(), folder_path / "model.pth")

        if ema and ema_save:
            ema.restore(model)

    tqdm.write(f"  best loss: {best_valid_loss:.6f} at epoch {best_valid_epoch}")


def CalcQuantileLoss(target, forecast, q: float, eval_points) -> torch.Tensor:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def CalcQuantileCRPS(target, forecast, eval_points, center, scale):
    target = target * scale + center
    forecast = forecast * scale + center

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


def CalcQuantileCRPSSum(target, forecast, eval_points, center, scale):
    eval_points = eval_points.float().mean(-1)
    target = target * scale + center
    target = target.sum(-1)
    forecast = forecast * scale + center

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
    nsample: int = 5,
    folder_name: str | None = None,
):
    if not folder_name:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{model.model_name}_{current_time}"

    folder_path = EVALUATION_PATH / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    evalpoints_total = 0.0

    min_tensor = torch.as_tensor(normalization.min, device=model.device)
    max_tensor = torch.as_tensor(normalization.max, device=model.device)
    center_tensor = (max_tensor + min_tensor) / 2
    scale_tensor = (max_tensor - min_tensor) / 2
    scale_tensor = torch.where(
        scale_tensor == 0, torch.ones_like(scale_tensor), scale_tensor
    )

    # When the dataset used first_differences, center/scale refer to
    # difference-space normalisation.  Keep a copy for reconstruction,
    # then switch the metric tensors to identity so that MSE/MAE/CRPS
    # are computed directly in raw-level space.
    first_differences = getattr(test_loader.dataset, "first_differences", False)
    if first_differences:
        diff_center = center_tensor.clone()
        diff_scale = scale_tensor.clone()
        # Metrics will operate in raw space after reconstruction.
        scale_tensor = torch.ones_like(scale_tensor)
        center_tensor = torch.zeros_like(center_tensor)

    all_forecasted_data: list[torch.Tensor] = []
    all_forecast_mask: list[torch.Tensor] = []
    all_observed_data: list[torch.Tensor] = []
    all_observed_mask: list[torch.Tensor] = []
    all_time_points: list[torch.Tensor] = []

    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            f: ForecastedData = model.evaluate(test_batch, nsample)

            if first_differences and f.base_level is not None:
                # De-normalise from difference space to raw differences …
                raw_diffs_pred = f.forecasted_data * diff_scale + diff_center
                raw_diffs_obs = f.observed_data * diff_scale + diff_center
                base = f.base_level.to(
                    device=raw_diffs_pred.device, dtype=raw_diffs_pred.dtype
                )
                # Anchor the cumsum at the start of the horizon: keep observed
                # diffs in the context so context-prediction errors don't leak
                # into the reconstructed raw-level forecast.
                horizon = f.forecast_mask.unsqueeze(1).bool()
                raw_diffs_pred = torch.where(
                    horizon,
                    raw_diffs_pred,
                    raw_diffs_obs.unsqueeze(1).expand_as(raw_diffs_pred),
                )
                # … then integrate and anchor to the raw base level.
                f = ForecastedData(
                    forecasted_data=base.unsqueeze(1).unsqueeze(1)
                    + raw_diffs_pred.cumsum(dim=2),
                    observed_data=base.unsqueeze(1) + raw_diffs_obs.cumsum(dim=1),
                    forecast_mask=f.forecast_mask,
                    observed_mask=f.observed_mask,
                    time_points=f.time_points,
                    base_level=f.base_level,
                )

            all_forecasted_data.append(f.forecasted_data)
            all_forecast_mask.append(f.forecast_mask)
            all_observed_data.append(f.observed_data)
            all_observed_mask.append(f.observed_mask)
            all_time_points.append(f.time_points)

            forecasted_data_median = f.forecasted_data.median(dim=1).values

            mse_current = (
                ((forecasted_data_median - f.observed_data) * f.forecast_mask) ** 2
            ) * (scale_tensor**2)
            mae_current = (
                torch.abs((forecasted_data_median - f.observed_data) * f.forecast_mask)
            ) * scale_tensor

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

    with open(folder_path / f"generated_outputs_nsample{nsample}.pk", "wb") as file:
        pickle.dump(
            [
                all_forecasted_data_tensor,
                all_forecast_mask_tensor,
                all_observed_data_tensor,
                all_observed_mask_tensor,
                all_time_points_tensor,
                scale_tensor.cpu(),
                center_tensor.cpu(),
            ],
            file,
        )

    crps = CalcQuantileCRPS(
        all_observed_data_tensor,
        all_forecasted_data_tensor,
        all_forecast_mask_tensor,
        center_tensor,
        scale_tensor,
    )
    crps_sum = CalcQuantileCRPSSum(
        all_observed_data_tensor,
        all_forecasted_data_tensor,
        all_forecast_mask_tensor,
        center_tensor,
        scale_tensor,
    )

    rmse = np.sqrt(mse_total / evalpoints_total)
    mae = mae_total / evalpoints_total

    metrics = [rmse, mae, crps, crps_sum]
    metrics = [float(metric) for metric in metrics]
    with open(folder_path / f"result_nsample{nsample}.pk", "wb") as file:
        pickle.dump(metrics, file)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", crps)
    print("CRPS_sum:", crps_sum)


def TrainAndEvaluate(
    model: ForecastingModel,
    dataset_cls: type[ForecastingDataset],
    # --- data ---
    batch_size: int = 128,
    num_workers: int = 0,
    dataset_kwargs: dict[str, Any] | None = None,
    # --- training ---
    lr: float = 1.0e-3,
    epochs: int = 200,
    valid_epoch_interval: int = 10,
    verbose: bool = True,
    warmup_epochs: int = 10,
    warmup_start_factor: float = 0.1,
    grad_clip_norm: float | None = 0.3,
    weight_decay: float = 1.0e-5,
    use_ema: bool = False,
    ema_decay: float = 0.995,
    ema_eval: bool = True,
    ema_save: bool = True,
    # --- evaluation ---
    nsample: int = 10,
    # --- shared ---
    folder_name: str | None = None,
    eval_only: bool = False,
) -> None:
    """Train a model, load the best checkpoint, then evaluate on the test set.

    Combines :func:`ForecastingDataLoader`, :func:`Train`, and
    :func:`Evaluate` into a single convenience call.  The ``folder_name``
    is shared between training (checkpoint) and evaluation (results) so
    both artefacts end up under the same name in their respective
    ``data/models/`` sub-directories.

    Args:
        model:               Model to train and evaluate.
        dataset_cls:         Dataset class (e.g. ``DatasetETTh``).
        batch_size:          Batch size for all data loaders.
        num_workers:         DataLoader worker processes.
        dataset_kwargs:      Extra keyword arguments forwarded to the
                             dataset constructor (e.g. ``seq_len``, ``horizon``).
        folder_name:         Optional name for checkpoint / results folders.
                             Auto-generated from model name + timestamp if omitted.
        eval_only:           If True, skip training and evaluate an existing
                             checkpoint at ``TRAINING_PATH / folder_name / best_model.pth``.
        All remaining args:  Forwarded verbatim to :func:`Train` / :func:`Evaluate`.
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    if not folder_name:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{model.model_name}_{current_time}"

    # Build loaders
    train_loader, valid_loader, test_loader = ForecastingDataLoader(
        dataset_cls,
        batch_size=batch_size,
        num_workers=num_workers,
        **dataset_kwargs,
    )

    if not eval_only:
        # Train — saves best_model.pth under TRAINING_PATH / folder_name
        Train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=lr,
            epochs=epochs,
            valid_epoch_interval=valid_epoch_interval,
            verbose=verbose,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            grad_clip_norm=grad_clip_norm,
            weight_decay=weight_decay,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_eval=ema_eval,
            ema_save=ema_save,
            folder_name=folder_name,
        )

    # Load the best checkpoint (saved by Train() in the normal flow, or
    # produced by a previous run when ``eval_only`` is set).
    best_ckpt = TRAINING_PATH / folder_name / "best_model.pth"
    if eval_only and not best_ckpt.is_file():
        raise FileNotFoundError(
            f"eval_only=True but no checkpoint was found at {best_ckpt}"
        )
    model.load_state_dict(torch.load(best_ckpt, map_location=model.device))

    # Evaluate on test set using normalization stats from the training split
    normalization = train_loader.dataset.normalization_stats  # type: ignore[union-attr]

    Evaluate(
        model=model,
        test_loader=test_loader,
        normalization=normalization,
        nsample=nsample,
        folder_name=folder_name,
    )
