import pickle
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from fits.config import EVALUATION_PATH
from fits.dataframes.dataset import NormalizationStats
from fits.modelling.FITSJ.transformer import Decomposition
from fits.modelling.FITSJ.model import FITSModel


def _normalize_feature_indices(
    feature_index: int | Sequence[int] | None,
    n_features: int,
) -> list[int]:
    if feature_index is None:
        return list(range(n_features))
    if isinstance(feature_index, int):
        feature_index = [feature_index]
    return [idx for idx in feature_index if 0 <= idx < n_features]


@torch.no_grad()
def EvaluateFITSJWithDecomposition(
    model: FITSModel,
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

    min_tensor = torch.as_tensor(normalization.min, device=model.device)
    max_tensor = torch.as_tensor(normalization.max, device=model.device)
    center_tensor = (max_tensor + min_tensor) / 2
    scale_tensor = (max_tensor - min_tensor) / 2
    scale_tensor = torch.where(
        scale_tensor == 0, torch.ones_like(scale_tensor), scale_tensor
    )

    all_forecasted_data: list[torch.Tensor] = []
    all_forecast_mask: list[torch.Tensor] = []
    all_observed_data: list[torch.Tensor] = []
    all_observed_mask: list[torch.Tensor] = []
    all_time_points: list[torch.Tensor] = []
    all_trend: list[torch.Tensor] = []
    all_season: list[torch.Tensor] = []
    all_jump: list[torch.Tensor] = []
    all_error: list[torch.Tensor] = []

    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for test_batch in it:
            forecasted, decomposed = model.decomposition_evaluate(
                test_batch, nsample
            )

            all_forecasted_data.append(forecasted.forecasted_data)
            all_forecast_mask.append(forecasted.forecast_mask)
            all_observed_data.append(forecasted.observed_data)
            all_observed_mask.append(forecasted.observed_mask)
            all_time_points.append(forecasted.time_points)

            all_trend.append(decomposed.trend)
            all_season.append(decomposed.season)
            all_jump.append(decomposed.jump_term)
            all_error.append(decomposed.error)

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

    decomposed = Decomposition(
        trend=torch.cat(all_trend, dim=0),
        season=torch.cat(all_season, dim=0),
        jump_term=torch.cat(all_jump, dim=0),
        error=torch.cat(all_error, dim=0),
    )
    with open(folder_path / f"decomposition_outputs_nsample{nsample}.pk", "wb") as file:
        pickle.dump(decomposed, file)


def PlotFITSJDecomposition(
    eval_foldername: str | Path,
    nsample: int = 5,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    evaluation_dir = Path(f"../data/models/evaluation/{eval_foldername}")
    generated_path = evaluation_dir / f"generated_outputs_nsample{nsample}.pk"
    decomposition_path = evaluation_dir / f"decomposition_outputs_nsample{nsample}.pk"

    with open(generated_path, "rb") as f:
        (
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            time_points,
            scale_tensor,
            center_tensor,
        ) = pickle.load(f)

    with open(decomposition_path, "rb") as f:
        decomposed: Decomposition = pickle.load(f)

    forecasted_data = forecasted_data.cpu()
    forecast_mask = forecast_mask.cpu()
    observed_data = observed_data.cpu()
    observed_mask = observed_mask.cpu()
    time_points = time_points.cpu()
    scale_tensor = scale_tensor.cpu()
    center_tensor = center_tensor.cpu()

    n_features = observed_data.shape[-1]
    selected_features = _normalize_feature_indices(feature_index, n_features)
    if not selected_features:
        raise ValueError("No valid feature indices provided.")

    ncols = len(selected_features)
    nrows = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    time_axis = time_points[sample_index].numpy()

    forecast_samples = forecasted_data[sample_index]
    forecast_median = forecast_samples.median(dim=0).values
    forecast_lower, forecast_upper = torch.quantile(
        forecast_samples, torch.tensor([0.1, 0.9]), dim=0
    )

    trend_median = decomposed.trend[sample_index].median(dim=0).values.cpu()
    season_median = decomposed.season[sample_index].median(dim=0).values.cpu()
    jump_median = decomposed.jump_term[sample_index].median(dim=0).values.cpu()
    error_median = decomposed.error[sample_index].median(dim=0).values.cpu()

    for col_idx, feat in enumerate(selected_features):
        horizon_mask = forecast_mask[sample_index, :, feat].bool()
        observed_series = observed_data[sample_index, :, feat]
        observed_series_mask = observed_mask[sample_index, :, feat].bool()
        observed_horizon_mask = horizon_mask & observed_series_mask

        scale = scale_tensor[feat]
        center = center_tensor[feat]

        observed_denorm = observed_series * scale + center
        forecast_denorm = forecast_median[:, feat] * scale + center
        forecast_lower_denorm = forecast_lower[:, feat] * scale + center
        forecast_upper_denorm = forecast_upper[:, feat] * scale + center

        trend_denorm = trend_median[:, feat] * scale + center
        season_denorm = season_median[:, feat] * scale
        jump_denorm = jump_median[:, feat] * scale
        error_denorm = error_median[:, feat] * scale

        axes[0, col_idx].plot(
            time_axis[observed_horizon_mask.numpy()],
            observed_denorm[observed_horizon_mask].numpy(),
            color="black",
            label="Observed",
        )
        axes[0, col_idx].plot(
            time_axis[horizon_mask.numpy()],
            forecast_denorm[horizon_mask].numpy(),
            color="tab:green",
            label="Forecast (median)",
        )
        axes[0, col_idx].fill_between(
            time_axis,
            forecast_lower_denorm.numpy(),
            forecast_upper_denorm.numpy(),
            where=horizon_mask.numpy(),
            alpha=0.3,
            color="tab:green",
            label="10–90%",
        )
        axes[0, col_idx].set_title(f"Feature {feat} | Observed vs Forecast")
        axes[0, col_idx].legend()
        axes[0, col_idx].grid(True)

        axes[1, col_idx].plot(
            time_axis[horizon_mask.numpy()],
            trend_denorm[horizon_mask].numpy(),
            color="tab:blue",
        )
        axes[1, col_idx].set_title("Trend")
        axes[1, col_idx].grid(True)

        axes[2, col_idx].plot(
            time_axis[horizon_mask.numpy()],
            season_denorm[horizon_mask].numpy(),
            color="tab:orange",
        )
        axes[2, col_idx].set_title("Season")
        axes[2, col_idx].grid(True)

        axes[3, col_idx].plot(
            time_axis[horizon_mask.numpy()],
            jump_denorm[horizon_mask].numpy(),
            color="tab:red",
        )
        axes[3, col_idx].set_title("Jump term")
        axes[3, col_idx].grid(True)

        axes[4, col_idx].plot(
            time_axis[horizon_mask.numpy()],
            error_denorm[horizon_mask].numpy(),
            color="tab:purple",
        )
        axes[4, col_idx].set_title("Error")
        axes[4, col_idx].grid(True)

        for row in range(nrows):
            axes[row, col_idx].set_xlabel("Time step")
            axes[row, col_idx].set_ylabel("Value")

    plt.tight_layout()
    plt.show()
