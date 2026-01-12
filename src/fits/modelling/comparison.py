import math
import pickle
from pathlib import Path
from typing import Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.manifold import TSNE


def CalculateParams(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ReadMetrics(
    eval_foldername: str | Path,
    nsample: int = 10,
):
    evaluation_dir = Path(f"../data/models/evaluation/{eval_foldername}")
    generated_path = evaluation_dir / f"result_nsample{nsample}.pk"

    with open(generated_path, "rb") as f:
        rmse, mae, crps, crps_sum = pickle.load(f)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", crps)
    print("CRPS_sum:", crps_sum)

    return {"rmse": rmse, "mae": mae, "crps": crps, "crps_sum": crps_sum}


def VisualizeForecastSample(
    eval_foldername: str | Path,
    nsample: int = 10,
    n_features: int = 36,
    sample_index: int = 0,
    ncols: int = 4,
    figsize=(24, 36),
) -> None:
    """
    Plot a separate subplot for each feature in feature_index.
    """
    evaluation_dir = Path(f"../data/models/evaluation/{eval_foldername}")
    generated_path = evaluation_dir / f"generated_outputs_nsample{nsample}.pk"

    with open(generated_path, "rb") as f:
        (
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            time_points,
            scaler_tensor,
            mean_tensor,
        ) = pickle.load(f)

    forecasted_data = forecasted_data.cpu()
    forecast_mask = forecast_mask.cpu()
    observed_data = observed_data.cpu()
    observed_mask = observed_mask.cpu()
    time_points = time_points.cpu()

    time_axis = time_points[sample_index].numpy()

    ncols = min(ncols, n_features)
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    # ---- LOOP OVER FEATURES ----
    for ax, feat in zip(axes, range(n_features)):
        forecast_samples = forecasted_data[sample_index, :, :, feat]
        sample_mask = forecast_mask[sample_index, :, feat].bool()
        observed_series = observed_data[sample_index, :, feat]
        observed_series_mask = observed_mask[sample_index, :, feat].bool()

        # Median + intervals
        median = forecast_samples.median(dim=0).values
        lower, upper = torch.quantile(forecast_samples, torch.tensor([0.1, 0.9]), dim=0)

        # Mask missing
        median = median.masked_fill(~sample_mask, torch.nan)
        lower = lower.masked_fill(~sample_mask, torch.nan)
        upper = upper.masked_fill(~sample_mask, torch.nan)

        # Convert
        obs_mask_np = observed_series_mask.numpy()
        obs_series_np = observed_series.numpy()
        sample_mask_np = sample_mask.numpy()
        median_np = median.numpy()
        lower_np = lower.numpy()
        upper_np = upper.numpy()

        # ---- PLOTTING INTO ax ----
        ax.scatter(
            time_axis[obs_mask_np],
            obs_series_np[obs_mask_np],
            color="black",
            s=10,
            label="Observed",
        )
        ax.plot(time_axis, median_np, label="Median", color="tab:green")
        ax.fill_between(
            time_axis,
            lower_np,
            upper_np,
            where=sample_mask_np,
            alpha=0.3,
            color="tab:green",
            label="10–90%",
        )

        ax.set_title(f"Feature {feat}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    # Turn off unused axes if any
    for ax in axes[n_features:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _load_generated_outputs(
    eval_foldername: str | Path,
    nsample: int,
):
    evaluation_dir = Path(f"../data/models/evaluation/{eval_foldername}")
    generated_path = evaluation_dir / f"generated_outputs_nsample{nsample}.pk"

    with open(generated_path, "rb") as f:
        (
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            time_points,
            scaler_tensor,
            mean_tensor,
        ) = pickle.load(f)

    return (
        forecasted_data.cpu(),
        forecast_mask.cpu(),
        observed_data.cpu(),
        observed_mask.cpu(),
        time_points.cpu(),
        scaler_tensor.cpu(),
        mean_tensor.cpu(),
    )


def _normalize_feature_indices(
    feature_index: int | Sequence[int] | None,
    n_features: int,
) -> list[int]:
    if feature_index is None:
        return list(range(n_features))
    if isinstance(feature_index, int):
        feature_index = [feature_index]
    return [idx for idx in feature_index if 0 <= idx < n_features]


def PlotComparisonDataDensity(
    eval_foldername: str | Path,
    nsample: int = 10,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot data density comparison between observed values and forecast samples.
    """
    (
        forecasted_data,
        forecast_mask,
        observed_data,
        observed_mask,
        _,
        _,
        _,
    ) = _load_generated_outputs(eval_foldername, nsample)

    n_features = observed_data.shape[-1]
    selected_features = _normalize_feature_indices(feature_index, n_features)
    if not selected_features:
        raise ValueError("No valid feature indices provided.")

    forecast_samples = forecasted_data[sample_index, :, :, selected_features]
    forecast_samples = forecast_samples.reshape(-1, len(selected_features))
    forecast_mask_samples = forecast_mask[sample_index, :, selected_features]
    forecast_mask_samples = forecast_mask_samples.repeat(nsample, 1, 1).reshape(
        -1, len(selected_features)
    )

    observed_series = observed_data[sample_index, :, selected_features]
    observed_series_mask = observed_mask[sample_index, :, selected_features]

    forecast_values = forecast_samples[forecast_mask_samples.bool()]
    observed_values = observed_series[observed_series_mask.bool()]

    plt.figure(figsize=figsize)
    plt.hist(
        observed_values.numpy(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Observed",
        color="tab:blue",
    )
    plt.hist(
        forecast_values.numpy(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Forecast samples",
        color="tab:orange",
    )
    plt.title("Data Density Comparison")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotComparisonPSA(
    eval_foldername: str | Path,
    nsample: int = 10,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    max_points: int = 5000,
    seed: int = 42,
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """
    Plot a prediction scatter analysis (observed vs. median forecast).
    """
    (
        forecasted_data,
        forecast_mask,
        observed_data,
        observed_mask,
        _,
        _,
        _,
    ) = _load_generated_outputs(eval_foldername, nsample)

    n_features = observed_data.shape[-1]
    selected_features = _normalize_feature_indices(feature_index, n_features)
    if not selected_features:
        raise ValueError("No valid feature indices provided.")

    forecast_samples = forecasted_data[sample_index, :, :, selected_features]
    forecast_median = forecast_samples.median(dim=0).values
    sample_mask = forecast_mask[sample_index, :, selected_features].bool()
    observed_series = observed_data[sample_index, :, selected_features]
    observed_series_mask = observed_mask[sample_index, :, selected_features].bool()

    valid_mask = sample_mask & observed_series_mask
    forecast_values = forecast_median[valid_mask]
    observed_values = observed_series[valid_mask]

    if forecast_values.numel() == 0:
        raise ValueError("No overlapping observed and forecasted values to plot.")

    if forecast_values.numel() > max_points:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(forecast_values.numel(), generator=rng)[:max_points]
        forecast_values = forecast_values.flatten()[indices]
        observed_values = observed_values.flatten()[indices]

    plt.figure(figsize=figsize)
    plt.scatter(observed_values.numpy(), forecast_values.numpy(), alpha=0.4, s=12)
    min_val = min(observed_values.min().item(), forecast_values.min().item())
    max_val = max(observed_values.max().item(), forecast_values.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], color="tab:red", linestyle="--")
    plt.title("Prediction Scatter Analysis")
    plt.xlabel("Observed")
    plt.ylabel("Median forecast")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotComparisonTSNE(
    eval_foldername: str | Path,
    nsample: int = 10,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot t-SNE comparison between observed series and median forecast series.
    """
    (
        forecasted_data,
        forecast_mask,
        observed_data,
        observed_mask,
        _,
        _,
        _,
    ) = _load_generated_outputs(eval_foldername, nsample)

    n_features = observed_data.shape[-1]
    selected_features = _normalize_feature_indices(feature_index, n_features)
    if not selected_features:
        raise ValueError("No valid feature indices provided.")

    observed_series = observed_data[sample_index, :, selected_features]
    observed_series_mask = observed_mask[sample_index, :, selected_features].bool()
    forecast_samples = forecasted_data[sample_index, :, :, selected_features]
    forecast_median = forecast_samples.median(dim=0).values
    forecast_mask_series = forecast_mask[sample_index, :, selected_features].bool()

    valid_time_mask = (observed_series_mask & forecast_mask_series).all(dim=-1)
    if valid_time_mask.sum() < 2:
        raise ValueError("Not enough valid time steps for t-SNE.")

    observed_points = observed_series[valid_time_mask]
    forecast_points = forecast_median[valid_time_mask]

    data = torch.cat([observed_points, forecast_points], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedding = tsne.fit_transform(data)

    n_obs = observed_points.shape[0]
    obs_embed = embedding[:n_obs]
    forecast_embed = embedding[n_obs:]

    plt.figure(figsize=figsize)
    plt.scatter(obs_embed[:, 0], obs_embed[:, 1], label="Observed", alpha=0.7)
    plt.scatter(
        forecast_embed[:, 0],
        forecast_embed[:, 1],
        label="Median forecast",
        alpha=0.7,
    )
    plt.title("t-SNE: Observed vs. Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
