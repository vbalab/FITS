import math
import torch
import pickle
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path


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
