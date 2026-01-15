import math
import pickle
from pathlib import Path
from typing import Sequence

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity


def CalculateParams(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def LoadBestModel(model: nn.Module, model_foldername: str, device: torch.device):
    model_path = f"../data/models/training/{model_foldername}/best_model.pth"

    state = torch.load(model_path, map_location=device)

    model.load_state_dict(state)
    model.to(device)


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
        ax.plot(
            time_axis[obs_mask_np],
            obs_series_np[obs_mask_np],
            color="black",
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


def _compute_feature_means(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    mask = observed_mask.bool()
    masked = observed_data.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=(0, 1)).clamp_min(1).to(masked.dtype)
    feature_means = masked.sum(dim=(0, 1)) / denom
    return feature_means


def _impute_missing(
    data: torch.Tensor,
    mask: torch.Tensor,
    feature_means: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    filled = data.clone()
    for feature_idx in range(data.shape[-1]):
        filled[..., feature_idx] = torch.where(
            mask[..., feature_idx],
            filled[..., feature_idx],
            feature_means[feature_idx],
        )
    return filled


def _flatten_series(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    forecasted_data: torch.Tensor,
    forecast_mask: torch.Tensor,
    selected_features: Sequence[int],
    nsample: int,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    comparison_mask = forecast_mask[:, :, selected_features]
    feature_means = _compute_feature_means(
        observed_data[:, :, selected_features],
        comparison_mask,
    )
    observed_filled = _impute_missing(
        observed_data[:, :, selected_features],
        comparison_mask,
        feature_means,
    )
    forecast_mask_samples = comparison_mask.unsqueeze(1)
    forecast_mask_samples = forecast_mask_samples.expand(
        -1, nsample, -1, -1
    ).reshape(forecasted_data.shape[0] * nsample, *forecast_mask_samples.shape[2:])
    forecast_filled = _impute_missing(
        forecasted_data[:, :, :, selected_features].reshape(
            forecasted_data.shape[0] * nsample,
            forecasted_data.shape[2],
            len(selected_features),
        ),
        forecast_mask_samples,
        feature_means,
    )

    observed_vectors = observed_filled.reshape(observed_filled.shape[0], -1).numpy()
    forecast_vectors = forecast_filled.reshape(forecast_filled.shape[0], -1).numpy()

    if max_points is not None:
        rng = np.random.default_rng(seed)
        if observed_vectors.shape[0] > max_points:
            observed_vectors = observed_vectors[
                rng.choice(observed_vectors.shape[0], size=max_points, replace=False)
            ]
        if forecast_vectors.shape[0] > max_points:
            forecast_vectors = forecast_vectors[
                rng.choice(forecast_vectors.shape[0], size=max_points, replace=False)
            ]

    return observed_vectors, forecast_vectors


def _kde_curve(
    values: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(values[:, None])
    log_density = kde.score_samples(grid[:, None])
    return np.exp(log_density)


def PlotComparisonDataDensity(
    eval_foldername: str | Path,
    nsample: int = 10,
    sample_index: int = 0,
    feature_index: int | Sequence[int] | None = None,
    bins: int = 200,
    bandwidth: float | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot 1D marginal data density for real vs. generated values.
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

    observed_values = observed_data[:, :, selected_features]
    observed_values = observed_values[forecast_mask[:, :, selected_features].bool()]
    forecast_values = forecasted_data[:, :, :, selected_features]
    forecast_mask_samples = forecast_mask[:, :, selected_features].unsqueeze(1)
    forecast_mask_samples = forecast_mask_samples.expand(
        -1, nsample, -1, -1
    ).reshape(forecasted_data.shape[0] * nsample, -1, len(selected_features))
    forecast_values = forecast_values.reshape(
        forecasted_data.shape[0] * nsample, -1, len(selected_features)
    )
    forecast_values = forecast_values[forecast_mask_samples.bool()]

    if observed_values.numel() == 0 or forecast_values.numel() == 0:
        raise ValueError("Insufficient data to plot density curves.")

    observed_np = observed_values.numpy()
    forecast_np = forecast_values.numpy()
    combined = np.concatenate([observed_np, forecast_np])
    data_min, data_max = combined.min(), combined.max()
    padding = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    grid = np.linspace(data_min - padding, data_max + padding, bins)
    if bandwidth is None:
        bandwidth = 0.2 * np.std(combined) if np.std(combined) > 0 else 1.0

    observed_density = _kde_curve(observed_np, grid, bandwidth)
    forecast_density = _kde_curve(forecast_np, grid, bandwidth)

    plt.figure(figsize=figsize)
    plt.plot(grid, observed_density, label="Original", color="tab:red")
    plt.plot(grid, forecast_density, label="Generated", color="tab:blue", linestyle="--")
    plt.title("Data Density Comparison")
    plt.xlabel("Data Value")
    plt.ylabel("Data Density Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotComparisonPCA(
    eval_foldername: str | Path,
    nsample: int = 10,
    feature_index: int | Sequence[int] | None = None,
    max_points: int = 2000,
    seed: int = 42,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot PCA projection (PC1/PC2) of flattened time-series samples.
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

    observed_vectors, forecast_vectors = _flatten_series(
        observed_data,
        observed_mask,
        forecasted_data,
        forecast_mask,
        selected_features,
        nsample,
        max_points,
        seed,
    )

    combined = np.vstack([observed_vectors, forecast_vectors])
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(combined)
    n_obs = observed_vectors.shape[0]
    obs_embed = embedding[:n_obs]
    gen_embed = embedding[n_obs:]

    plt.figure(figsize=figsize)
    plt.scatter(obs_embed[:, 0], obs_embed[:, 1], s=10, alpha=0.6, label="Original")
    plt.scatter(
        gen_embed[:, 0],
        gen_embed[:, 1],
        s=10,
        alpha=0.6,
        label="Generated",
    )
    plt.title("PCA: Original vs Generated")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotComparisonTSNE(
    eval_foldername: str | Path,
    nsample: int = 10,
    feature_index: int | Sequence[int] | None = None,
    max_points: int = 2000,
    seed: int = 42,
    perplexity: float = 30.0,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot t-SNE comparison between real and generated flattened series.
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

    observed_vectors, forecast_vectors = _flatten_series(
        observed_data,
        observed_mask,
        forecasted_data,
        forecast_mask,
        selected_features,
        nsample,
        max_points,
        seed,
    )

    data = np.vstack([observed_vectors, forecast_vectors])
    n_samples = data.shape[0]
    if n_samples < 2:
        raise ValueError("Not enough samples for t-SNE.")
    effective_perplexity = min(perplexity, max(1, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
    )
    embedding = tsne.fit_transform(data)
    n_obs = observed_vectors.shape[0]
    obs_embed = embedding[:n_obs]
    gen_embed = embedding[n_obs:]

    plt.figure(figsize=figsize)
    plt.scatter(obs_embed[:, 0], obs_embed[:, 1], s=10, alpha=0.6, label="Original")
    plt.scatter(
        gen_embed[:, 0],
        gen_embed[:, 1],
        s=10,
        alpha=0.6,
        label="Generated",
    )
    plt.title("t-SNE: Original vs Generated")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _plot_data_density(
    ax: plt.Axes,
    forecasted_data: torch.Tensor,
    forecast_mask: torch.Tensor,
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    nsample: int,
    selected_features: Sequence[int],
    bins: int,
) -> None:
    observed_values = observed_data[:, :, selected_features]
    observed_values = observed_values[forecast_mask[:, :, selected_features].bool()]
    forecast_values = forecasted_data[:, :, :, selected_features]
    forecast_mask_samples = forecast_mask[:, :, selected_features].unsqueeze(1)
    forecast_mask_samples = forecast_mask_samples.expand(
        -1, nsample, -1, -1
    ).reshape(forecasted_data.shape[0] * nsample, -1, len(selected_features))
    forecast_values = forecast_values.reshape(
        forecasted_data.shape[0] * nsample, -1, len(selected_features)
    )
    forecast_values = forecast_values[forecast_mask_samples.bool()]

    if observed_values.numel() == 0 or forecast_values.numel() == 0:
        raise ValueError("Insufficient data to plot density curves.")

    observed_np = observed_values.numpy()
    forecast_np = forecast_values.numpy()
    combined = np.concatenate([observed_np, forecast_np])
    data_min, data_max = combined.min(), combined.max()
    padding = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    grid = np.linspace(data_min - padding, data_max + padding, bins)
    bandwidth = 0.2 * np.std(combined) if np.std(combined) > 0 else 1.0

    observed_density = _kde_curve(observed_np, grid, bandwidth)
    forecast_density = _kde_curve(forecast_np, grid, bandwidth)

    ax.plot(grid, observed_density, label="Original", color="tab:red")
    ax.plot(grid, forecast_density, label="Generated", color="tab:blue", linestyle="--")
    ax.set_xlabel("Data Value")
    ax.set_ylabel("Data Density Estimate")
    ax.grid(True)


def _plot_pca(
    ax: plt.Axes,
    forecasted_data: torch.Tensor,
    forecast_mask: torch.Tensor,
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    selected_features: Sequence[int],
    nsample: int,
    max_points: int,
    seed: int,
) -> None:
    observed_vectors, forecast_vectors = _flatten_series(
        observed_data,
        observed_mask,
        forecasted_data,
        forecast_mask,
        selected_features,
        nsample,
        max_points,
        seed,
    )

    combined = np.vstack([observed_vectors, forecast_vectors])
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(combined)
    n_obs = observed_vectors.shape[0]
    obs_embed = embedding[:n_obs]
    gen_embed = embedding[n_obs:]

    ax.scatter(obs_embed[:, 0], obs_embed[:, 1], s=10, alpha=0.6, label="Original")
    ax.scatter(gen_embed[:, 0], gen_embed[:, 1], s=10, alpha=0.6, label="Generated")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)


def _plot_tsne(
    ax: plt.Axes,
    forecasted_data: torch.Tensor,
    forecast_mask: torch.Tensor,
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    selected_features: Sequence[int],
    nsample: int,
    max_points: int,
    seed: int,
    perplexity: float,
    random_state: int,
) -> None:
    observed_vectors, forecast_vectors = _flatten_series(
        observed_data,
        observed_mask,
        forecasted_data,
        forecast_mask,
        selected_features,
        nsample,
        max_points,
        seed,
    )

    data = np.vstack([observed_vectors, forecast_vectors])
    n_samples = data.shape[0]
    if n_samples < 2:
        raise ValueError("Not enough samples for t-SNE.")
    effective_perplexity = min(perplexity, max(1, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
    )
    embedding = tsne.fit_transform(data)
    n_obs = observed_vectors.shape[0]
    obs_embed = embedding[:n_obs]
    gen_embed = embedding[n_obs:]

    ax.scatter(obs_embed[:, 0], obs_embed[:, 1], s=10, alpha=0.6, label="Original")
    ax.scatter(gen_embed[:, 0], gen_embed[:, 1], s=10, alpha=0.6, label="Generated")
    ax.grid(True)


def PlotComparisonModelGrid(
    eval_foldernames: Sequence[str | Path],
    nsample: int = 10,
    feature_index: int | Sequence[int] | None = None,
    bins: int = 50,
    max_points: int = 5000,
    seed: int = 42,
    perplexity: float = 30.0,
    random_state: int = 42,
    figsize: tuple[int, int] | None = None,
    model_names: Sequence[str] | None = None,
) -> None:
    """
    Plot a comparison grid where rows are plot types and columns are models.
    """
    if not eval_foldernames:
        raise ValueError("At least one evaluation folder name is required.")

    n_models = len(eval_foldernames)
    nrows = 3
    if figsize is None:
        figsize = (4 * n_models, 12)
    fig, axes = plt.subplots(nrows=nrows, ncols=n_models, figsize=figsize)

    if n_models == 1:
        axes = axes.reshape(nrows, 1)

    for col, eval_foldername in enumerate(eval_foldernames):
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

        _plot_data_density(
            axes[0, col],
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            nsample,
            selected_features,
            bins,
        )
        _plot_pca(
            axes[1, col],
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            selected_features,
            nsample,
            max_points,
            seed,
        )
        _plot_tsne(
            axes[2, col],
            forecasted_data,
            forecast_mask,
            observed_data,
            observed_mask,
            selected_features,
            nsample,
            max_points,
            seed,
            perplexity,
            random_state,
        )

        axes[0, col].set_title(model_names[col] if model_names else str(eval_foldername))

    row_labels = ["Data Density", "PCA", "t-SNE"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)

    for ax in axes.flatten():
        ax.legend()

    plt.tight_layout()
    plt.show()
