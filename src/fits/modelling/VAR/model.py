from dataclasses import dataclass

import torch
from torch import nn

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class VARConfig(ModelConfig):
    seq_len: int = 48
    horizon: int = 6
    feature_size: int = 36
    lag_order: int = 3
    seasonal_period: int = 24


class SeasonalVAR(ForecastingModel):
    def __init__(self, config: VARConfig = VARConfig()):
        super().__init__(config)
        self.config: VARConfig = config

        self.coefficients = nn.Parameter(
            0.01
            * torch.randn(
                config.lag_order,
                config.feature_size,
                config.feature_size,
            )
        )
        self.seasonal_embedding = nn.Parameter(
            torch.zeros(config.seasonal_period, config.feature_size)
        )
        self.bias = nn.Parameter(torch.zeros(config.feature_size))

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        observed_data = batch.observed_data.to(self.device, dtype=torch.float32)
        observed_mask = batch.observed_mask.to(self.device, dtype=torch.float32)
        forecast_mask = batch.forecast_mask.to(self.device, dtype=torch.float32)
        time_points = batch.time_points.to(self.device, dtype=torch.float32)

        predictions = self._predict_sequence(observed_data, time_points)

        mask = observed_mask * forecast_mask
        if mask.sum() == 0:
            mask = observed_mask

        loss = ((predictions - observed_data) ** 2) * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        self.eval()

        observed_data = batch.observed_data.to(self.device, dtype=torch.float32)
        time_points = batch.time_points.to(self.device, dtype=torch.float32)
        forecast_mask = batch.forecast_mask.to(self.device, dtype=torch.float32)
        batch_size, seq_len, _ = observed_data.shape

        generated = observed_data.clone()
        for t in range(self.config.lag_order, seq_len):
            step_prediction = self._predict_step(
                generated[:, t - self.config.lag_order : t, :],
                time_points[:, t, 0],
            )
            should_forecast = forecast_mask[:, t, :] > 0
            if should_forecast.any():
                generated[:, t, :] = torch.where(
                    should_forecast,
                    step_prediction,
                    generated[:, t, :],
                )

        stacked = generated.unsqueeze(1).repeat(1, n_samples, 1, 1)

        return ForecastedData(
            forecasted_data=stacked,
            forecast_mask=batch.forecast_mask.to(self.device, dtype=torch.float32),
            observed_data=batch.observed_data.to(self.device, dtype=torch.float32),
            observed_mask=batch.observed_mask.to(self.device, dtype=torch.float32),
            time_points=batch.time_points[..., 0].to(self.device, dtype=torch.float32),
        )

    def _predict_sequence(
        self, observed_data: torch.Tensor, time_points: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, feature_size = observed_data.shape
        predictions = torch.zeros(
            batch_size,
            seq_len,
            feature_size,
            device=observed_data.device,
            dtype=observed_data.dtype,
        )

        for t in range(self.config.lag_order, seq_len):
            predictions[:, t, :] = self._predict_step(
                observed_data[:, t - self.config.lag_order : t, :],
                time_points[:, t, 0],
            )

        return predictions

    def _predict_step(
        self, history_window: torch.Tensor, time_point: torch.Tensor
    ) -> torch.Tensor:
        season_index = torch.remainder(
            time_point.to(torch.long), self.config.seasonal_period
        )
        seasonal = self.seasonal_embedding[season_index]
        autoregressive = torch.einsum("bpk,pkh->bh", history_window, self.coefficients)
        return autoregressive + seasonal + self.bias
