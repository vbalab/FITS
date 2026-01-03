from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

from fits.modelling.CSDI.model import CSDI_Forecasting
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig
from fits.dataframes.dataset import ForecastingData


@dataclass
class CSDIDiffusionConfig:
    layers: int = 6
    channels: int = 64  # d_model
    nheads: int = 8
    diffusion_embedding_dim: int = 128
    beta_start: float = 0.0001
    beta_end: float = 0.1
    num_steps: int = 100
    schedule: Literal["quad", "linear"] = "quad"


@dataclass
class CSDIConfig(ModelConfig):
    target_dim: int = 36
    time_embedding_dim: int = 128
    feature_embedding_dim: int = 16
    is_unconditional: bool = False
    target_strategy: Literal["mix", "random", "historical"] = "mix"
    num_sample_features: int = 64  # > 36 -> no feature sampling
    diffusion: CSDIDiffusionConfig = field(default_factory=CSDIDiffusionConfig)

    def as_csdi_dict(self) -> dict:
        return {
            "model": {
                "timeemb": self.time_embedding_dim,
                "featureemb": self.feature_embedding_dim,
                "is_unconditional": self.is_unconditional,
                "target_strategy": self.target_strategy,
                "num_sample_features": self.num_sample_features,
            },
            "diffusion": asdict(self.diffusion),
        }


class CSDIAdapter(ForecastingModel):
    def __init__(self, config: CSDIConfig = CSDIConfig()):
        super().__init__(config)
        self.target_dim = config.target_dim

        self.csdi_model = CSDI_Forecasting(
            config=config.as_csdi_dict(),
            device=self.device,
            target_dim=self.target_dim,
        ).to(self.device)

    def forward(self, batch: ForecastingData):
        if self.training:
            is_train = 1
        else:
            is_train = 0

        csdi_batch = self._adapt_batch(batch)
        return self.csdi_model(csdi_batch, is_train=is_train)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        csdi_batch = self._adapt_batch(batch)
        samples, observed_data, target_mask, observed_mask, time_points = (
            self.csdi_model.evaluate(csdi_batch, n_samples)
        )

        return ForecastedData(
            forecasted_data=samples.permute(0, 1, 3, 2),
            observed_data=observed_data.permute(
                0, 2, 1
            ),  # TODO: change to batch.observed_data
            observed_mask=observed_mask.permute(
                0, 2, 1
            ),  # TODO: change to batch.observed_mask
            forecast_mask=batch.forecast_mask.to(device=self.device, dtype=torch.float32),
            time_points=time_points,
        )

    def _adapt_batch(self, batch: ForecastingData) -> dict:
        """Convert :class:`ForecastingData` batches into CSDI's expected dict format."""
        assert isinstance(batch, ForecastingData)

        observed_data = batch.observed_data
        observed_mask = batch.observed_mask
        time_points = batch.time_points[
            ..., 0
        ]  # CSDI expects shape [B, L] for timepoints; ForecastingData stores [B, L, K].
        gt_mask = batch.observed_mask * (1 - batch.forecast_mask)
        # 0 - to be generated
        # 1 - known at generation

        return {
            "observed_data": observed_data.to(dtype=torch.float32, device=self.device),
            "observed_mask": observed_mask.to(dtype=torch.float32, device=self.device),
            "timepoints": time_points.to(dtype=torch.float32, device=self.device),
            "gt_mask": gt_mask.to(dtype=torch.float32, device=self.device),
        }
