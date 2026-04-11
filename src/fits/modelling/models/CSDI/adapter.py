from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.model import CSDI_Forecasting


@dataclass
class CSDIDiffusionConfig:
    layers: int = 4
    channels: int = 64  # d_model
    nheads: int = 8
    diffusion_embedding_dim: int = 128
    beta_start: float = 0.0001
    beta_end: float = 0.1
    num_steps: int = 50
    schedule: Literal["quad", "linear"] = "quad"


@dataclass
class CSDIConfig(ModelConfig):
    feature_size: int = 7
    time_embedding_dim: int = 128
    feature_embedding_dim: int = 16
    is_unconditional: bool = False
    target_strategy: Literal["mix", "random", "historical"] = "mix"
    num_sample_features: int = 64  # > feature_size -> no feature sampling
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
        self.target_dim = config.feature_size

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
        # CSDI outputs [B, n_samples, K, L] (feature-first); permute to [B, n_samples, L, K]
        # so it matches the [B, L, K] convention used by gt_mask and observed_data below.
        samples = samples.permute(0, 1, 3, 2)

        mask = (csdi_batch["gt_mask"] > 0.1).unsqueeze(1)
        obs = csdi_batch["observed_data"].unsqueeze(1)
        samples = torch.where(mask, obs, samples)

        return ForecastedData(
            forecasted_data=samples,
            observed_data=batch.observed_data.to(
                device=self.device, dtype=torch.float32
            ),
            observed_mask=batch.observed_mask.to(
                device=self.device, dtype=torch.float32
            ),
            forecast_mask=batch.forecast_mask.to(
                device=self.device, dtype=torch.float32
            ),
            time_points=time_points,
            base_level=batch.base_level,
        )

    def _adapt_batch(self, batch: ForecastingData) -> dict:
        """Convert :class:`ForecastingData` batches into CSDI's expected dict format."""
        assert isinstance(batch, ForecastingData)

        observed_data = batch.observed_data.to(dtype=torch.float32, device=self.device)
        observed_mask = batch.observed_mask.to(dtype=torch.float32, device=self.device)
        time_points = batch.time_points[
            ..., 0
        ]  # CSDI expects shape [B, L] for timepoints; ForecastingData stores [B, L, K].
        gt_mask = (batch.observed_mask * (1 - batch.forecast_mask)).to(
            dtype=torch.float32, device=self.device
        )
        # 0 - to be generated
        # 1 - known at generation

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "timepoints": time_points.to(dtype=torch.float32, device=self.device),
            "gt_mask": gt_mask,
        }
