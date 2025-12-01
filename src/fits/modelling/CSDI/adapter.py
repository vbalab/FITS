# mypy: ignore-errors

from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

from fits.modelling.CSDI.model import CSDI_Forecasting
from fits.modelling.framework import ForecastingModel, ModelConfig
from fits.data.dataset import ForecastingData


@dataclass
class CSDIDiffusionConfig:
    """Typed configuration for the diffusion backbone inside CSDI."""

    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_steps: int = 50
    schedule: Literal["quad", "linear"] = "quad"
    is_linear: bool = True
    channels: int = 64
    diffusion_embedding_dim: int = 128
    nheads: int = 8
    layers: int = 4


@dataclass
class CSDIConfig(ModelConfig):
    """Typed configuration for the :class:`CSDIAdapter`."""

    target_dim: int = 1
    time_embedding_dim: int = 128
    feature_embedding_dim: int = 128
    is_unconditional: bool = False
    target_strategy: Literal["mix", "random", "historical"] = "mix"
    num_sample_features: int = 1
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
    """Adapter that wraps the original CSDI implementation into :class:`ForecastingModel`."""

    def __init__(self, config: CSDIConfig):
        super().__init__(config)
        self.target_dim = config.target_dim
        self.csdi_model = CSDI_Forecasting(
            config=config.as_csdi_dict(), device=self.device, target_dim=self.target_dim
        ).to(self.device)

    def forward(self, batch, is_train: int = 1):
        csdi_batch = self._adapt_batch(batch)
        return self.csdi_model(csdi_batch, is_train=is_train)

    def evaluate(self, batch, n_samples: int):
        return self.csdi_model.evaluate(batch, n_samples)

    def _adapt_batch(self, batch) -> dict:
        """Convert :class:`ForecastingData` batches into CSDI's expected dict format."""

        if isinstance(batch, ForecastingData):
            observed_data = batch.observed_data
            observed_mask = batch.observed_mask
            time_points = batch.time_points
        else:
            # Assume mapping-like batches from a custom collate_fn.
            observed_data = batch["observed_data"]
            observed_mask = batch["observed_mask"]
            time_points = (
                batch["time_points"] if "time_points" in batch else batch["timepoints"]
            )

        # CSDI expects shape [B, L] for timepoints; ForecastingData stores [B, L, K].
        if time_points.dim() == 3:
            time_points = time_points[..., 0]

        gt_mask = observed_mask

        return {
            "observed_data": observed_data.to(dtype=torch.float32, device=self.device),
            "observed_mask": observed_mask.to(dtype=torch.float32, device=self.device),
            "timepoints": time_points.to(dtype=torch.float32, device=self.device),
            "gt_mask": gt_mask.to(dtype=torch.float32, device=self.device),
        }
