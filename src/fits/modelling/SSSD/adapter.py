from dataclasses import dataclass
from typing import Literal

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class SSSDConfig(ModelConfig):
    """Configuration for the SSSD (Structured State Space Diffusion) model.

    SSSD (Alcaraz & Strodthoff, 2022) replaces the transformer backbone of
    CSDI with structured state space (S4) layers inside a DiffWave-style
    residual architecture, reducing computational complexity.
    """

    feature_size: int = 7
    diffusion_steps: int = 200
    s4_d_state: int = 64
    s4_d_model: int = 64
    n_residual_layers: int = 36
    residual_channels: int = 256
    beta_start: float = 0.0001
    beta_end: float = 0.5
    beta_schedule: Literal["linear", "quad"] = "quad"

    def model_kwargs(self) -> dict:  # noqa: ANN401
        """Map config fields to the underlying SSSD constructor args."""
        ...


class SSSDAdapter(ForecastingModel):
    """Adapter wrapping the SSSD source code into :class:`ForecastingModel`."""

    def __init__(self, config: SSSDConfig = SSSDConfig()) -> None:
        super().__init__(config)
        self.config: SSSDConfig = config
        # TODO: instantiate underlying SSSDS4 model
        ...

    def forward(self, batch: ForecastingData) -> torch.Tensor: ...

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData: ...

    def _adapt_batch(self, batch: ForecastingData) -> dict[str, torch.Tensor]:
        """Convert :class:`ForecastingData` to SSSD's expected format.

        SSSD uses the same mask-based conditioning as CSDI: a ``cond_mask``
        tensor marks which time-feature positions are observed context, and
        the model generates the remaining positions via reverse diffusion.
        """
        ...
