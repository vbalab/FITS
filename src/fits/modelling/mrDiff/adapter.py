from dataclasses import dataclass
from typing import Literal

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class MrDiffConfig(ModelConfig):
    """Configuration for the mr-Diff multi-resolution diffusion model.

    mr-Diff (Li et al., ICLR 2024) decomposes time series into multiple
    resolutions via seasonal-trend decomposition and uses coarse-to-fine
    trends as intermediate latent variables to guide denoising.
    """

    seq_len: int = 96
    feature_size: int = 7
    pred_len: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    num_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.1
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    n_resolutions: int = 3

    def model_kwargs(self) -> dict:  # noqa: ANN401
        """Map config fields to the underlying mr-Diff constructor args."""
        ...


class MrDiffAdapter(ForecastingModel):
    """Adapter wrapping the mr-Diff source code into :class:`ForecastingModel`."""

    def __init__(self, config: MrDiffConfig = MrDiffConfig()) -> None:
        super().__init__(config)
        self.config: MrDiffConfig = config
        # TODO: instantiate underlying mr-Diff model
        ...

    def forward(self, batch: ForecastingData) -> torch.Tensor: ...

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData: ...

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert :class:`ForecastingData` to mr-Diff's expected format.

        mr-Diff expects the full sequence ``[B, L, K]`` and a context mask.
        Internally it decomposes the input into multiple resolution levels
        before applying the diffusion process.
        """
        ...
