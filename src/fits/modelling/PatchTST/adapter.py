from dataclasses import dataclass

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class PatchTSTConfig(ModelConfig):
    """Configuration for the PatchTST deterministic transformer.

    PatchTST (Nie et al., ICLR 2023) segments time series into
    subseries-level patches and processes them with a channel-independent
    Transformer encoder.
    """

    seq_len: int = 96
    feature_size: int = 7
    pred_len: int = 24
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    individual: bool = False  # channel-independent heads

    def model_kwargs(self) -> dict:  # noqa: ANN401
        """Map config fields to the underlying PatchTST constructor args."""
        ...


class PatchTSTAdapter(ForecastingModel):
    """Adapter wrapping the PatchTST source code into :class:`ForecastingModel`.

    PatchTST is deterministic: :meth:`evaluate` repeats the single-point
    forecast ``n_samples`` times to conform to the probabilistic interface.
    """

    def __init__(self, config: PatchTSTConfig = PatchTSTConfig()) -> None:
        super().__init__(config)
        self.config: PatchTSTConfig = config
        # TODO: instantiate underlying PatchTST model
        ...

    def forward(self, batch: ForecastingData) -> torch.Tensor: ...

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData: ...

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert :class:`ForecastingData` to PatchTST's expected format.

        PatchTST expects ``(x_history, x_mark)`` where ``x_history`` is the
        context window ``[B, seq_len, K]`` and returns point forecasts
        ``[B, pred_len, K]``.
        """
        ...
