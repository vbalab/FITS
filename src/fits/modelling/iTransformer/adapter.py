from dataclasses import dataclass

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class ITransformerConfig(ModelConfig):
    """Configuration for the iTransformer inverted-attention model.

    iTransformer (Liu et al., ICLR 2024 Spotlight) treats each variate as a
    token and applies attention across variables rather than time steps.
    """

    seq_len: int = 96
    feature_size: int = 7
    pred_len: int = 24
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    def model_kwargs(self) -> dict:  # noqa: ANN401
        """Map config fields to the underlying iTransformer constructor args."""
        ...


class ITransformerAdapter(ForecastingModel):
    """Adapter wrapping the iTransformer source code into :class:`ForecastingModel`.

    iTransformer is deterministic: :meth:`evaluate` repeats the single-point
    forecast ``n_samples`` times to conform to the probabilistic interface.
    """

    def __init__(self, config: ITransformerConfig = ITransformerConfig()) -> None:
        super().__init__(config)
        self.config: ITransformerConfig = config
        # TODO: instantiate underlying iTransformer model
        ...

    def forward(self, batch: ForecastingData) -> torch.Tensor: ...

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData: ...

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert :class:`ForecastingData` to iTransformer's expected format.

        iTransformer expects ``(x_history, x_mark)`` where ``x_history`` has
        shape ``[B, seq_len, K]``.  The model inverts tokens so that each of
        the ``K`` variates becomes a token of length ``seq_len``.
        """
        ...
