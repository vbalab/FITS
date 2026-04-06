from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig
from .source.model import Model as ITransformerModel


@dataclass
class ITransformerConfig(ModelConfig):
    """Configuration for the iTransformer inverted-attention model.

    iTransformer (Liu et al., ICLR 2024 Spotlight) treats each variate as a
    token, embedding its full time series into d_model dimensions, then
    applies self-attention across variates — inverting the conventional
    Transformer token axis.

    Source files used verbatim (pure PyTorch):
      - model.py                         — Model (iTransformer.py)
      - layers/Transformer_EncDec.py     — Encoder, EncoderLayer
      - layers/Embed.py                  — DataEmbedding_inverted
      - layers/SelfAttention_iTransformer.py — FullAttention, AttentionLayer
        (extracted from SelfAttention_Family.py; other classes in that file
        depend on reformer_pytorch/einops/utils.masking and are not used
        by iTransformer)

    Three ``from layers.X`` absolute imports in model.py were converted to
    relative imports — zero algorithmic changes.

    iTransformer is deterministic: :meth:`evaluate` repeats the single-point
    forecast ``n_samples`` times to conform to the probabilistic interface.
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7  # number of variates (enc_in / N)
    seq_len: int = 96  # context window length
    pred_len: int = 24  # forecast horizon

    # --- Transformer ---------------------------------------------------------
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2  # number of encoder layers
    d_ff: int = 256
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    factor: int = 1  # attention factor (unused by FullAttention)

    # --- Normalization -------------------------------------------------------
    use_norm: bool = True  # instance normalization inside the model

    # --- Preprocessing -------------------------------------------------------
    first_differences: bool = False

    def as_configs(self) -> SimpleNamespace:
        """Build the SimpleNamespace expected by iTransformer's Model.__init__."""
        return SimpleNamespace(
            seq_len=self.seq_len - self.pred_len,  # context length
            pred_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            factor=self.factor,
            use_norm=self.use_norm,
            output_attention=False,
            embed="timeF",  # embedding type (x_mark unused → no effect)
            freq="h",  # frequency (x_mark unused → no effect)
            class_strategy="projection",  # referenced in __init__ but unused in forecast
        )


class ITransformerAdapter(ForecastingModel):
    """Adapter wrapping iTransformer into :class:`ForecastingModel`.

    Data flow:

    Training:
        observed_data [B, L, K]
            → context [B, ctx_len, K]     (first L - pred_len steps)
            → Model(x_enc, None, None, None) → [B, pred_len, K]
            → MSE vs target [B, pred_len, K]

    Inference (deterministic → repeated n_samples times):
        same forward pass, no sampling loop.
        forecasted_data: [B, n_samples, L, K]
            - context region: original observed values
            - horizon region: model predictions (repeated)

    Normalization note: iTransformer applies its own instance normalization
    internally when ``use_norm=True``. When ``first_differences=True``, the
    adapter converts to differences first; iTransformer's norm then operates
    in difference space, which is effectively stationary.
    """

    def __init__(self, config: ITransformerConfig = ITransformerConfig()) -> None:
        super().__init__(config)
        self.config: ITransformerConfig = config
        # iTransformer source — unchanged except relative imports
        self.model = ITransformerModel(config.as_configs()).to(self.device)

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        ctx, target = self._adapt_batch(batch)
        pred = self._run_model(ctx)  # [B, pred_len, K]
        obs_mask = batch.forecast_mask.to(device=self.device, dtype=torch.float32)
        obs_mask = obs_mask[:, -self.config.pred_len :]  # [B, pred_len, K]
        loss = F.mse_loss(pred * obs_mask, target * obs_mask, reduction="sum")
        return loss / obs_mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        ctx, _ = self._adapt_batch(batch)
        pred = self._run_model(ctx)  # [B, pred_len, K]

        if self.config.first_differences:
            # Restore level space: base = last context value in original space
            pred_len = self.config.pred_len
            base = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                :, -(pred_len + 1) : -(pred_len)
            ]  # [B, 1, K]
            pred = base + pred.cumsum(dim=1)

        # Build full sequence: observed context + predicted horizon
        ctx_len = self.config.seq_len - self.config.pred_len
        context = batch.observed_data.to(device=self.device, dtype=torch.float32)[
            :, :ctx_len
        ]
        full = torch.cat([context, pred], dim=1)  # [B, L, K]

        # Deterministic model: repeat the same forecast n_samples times
        stacked = full.unsqueeze(1).expand(
            -1, n_samples, -1, -1
        )  # [B, n_samples, L, K]

        return ForecastedData(
            forecasted_data=stacked,
            observed_data=batch.observed_data.to(
                device=self.device, dtype=torch.float32
            ),
            observed_mask=batch.observed_mask.to(
                device=self.device, dtype=torch.float32
            ),
            forecast_mask=batch.forecast_mask.to(
                device=self.device, dtype=torch.float32
            ),
            time_points=batch.time_points[..., 0].to(
                device=self.device, dtype=torch.float32
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_model(self, ctx: torch.Tensor) -> torch.Tensor:
        """Run iTransformer on context [B, ctx_len, K] → [B, pred_len, K].

        x_mark_enc, x_dec, x_mark_dec are all None — iTransformer's
        DataEmbedding_inverted handles None x_mark, and x_dec/x_mark_dec
        are not accessed in the forecast() path.
        """
        return self.model(ctx, None, None, None)  # [B, pred_len, K]

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, target) in [B, ctx_len/pred_len, K].

        If ``first_differences=True``, both context and target are
        converted to differences so the MSE loss stays in difference space.
        iTransformer's internal ``use_norm`` then handles remaining
        instance normalization.
        """
        pred_len = self.config.pred_len
        x = batch.observed_data.to(device=self.device, dtype=torch.float32)

        if self.config.first_differences:
            x = self._first_differences(x)

        ctx = x[:, :-pred_len]  # [B, ctx_len, K]
        target = x[:, -pred_len:]  # [B, pred_len, K]
        return ctx, target

    @staticmethod
    def _first_differences(data: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros_like(data)
        diffs[:, 1:] = data[:, 1:] - data[:, :-1]
        return diffs
