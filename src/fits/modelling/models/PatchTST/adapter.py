from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.layers.PatchTST_backbone import PatchTST_backbone


@dataclass
class PatchTSTConfig(ModelConfig):
    """Configuration for PatchTST.

    PatchTST (Nie et al., ICLR 2023) segments time series into
    subseries-level patches and processes them with a channel-independent
    Transformer encoder, achieving SOTA on long-horizon benchmarks.

    Source files used verbatim (pure PyTorch):
      - layers/PatchTST_backbone.py  — full backbone, encoder, attention
      - layers/PatchTST_layers.py    — positional encodings, decomposition
      - layers/RevIN.py              — reversible instance normalization

    Only the two ``from layers.X import Y`` absolute imports were converted
    to relative imports (``from .X import Y``) so the files work inside a
    Python package — zero algorithmic changes.

    PatchTST is deterministic: :meth:`evaluate` repeats the single-point
    forecast ``n_samples`` times to conform to the probabilistic interface.
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7  # number of variates (c_in)
    seq_len: int = 96  # total window length (context + horizon)
    pred_len: int = 24  # forecast horizon (last pred_len steps)

    # --- Patching ------------------------------------------------------------
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"  # pad end so all time steps are covered

    # --- Transformer ---------------------------------------------------------
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 192
    dropout: float = 0.2
    attn_dropout: float = 0.0
    fc_dropout: float = 0.2
    head_dropout: float = 0.0

    # --- Architecture options ------------------------------------------------
    individual: bool = False  # channel-independent heads
    revin: bool = True  # reversible instance normalization
    affine: bool = True  # learnable RevIN affine parameters
    subtract_last: bool = False  # RevIN: subtract last value instead of mean
    res_attention: bool = True  # residual attention across layers

    def backbone_kwargs(self) -> dict:  # type: ignore[return]
        ctx_len = self.seq_len - self.pred_len
        return {
            "c_in": self.feature_size,
            "context_window": ctx_len,
            "target_window": self.pred_len,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "padding_patch": self.padding_patch,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "fc_dropout": self.fc_dropout,
            "head_dropout": self.head_dropout,
            "individual": self.individual,
            "revin": self.revin,
            "affine": self.affine,
            "subtract_last": self.subtract_last,
            "res_attention": self.res_attention,
        }


class PatchTSTAdapter(ForecastingModel):
    """Adapter wrapping PatchTST_backbone into :class:`ForecastingModel`.

    Data flow:

    Training:
        observed_data [B, L, K]
            → context [B, ctx, K]  (L - pred_len steps)
            → permute → [B, K, ctx]
            → PatchTST_backbone → [B, K, pred_len]
            → permute → [B, pred_len, K]
            → MSE vs target [B, pred_len, K]

    Inference (deterministic → repeated n_samples times):
        same forward pass, no sampling loop.
        forecasted_data: [B, n_samples, L, K]
            - context region: original observed values
            - horizon region: model predictions (repeated)
    """

    def __init__(self, config: PatchTSTConfig = PatchTSTConfig()) -> None:
        super().__init__(config)
        self.config: PatchTSTConfig = config
        # PatchTST source — unchanged except two relative-import lines
        self.model = PatchTST_backbone(**config.backbone_kwargs()).to(self.device)

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        ctx, target = self._adapt_batch(batch)
        pred = self._run_model(ctx)  # [B, pred_len, K]
        # MSE on observed target positions
        obs_mask = batch.forecast_mask.to(device=self.device, dtype=torch.float32)
        obs_mask = obs_mask[:, -self.config.pred_len :]  # [B, pred_len, K]
        loss = F.mse_loss(pred * obs_mask, target * obs_mask, reduction="sum")
        return loss / obs_mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        ctx, _ = self._adapt_batch(batch)
        pred = self._run_model(ctx)  # [B, pred_len, K]

        # Build full sequence: observed context + predicted horizon
        ctx_len = self.config.seq_len - self.config.pred_len
        context = batch.observed_data.to(device=self.device, dtype=torch.float32)[
            :, :ctx_len
        ]  # [B, ctx_len, K]
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
            base_level=batch.base_level,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_model(self, ctx: torch.Tensor) -> torch.Tensor:
        """Run backbone on context [B, ctx_len, K] → predictions [B, pred_len, K]."""
        # PatchTST_backbone expects [B, K, ctx_len] (channel-first)
        x = ctx.permute(0, 2, 1)  # [B, K, ctx_len]
        out = self.model(x)  # [B, K, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, K]

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, target) in [B, ctx_len/pred_len, K].

        RevIN inside PatchTST_backbone handles instance normalization.
        """
        pred_len = self.config.pred_len
        x = batch.observed_data.to(device=self.device, dtype=torch.float32)

        ctx = x[:, :-pred_len]  # [B, ctx_len, K]
        target = x[:, -pred_len:]  # [B, pred_len, K]
        return ctx, target
