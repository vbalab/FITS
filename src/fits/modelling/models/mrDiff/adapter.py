from dataclasses import dataclass, field

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.model import MrDiff, build_trend_sequence


@dataclass
class MrDiffConfig(ModelConfig):
    """Configuration for mr-Diff (Multi-Resolution Diffusion).

    mr-Diff (Shen, Chen & Kwok, ICLR 2024) uses multi-resolution seasonal-
    trend decomposition: the forecast target is progressively smoothed via
    average pooling into S coarse-to-fine trend levels. A separate DDPM
    denoising network is trained per level, with the coarser output used as
    conditioning for the finer level. Inference proceeds from coarsest to
    finest.

    Implemented from the paper — no official source code exists.
    Paper: https://openreview.net/forum?id=mmjnr0G8ZY

    Key paper hyperparameters (Table 1 / Appendix):
      n_stages=5, diff_steps=100, beta_start=1e-4, beta_end=0.1
      kernel_sizes default to [3, 5, 11, 21] for 5 stages
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7
    seq_len: int = 96  # total window length (context + horizon)
    pred_len: int = 24  # H — forecast horizon

    # --- Multi-resolution decomposition -------------------------------------
    n_stages: int = 5  # S — number of resolution stages
    # τ_s kernel sizes for average-pooling (S-1 values)
    # Default: [3, 5, 11, 21] — progressively wider as in paper
    kernel_sizes: list = field(default_factory=lambda: [3, 5, 11, 21])

    # --- Diffusion schedule --------------------------------------------------
    diff_steps: int = 100  # K — diffusion steps per stage
    beta_start: float = 1e-4
    beta_end: float = 0.1

    # --- Denoising network ---------------------------------------------------
    d_model: int = 48  # channel dim inside denoising nets
    d_emb: int = 48  # diffusion step embedding dim
    n_layers: int = 2  # conv layers per encoder/decoder
    conv_kernel: int = 3  # conv kernel size

    # --- Training ------------------------------------------------------------
    mixup_prob: float = 0.5  # future-mixup probability during training


class MrDiffAdapter(ForecastingModel):
    """Adapter wrapping MrDiff into :class:`ForecastingModel`.

    Training (multi-stage, per-batch):
        1. Decompose Y into [Y_0, ..., Y_{S-1}] via average-pooling cascade.
        2. For each stage s (finest→coarsest):
           - Build conditioning c_s from X_s + Y_{s+1}^0 + future-mixup.
           - Compute MSE diffusion loss for stage s.
        3. Total loss = mean over stages.

    Inference (coarsest→finest):
        1. Decompose X into [X_0, ..., X_{S-1}].
        2. Stage s=S-1 (coarsest): condition on X_{S-1} + zero coarser trend.
        3. Stage s=S-2 down to s=0: condition on X_s + previous Ŷ_{s+1}^0.
        4. Return Ŷ_0^0 as the final forecast.
        5. Repeat n_samples times → [B, n_samples, L, K].
    """

    def __init__(self, config: MrDiffConfig = MrDiffConfig()) -> None:
        super().__init__(config)
        self.config: MrDiffConfig = config
        self.model = MrDiff(
            n_features=config.feature_size,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            n_stages=config.n_stages,
            diff_steps=config.diff_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            kernel_sizes=config.kernel_sizes,
            d_model=config.d_model,
            d_emb=config.d_emb,
            n_layers=config.n_layers,
            conv_kernel=config.conv_kernel,
        ).to(self.device)

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        x, y = self._adapt_batch(batch)  # [B, ctx, N], [B, H, N]

        # Decompose Y into S levels (finest=0 → coarsest=S-1)
        y_trends = build_trend_sequence(y, self.config.kernel_sizes)
        # Decompose X into S levels (same kernel sizes)
        x_trends = build_trend_sequence(x, self.config.kernel_sizes)

        total_loss = torch.tensor(0.0, device=self.device)
        for s in range(self.config.n_stages):
            y_s = y_trends[s]  # [B, H, N] finest at s=0
            x_s = x_trends[s]  # [B, ctx, N]
            # Coarser trend: Y_{s+1}^0 (or zeros at the coarsest stage)
            y_coarser = (
                y_trends[s + 1]
                if s + 1 < self.config.n_stages
                else torch.zeros_like(y_s)
            )
            cond = self.model.build_condition(
                x_s=x_s,
                y_coarser=y_coarser,
                stage=s,
                y_s0=y_s,
                mixup_prob=self.config.mixup_prob,
            )
            total_loss = total_loss + self.model.stage_loss(y_s, cond, stage=s)

        return total_loss / self.config.n_stages

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        x, _ = self._adapt_batch(batch)  # [B, ctx, N]

        # Decompose X into S levels
        x_trends = build_trend_sequence(x, self.config.kernel_sizes)

        all_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            # Inference: coarsest stage first, feed estimate down to finest
            y_coarser_est = torch.zeros(
                x.shape[0],
                self.config.pred_len,
                self.config.feature_size,
                device=self.device,
            )
            for s in reversed(range(self.config.n_stages)):
                cond = self.model.build_condition(
                    x_s=x_trends[s],
                    y_coarser=y_coarser_est,
                    stage=s,
                    y_s0=None,  # no ground truth at inference
                )
                y_coarser_est = self.model.stage_sample(cond, stage=s)

            pred = y_coarser_est  # Ŷ_0^0  [B, H, N]

            ctx_len = self.config.seq_len - self.config.pred_len
            context = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                :, :ctx_len
            ]
            full = torch.cat([context, pred], dim=1)  # [B, L, N]
            all_samples.append(full)

        stacked = torch.stack(all_samples, dim=1)  # [B, n_samples, L, N]

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

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, target) in [B, ctx/pred_len, N]."""
        pred_len = self.config.pred_len
        x_full = batch.observed_data.to(device=self.device, dtype=torch.float32)
        return x_full[:, :-pred_len], x_full[:, -pred_len:]
