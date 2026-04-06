from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.g_backbone import SigmaEstimation
from .source.model import NsDiff as NsDiffModel
from .source.mu_backbone import Model as MuBackbone
from .source.nsdiff_utils import (
    cal_forward_noise,
    cal_sigma_tilde,
    p_sample_loop,
    q_sample,
)
from .source.sigma import wv_sigma_trailing

EPS = 10e-8


@dataclass
class NsDiffConfig(ModelConfig):
    """Configuration for NsDiff (Non-stationary Diffusion).

    NsDiff (Weiwei et al., ICML 2025) replaces the standard Additive Noise
    Model (ANM) assumption in DDPMs with a Location-Scale Noise Model (LSNM),
    where noise variance is context-dependent: Y = f(X) + sqrt(g(X)) * eps.
    An uncertainty-aware noise schedule dynamically adjusts noise levels to
    reflect time-varying data uncertainty.

    Three sub-models trained jointly (all registered as nn.Module attributes
    so optimizer sees all parameters via model.parameters()):
      - diffusion_model (NsDiff)   — conditional guided denoiser
      - mu_model (mu_backbone)     — Non-stationary Transformer predicting μ(x)
      - sigma_model (SigmaEstimation) — MLP predicting g(x) (context variance)

    Source files used verbatim (requires: torch-timeseries==0.1.10):
      - model.py          — NsDiff main model (src/models/NsDiff.py)
      - mu_backbone.py    — Non-stationary Transformer (src/layer/mu_backbone.py)
      - g_backbone.py     — SigmaEstimation (src/layer/g_backbone.py)
      - denoise.py        — ConditionalGuidedModel (src/layer/denoise.py)
      - nsdiff_utils.py   — LSNM diffusion math (src/layer/nsdiff_utils.py)
      - tmdm_diffusion_utils.py — make_beta_schedule (src/nn/tmdm_diffusion_utils.py)
      - sigma.py          — wv_sigma functions (src/utils/sigma.py)

    Import changes (4 lines total across 3 files): src.* → relative imports.
    torch_timeseries stays as pip package import, unchanged.
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7
    seq_len: int = 96  # total window length (context + horizon)
    pred_len: int = 24  # forecast horizon

    # --- Diffusion schedule --------------------------------------------------
    diff_steps: int = 20
    beta_start: float = 0.0001
    beta_end: float = 0.01
    beta_schedule: Literal["linear", "cosine", "quad"] = "linear"

    # --- Mu backbone (Non-stationary Transformer) ----------------------------
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 128
    factor: int = 3
    dropout: float = 0.05
    activation: str = "gelu"
    p_hidden_dims: list = field(default_factory=lambda: [64, 64])
    p_hidden_layers: int = 2

    # --- NsDiff denoiser embedding -------------------------------------------
    CART_input_x_embed_dim: int = 32

    # --- Sigma backbone (SigmaEstimation) ------------------------------------
    sigma_hidden_size: int = 64
    rolling_length: int = 96  # trailing window for sigma estimation

    # --- Preprocessing -------------------------------------------------------
    first_differences: bool = False

    def mu_backbone_configs(self) -> SimpleNamespace:
        """Build the configs namespace expected by mu_backbone.Model."""
        ctx_len = self.seq_len - self.pred_len
        label_len = ctx_len // 2
        return SimpleNamespace(
            seq_len=ctx_len,
            pred_len=self.pred_len,
            label_len=label_len,
            enc_in=self.feature_size,
            dec_in=self.feature_size,
            c_out=self.feature_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.dropout,
            activation=self.activation,
            embed="timeF",
            freq="h",
            output_attention=False,
            p_hidden_dims=self.p_hidden_dims,
            p_hidden_layers=self.p_hidden_layers,
        )

    def diffusion_model_configs(self) -> SimpleNamespace:
        """Build the configs namespace expected by NsDiff model."""
        ctx_len = self.seq_len - self.pred_len
        return SimpleNamespace(
            seq_len=ctx_len,
            pred_len=self.pred_len,
            enc_in=self.feature_size,
            timesteps=self.diff_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            CART_input_x_embed_dim=self.CART_input_x_embed_dim,
            embed="timeF",
            freq="h",
            dropout=self.dropout,
        )


class NsDiffAdapter(ForecastingModel):
    """Adapter wrapping NsDiff into :class:`ForecastingModel`.

    Architecture (three jointly-trained models):

        x [B, ctx, K]  ──►  mu_model  ──►  y_0_hat [B, pred, K]   (μ(x))
        x [B, ctx, K]  ──►  sigma_model ► gx [B, pred, K]          (g(x))
        (x, y_t, y_0_hat, gx, t)  ──►  diffusion_model ──►  (eps_theta, sigma_theta)

    Training loss = KL(LSNM) + MSE(μ) + MSE(√g):
        loss1 = (y_0_hat - y).mse()             # mean prediction accuracy
        loss2 = (sqrt(gx) - sqrt(y_sigma)).mse() # variance prediction accuracy
        kl_loss = (e - eps_theta).mse()
                  + sigma_tilde/sigma_theta
                  - log(sigma_tilde/sigma_theta)

    Inference: full LSNM reverse chain via p_sample_loop, n_samples times.

    All three sub-models are registered as nn.Module attributes so
    model.parameters() covers all weights — the project's Train() function
    builds one Adam optimizer over the entire adapter.
    """

    def __init__(self, config: NsDiffConfig = NsDiffConfig()) -> None:
        super().__init__(config)
        self.config: NsDiffConfig = config

        ctx_len = config.seq_len - config.pred_len
        label_len = ctx_len // 2
        self._label_len = label_len
        self._ctx_len = ctx_len

        # --- Three sub-models (all nn.Module attributes → included in parameters()) ---
        diff_configs = config.diffusion_model_configs()
        self.diffusion_model = NsDiffModel(diff_configs, self.device).to(self.device)

        mu_configs = config.mu_backbone_configs()
        self.mu_model = MuBackbone(mu_configs).float().to(self.device)

        self.sigma_model = (
            SigmaEstimation(
                seq_len=ctx_len,
                pred_len=config.pred_len,
                enc_in=config.feature_size,
                hidden_size=config.sigma_hidden_size,
                kernel_size=min(config.rolling_length, ctx_len - 1),
            )
            .float()
            .to(self.device)
        )

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        x, y = self._adapt_batch(batch)  # [B, ctx, K], [B, pred, K]
        B = x.shape[0]

        y_sigma = wv_sigma_trailing(
            torch.cat([x, y], dim=1), self.config.rolling_length
        )
        y_sigma = y_sigma[:, -self.config.pred_len :, :] + EPS

        # Decoder input for mu_backbone: label window + zero target
        dec_inp_label = x[:, -self._label_len :, :]
        dec_inp_pred = torch.zeros(
            B, self.config.pred_len, self.config.feature_size, device=self.device
        )
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        # Antithetic timestep sampling
        t_half = torch.randint(
            0, self.diffusion_model.num_timesteps, (B // 2 + 1,), device=self.device
        )
        t = torch.cat([t_half, self.diffusion_model.num_timesteps - 1 - t_half], dim=0)[
            :B
        ]

        # Get conditional predictions (x_mark=None → positional encoding only)
        y_0_hat, _ = self.mu_model(x, None, dec_inp, None)  # [B, pred, K]
        gx = self.sigma_model(x) + EPS  # [B, pred, K]

        loss1 = (y_0_hat - y).square().mean()
        loss2 = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()

        # LSNM forward diffusion
        y_T_mean = y_0_hat
        e = torch.randn_like(y)
        forward_noise = cal_forward_noise(
            self.diffusion_model.betas_tilde,
            self.diffusion_model.betas_bar,
            gx,
            y_sigma,
            t,
        )
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(
            self.diffusion_model.alphas,
            self.diffusion_model.alphas_cumprod,
            self.diffusion_model.alphas_cumprod_sum,
            self.diffusion_model.alphas_cumprod_prev,
            self.diffusion_model.alphas_cumprod_sum_prev,
            self.diffusion_model.betas_tilde_m_1,
            self.diffusion_model.betas_bar_m_1,
            gx,
            y_sigma,
            t,
        )
        y_t = q_sample(
            y,
            y_T_mean,
            self.diffusion_model.alphas_bar_sqrt,
            self.diffusion_model.one_minus_alphas_bar_sqrt,
            t,
            noise=noise,
        )

        eps_theta, sigma_theta = self.diffusion_model(x, None, y_t, y_0_hat, gx, t)
        sigma_theta = sigma_theta + EPS

        kl_loss = (
            (e - eps_theta).square().mean()
            + (sigma_tilde / sigma_theta).mean()
            - torch.log(sigma_tilde / sigma_theta).mean()
        )
        return kl_loss + loss1 + loss2

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        x, _ = self._adapt_batch(batch)
        B = x.shape[0]

        dec_inp_label = x[:, -self._label_len :, :]
        dec_inp_pred = torch.zeros(
            B, self.config.pred_len, self.config.feature_size, device=self.device
        )
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        y_0_hat, _ = self.mu_model(x, None, dec_inp, None)
        gx = self.sigma_model(x) + EPS
        y_T_mean = y_0_hat

        all_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            y_p_seq = p_sample_loop(
                model=self.diffusion_model,
                x=x,
                x_mark=None,
                y_0_hat=y_0_hat,
                gx=gx,
                y_T_mean=y_T_mean,
                n_steps=self.diffusion_model.num_timesteps,
                alphas=self.diffusion_model.alphas,
                one_minus_alphas_bar_sqrt=self.diffusion_model.one_minus_alphas_bar_sqrt,
                alphas_cumprod=self.diffusion_model.alphas_cumprod,
                alphas_cumprod_sum=self.diffusion_model.alphas_cumprod_sum,
                alpha_bar_prev=self.diffusion_model.alphas_cumprod_prev,
                alphas_cumprod_sum_prev=self.diffusion_model.alphas_cumprod_sum_prev,
                betas_tiled=self.diffusion_model.betas_tilde,
                betas_bar=self.diffusion_model.betas_bar,
                betas_tiled_m_1=self.diffusion_model.betas_tilde_m_1,
                betas_bar_m_1=self.diffusion_model.betas_bar_m_1,
            )
            pred = y_p_seq[-1]  # [B, pred, K] — final denoised prediction

            if self.config.first_differences:
                base = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                    :, -(self.config.pred_len + 1) : -(self.config.pred_len)
                ]
                pred = base + pred.cumsum(dim=1)

            # Build full sequence: observed context + predicted horizon
            ctx = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                :, : self._ctx_len
            ]
            full = torch.cat([ctx, pred], dim=1)  # [B, L, K]
            all_samples.append(full)

        stacked = torch.stack(all_samples, dim=1)  # [B, n_samples, L, K]

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
    # Batch conversion
    # ------------------------------------------------------------------

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, target) in [B, ctx/pred_len, K]."""
        pred_len = self.config.pred_len
        x = batch.observed_data.to(device=self.device, dtype=torch.float32)

        if self.config.first_differences:
            x = self._first_differences(x)

        return x[:, :-pred_len], x[:, -pred_len:]

    @staticmethod
    def _first_differences(data: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros_like(data)
        diffs[:, 1:] = data[:, 1:] - data[:, :-1]
        return diffs
