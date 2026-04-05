from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from fits.dataframes.dataset import ForecastingData
from fits.modelling.FALDA.model import FALDAModel, fourier_decompose
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class FALDAConfig(ModelConfig):
    """Configuration for FALDA (Fourier Adaptive Lite Diffusion Architecture).

    FALDA (Wang et al., arXiv:2505.11306) decomposes each time series into
    non-stationary, stationary, and noise components via FFT amplitude
    thresholding. An MLP handles the trend, a linear backbone handles the
    stationary part, and a DEMA diffusion denoiser handles the residual.

    Implemented from the paper — no official source code exists.
    Paper: https://arxiv.org/abs/2505.11306

    Key paper hyperparameters:
      diff_steps=1000 (K), ddim_steps=50 for inference
      beta schedule: linear 1e-4 → 0.02
      training: pretrain δ epochs, then alternate every Δ epochs
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7
    seq_len: int = 96  # total window (context + pred)
    pred_len: int = 24

    # --- Fourier decomposition -----------------------------------------------
    k_non: int = 5  # top-K₁ amplitude frequencies → non-stationary trend
    k_noise: int = 5  # bottom-K₂ amplitude frequencies → noise component

    # --- NS-Adapter (MLP) ----------------------------------------------------
    ns_hidden: int = 128

    # --- DEMA denoiser -------------------------------------------------------
    dema_d_model: int = 64
    dema_n_layers: int = 3
    dema_ma_kernel: int = 25  # moving-average kernel size inside DEMA

    # --- Diffusion schedule --------------------------------------------------
    diff_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 50  # inference acceleration via DDIM

    # --- Alternating optimisation schedule -----------------------------------
    pretrain_epochs: int = 5  # δ — epochs before DEMA training starts
    alt_interval: int = 5  # Δ — how often to alternate fine-tune vs DEMA

    # --- Preprocessing -------------------------------------------------------
    first_differences: bool = False


class FALDAAdapter(ForecastingModel):
    """Adapter wrapping FALDA into :class:`ForecastingModel`.

    Training uses the alternating schedule from Eqs. 8-11.
    The adapter tracks a global step counter to implement the schedule
    without relying on external epoch information:
      - Steps 0 .. pretrain_steps-1 : pretrain NS-Adapter + StatBackbone
      - Steps pretrain_steps+       : alternate between
          DEMA-only (step mod alt_interval ≠ 0) and
          backbone-fine-tune (step mod alt_interval = 0)

    Total loss = L_non + L_point + L_alter (Eq. 11):
      L_non    = L1(Ŷ_non, Y_non)
      L_point  = L1(Ŷ_non + Ŷ_stat, Y_non + Y_stat)
      L_alter  = MSE(R, R̂) or MSE(R̂, R) depending on phase (stop-gradient)

    Inference:
      1. Decompose X → X_non, X_stat, X_noise.
      2. Predict Ŷ_non via NS-Adapter, Ŷ_stat via StatBackbone.
      3. Sample residual R̂ via DDIM conditioned on X_noise.
      4. Return Ŷ = Ŷ_non + Ŷ_stat + R̂, repeated n_samples times.
         (Monte-Carlo: multiple DDIM draws give different samples.)
    """

    def __init__(self, config: FALDAConfig = FALDAConfig()) -> None:
        super().__init__(config)
        self.config: FALDAConfig = config
        ctx_len = config.seq_len - config.pred_len
        self.model = FALDAModel(
            ctx_len=ctx_len,
            pred_len=config.pred_len,
            n_features=config.feature_size,
            k_non=config.k_non,
            k_noise=config.k_noise,
            ns_hidden=config.ns_hidden,
            dema_d_model=config.dema_d_model,
            dema_n_layers=config.dema_n_layers,
            dema_ma_kernel=config.dema_ma_kernel,
            diff_steps=config.diff_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(self.device)
        # Step counter for alternating schedule
        self._train_step: int = 0
        # Approximate steps-per-epoch using a sentinel; will be corrected on
        # first forward call if needed. We use epoch counting via step counter
        # directly with pretrain_epochs * steps_per_epoch ≈ pretrain_steps.
        # Since we don't have batch-count info here we use a heuristic:
        # pretrain "epochs" interpreted as pretrain_epochs * 100 steps.
        self._pretrain_steps = config.pretrain_epochs * 100

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        x, y = self._adapt_batch(batch)  # [B, ctx, N], [B, pred, N]

        # Fourier decompose X (context) and Y (target)
        x_non, x_stat, x_noise = fourier_decompose(
            x, self.config.k_non, self.config.k_noise
        )
        y_non, y_stat, y_noise = fourier_decompose(
            y, self.config.k_non, self.config.k_noise
        )

        # Point predictions
        y_non_hat = self.model.ns_adapter(x_non)  # [B, pred, N]
        y_stat_hat = self.model.stat_backbone(x_stat)  # [B, pred, N]

        # Residual = Y - Ŷ_non - Ŷ_stat
        r_true = y - y_non_hat.detach() - y_stat_hat.detach()  # clean residual

        # L_non: non-stationary guidance loss (Eq. 8)
        l_non = F.l1_loss(y_non_hat, y_non)

        # L_point: overall point estimation loss (Eq. 8)
        l_point = F.l1_loss(y_non_hat + y_stat_hat, y_non + y_stat)

        # Determine training phase (alternating schedule, Eq. 10)
        step = self._train_step
        self._train_step += 1

        if step < self._pretrain_steps:
            # Pretrain phase: only L_non + L_point, no diffusion
            return l_non + l_point

        # Diffusion phase — sample a random timestep
        B = y.shape[0]
        k = torch.randint(0, self.config.diff_steps, (B,), device=self.device)
        noise = torch.randn_like(r_true)
        r_k = self.model.q_sample(r_true, k, noise)

        alt_phase = (step % self.config.alt_interval) == 0  # fine-tune backbone

        # X_noise projected to prediction length for DEMA conditioning
        x_noise_pred = F.interpolate(
            x_noise.permute(0, 2, 1),
            size=self.config.pred_len,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        if not alt_phase:
            # DEMA training: stop-gradient on R (Eq. 9, λ_s=1, η_s=0)
            r_hat = self.model.dema(r_k, k, x_noise_pred)
            l_alter = F.mse_loss(r_hat, r_true.detach())
        else:
            # Backbone fine-tune: stop-gradient on R̂ (Eq. 9, λ_s=0, η_s=1)
            with torch.no_grad():
                r_hat = self.model.dema(r_k, k, x_noise_pred)
            # Fine-tune by making R (= Y - Ŷ) close to the denoiser's output
            r_true_fresh = y - y_non_hat - y_stat_hat  # with fresh gradients
            l_alter = F.mse_loss(r_true_fresh, r_hat.detach())

        return l_non + l_point + l_alter

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        x, _ = self._adapt_batch(batch)

        x_non, x_stat, x_noise = fourier_decompose(
            x, self.config.k_non, self.config.k_noise
        )

        y_non_hat = self.model.ns_adapter(x_non)
        y_stat_hat = self.model.stat_backbone(x_stat)

        # X_noise projected to prediction length
        x_noise_pred = F.interpolate(
            x_noise.permute(0, 2, 1),
            size=self.config.pred_len,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        all_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            # DDIM sample residual (stochastic — different per call)
            r_hat = self.model.ddim_sample(
                x_noise_pred, n_sample_steps=self.config.ddim_steps
            )
            pred = y_non_hat + y_stat_hat + r_hat  # Ŷ = Ŷ_non + Ŷ_stat + R̂

            if self.config.first_differences:
                base = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                    :, -(self.config.pred_len + 1) : -(self.config.pred_len)
                ]
                pred = base + pred.cumsum(dim=1)

            ctx_len = self.config.seq_len - self.config.pred_len
            context = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                :, :ctx_len
            ]
            full = torch.cat([context, pred], dim=1)
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
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
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
