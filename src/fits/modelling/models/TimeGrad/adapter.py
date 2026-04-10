from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.epsilon_theta import EpsilonTheta
from .source.gaussian_diffusion import GaussianDiffusion


@dataclass
class TimeGradConfig(ModelConfig):
    """Configuration for the TimeGrad autoregressive diffusion model.

    TimeGrad (Rasul et al., ICML 2021) encodes the past with an RNN and
    uses the hidden state to condition a DDPM denoising network
    (EpsilonTheta) that generates one multivariate time step at a time.

    Source files used verbatim (pure PyTorch, no GluonTS deps):
      - epsilon_theta.py   — EpsilonTheta denoising network
      - gaussian_diffusion.py — GaussianDiffusion DDPM process

    The RNN encoder and training/inference loop are implemented here in the
    adapter because the original time_grad_network.py has hard GluonTS
    dependencies that would pull in the entire GluonTS stack.
    """

    # --- Dataset dimensions -------------------------------------------------
    feature_size: int = 7
    seq_len: int = 96
    pred_len: int = 24  # last N steps of seq_len to generate

    # --- RNN encoder --------------------------------------------------------
    num_cells: int = 64
    num_rnn_layers: int = 2
    cell_type: Literal["LSTM", "GRU"] = "LSTM"
    dropout_rate: float = 0.1

    # --- Conditioning projection --------------------------------------------
    # RNN hidden (num_cells) is projected to conditioning_length before being
    # passed to EpsilonTheta as cond_length.
    conditioning_length: int = 64

    # --- Diffusion process --------------------------------------------------
    diff_steps: int = 100
    loss_type: Literal["l1", "l2", "huber"] = "l2"
    beta_end: float = 0.1
    beta_schedule: Literal["linear", "quad", "cosine"] = "linear"

    # --- EpsilonTheta backbone ----------------------------------------------
    residual_layers: int = 12
    residual_channels: int = 32
    dilation_cycle_length: int = 2

    # --- Preprocessing ------------------------------------------------------
    scaling: bool = True
    first_differences: bool = False

    def epsilon_theta_kwargs(self) -> dict:  # type: ignore[return]
        return {
            "target_dim": self.feature_size,
            "cond_length": self.conditioning_length,
            "residual_layers": self.residual_layers,
            "residual_channels": self.residual_channels,
            "dilation_cycle_length": self.dilation_cycle_length,
        }

    def diffusion_kwargs(self) -> dict:  # type: ignore[return]
        return {
            "input_size": self.feature_size,
            "diff_steps": self.diff_steps,
            "loss_type": self.loss_type,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
        }


class TimeGradAdapter(ForecastingModel):
    """Adapter wrapping TimeGrad into :class:`ForecastingModel`.

    Architecture:

    Training (teacher-forced):
        x [B, L, K]  →  RNN  →  hidden [B, L, H]
                                    ↓  proj
                              cond [B, L, C]  (shifted by 1)
        loss = GaussianDiffusion.log_prob(x[:, -horizon:], cond[:, -horizon:])

    Inference (autoregressive):
        context [B, ctx, K]  →  RNN  →  final state
        for t in range(horizon):
            cond_t = proj(last_hidden)  →  [B, 1, C]
            x_t   = GaussianDiffusion.sample(cond=cond_t)  →  [B, 1, K]
            last_hidden = RNN_step(x_t, state)
    """

    def __init__(self, config: TimeGradConfig = TimeGradConfig()) -> None:
        super().__init__(config)
        self.config: TimeGradConfig = config

        # RNN encoder
        rnn_cls: type[nn.LSTM] | type[nn.GRU] = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }[config.cell_type]
        self.rnn = rnn_cls(
            input_size=config.feature_size,
            hidden_size=config.num_cells,
            num_layers=config.num_rnn_layers,
            dropout=config.dropout_rate if config.num_rnn_layers > 1 else 0.0,
            batch_first=True,
        ).to(self.device)

        # Projection from RNN hidden dim → conditioning dim for EpsilonTheta
        self.cond_proj = nn.Linear(config.num_cells, config.conditioning_length).to(
            self.device
        )

        # TimeGrad source — unchanged
        denoise_fn = EpsilonTheta(**config.epsilon_theta_kwargs())
        self.diffusion = GaussianDiffusion(denoise_fn, **config.diffusion_kwargs()).to(
            self.device
        )

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        x, _, _, scale = self._adapt_batch(batch)
        # x: [B, L, K] in our scaled (+ optionally differenced) space

        # Teacher-forced RNN over the full window
        rnn_out, _ = self.rnn(x)  # [B, L, num_cells]
        cond = self.cond_proj(rnn_out)  # [B, L, conditioning_length]

        # Shift by 1: condition for position t = RNN output after seeing t-1
        zeros = torch.zeros_like(cond[:, :1])
        cond_shifted = torch.cat([zeros, cond[:, :-1]], dim=1)  # [B, L, C]

        # Compute diffusion loss only on the forecast horizon positions
        horizon = self.config.pred_len
        x_target = x[:, -horizon:].contiguous()  # [B, horizon, K]
        cond_target = cond_shifted[:, -horizon:].contiguous()  # [B, horizon, C]

        # GaussianDiffusion.log_prob expects [B, T, K] and [B, T, C]
        # scale is already applied; ensure diffusion.scale is None so it
        # does not double-divide inside log_prob.
        self.diffusion.scale = None
        return self.diffusion.log_prob(x_target, cond_target)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        x, obs_mask, fcst_mask, scale = self._adapt_batch(batch)
        B, L, K = x.shape
        horizon = self.config.pred_len
        context_len = L - horizon

        # Encode the context region with the RNN
        context = x[:, :context_len]  # [B, ctx, K]
        _, rnn_state = self.rnn(context)
        # rnn_state: (h_n, c_n) for LSTM  or  h_n for GRU
        # h_n shape: [num_layers, B, num_cells]

        self.diffusion.scale = None  # scaling handled by adapter

        all_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            current_state = rnn_state
            # Last hidden of the top RNN layer → [B, num_cells]
            last_hidden = self._top_hidden(current_state)

            generated: list[torch.Tensor] = []
            for _ in range(horizon):
                # Condition = projection of current hidden → [B, 1, C]
                cond_t = self.cond_proj(last_hidden).unsqueeze(1)

                # Sample one multivariate timestep via full DDPM reverse chain
                # GaussianDiffusion.sample uses cond.shape[:-1] + (K,) for shape
                sample_t = self.diffusion.sample(cond=cond_t)  # [B, 1, K]
                generated.append(sample_t)

                # Advance the RNN with the sampled value
                _, current_state = self.rnn(sample_t, current_state)
                last_hidden = self._top_hidden(current_state)

            # [B, horizon, K]
            horizon_samples = torch.cat(generated, dim=1)

            # Reconstruct full sequence: observed context + generated horizon
            full = torch.cat([context, horizon_samples], dim=1)  # [B, L, K]
            all_samples.append(full)

        stacked = torch.stack(all_samples, dim=1)  # [B, n_samples, L, K]

        # Undo mean-abs scaling
        if scale is not None:
            stacked = stacked * scale.unsqueeze(1)  # broadcast over n_samples

        # Undo first differences → back to level space
        if self.config.first_differences:
            base = batch.observed_data.to(device=self.device, dtype=torch.float32)[
                :, :1
            ]
            stacked = self._restore_levels(stacked, base)

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

    def _adapt_batch(
        self, batch: ForecastingData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (x, observed_mask, forecast_mask, scale).

        Applies first-differencing and mean-abs scaling in that order so
        both are reversible cleanly in :meth:`evaluate`.
        """
        x = batch.observed_data.to(device=self.device, dtype=torch.float32)
        obs_mask = batch.observed_mask.to(device=self.device, dtype=torch.float32)
        fcst_mask = batch.forecast_mask.to(device=self.device, dtype=torch.float32)

        # Compute context mask from original (un-differenced) masks so the
        # 0→1 transition in forecast_mask is preserved correctly.
        context_mask = obs_mask * (1.0 - fcst_mask)  # [B, L, K]

        if self.config.first_differences:
            x = self._first_differences(x)
            obs_mask = self._first_difference_mask(obs_mask)
            context_mask = self._first_difference_mask(context_mask)

        scale: torch.Tensor | None = None
        if self.config.scaling:
            sum_abs = (x.abs() * context_mask).sum(dim=1, keepdim=True)
            count = context_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            scale = (sum_abs / count).clamp(min=1e-8)  # [B, 1, K]
            x = x / scale

        return x, obs_mask, fcst_mask, scale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _top_hidden(
        self,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Extract the top-layer hidden vector from an RNN state.

        Returns shape [B, num_cells].
        """
        if self.config.cell_type == "LSTM":
            # state = (h_n, c_n), h_n: [num_layers, B, num_cells]
            assert isinstance(state, tuple)
            return state[0][-1]
        # GRU: state = h_n: [num_layers, B, num_cells]
        assert isinstance(state, torch.Tensor)
        return state[-1]

    @staticmethod
    def _first_differences(data: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros_like(data)
        diffs[:, 1:] = data[:, 1:] - data[:, :-1]
        return diffs

    @staticmethod
    def _first_difference_mask(mask: torch.Tensor) -> torch.Tensor:
        diff_mask = torch.zeros_like(mask)
        diff_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
        diff_mask[:, 0] = mask[:, 0]
        return diff_mask

    @staticmethod
    def _restore_levels(
        differences: torch.Tensor, base_levels: torch.Tensor
    ) -> torch.Tensor:
        """Cumulative sum back to level space.

        Args:
            differences: [B, n_samples, L, K]
            base_levels: [B, 1, K]  (first time step in level space)
        """
        base = base_levels.unsqueeze(1)  # [B, 1, 1, K]
        return base + differences.cumsum(dim=2)
