import io
import sys
from dataclasses import dataclass

import torch
from torch import nn

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig

from .source.SSSDS4Imputer import SSSDS4Imputer
from .source.util import (
    calc_diffusion_hyperparams,
    sampling,
    training_loss,
)


@dataclass
class SSSDConfig(ModelConfig):
    """Configuration for SSSD (Structured State Space Diffusion).

    SSSD (Alcaraz & Strodthoff, 2022) replaces the Transformer backbone of
    CSDI with structured state space (S4) layers inside a DiffWave-style
    residual architecture, capturing long-range dependencies more efficiently.

    Source files used verbatim:
      - SSSDS4Imputer.py  — denoising network (S4 residual blocks)
      - S4Model.py        — S4 state space layer implementation
      - util.py           — diffusion schedule + training_loss + sampling
        (subset of src/utils/util.py; dataset-masking helpers omitted)

    Two ``from utils.util`` / ``from imputers.S4Model`` absolute imports in
    SSSDS4Imputer.py were converted to relative imports — zero algorithmic
    changes.

    Additional runtime packages required by S4Model.py:
      einops, opt_einsum, scipy, lightning  (see requirements.txt)
    """

    # --- Dataset dimensions --------------------------------------------------
    feature_size: int = 7
    seq_len: int = 96  # total sequence length (context + horizon)

    # --- SSSDS4 architecture -------------------------------------------------
    res_channels: int = 24
    skip_channels: int = 24
    num_res_layers: int = 8
    diffusion_step_embed_dim_in: int = 64
    diffusion_step_embed_dim_mid: int = 128
    diffusion_step_embed_dim_out: int = 128
    s4_d_state: int = 16
    s4_dropout: float = 0.0
    s4_bidirectional: bool = True
    s4_layernorm: bool = True

    # --- Diffusion schedule --------------------------------------------------
    diff_steps: int = 200
    beta_start: float = 0.0001
    beta_end: float = 0.5

    # --- Preprocessing -------------------------------------------------------
    first_differences: bool = False

    def imputer_kwargs(self) -> dict:  # type: ignore[return]
        return {
            "in_channels": self.feature_size,
            "res_channels": self.res_channels,
            "skip_channels": self.skip_channels,
            "out_channels": self.feature_size,
            "num_res_layers": self.num_res_layers,
            "diffusion_step_embed_dim_in": self.diffusion_step_embed_dim_in,
            "diffusion_step_embed_dim_mid": self.diffusion_step_embed_dim_mid,
            "diffusion_step_embed_dim_out": self.diffusion_step_embed_dim_out,
            "s4_lmax": self.seq_len,
            "s4_d_state": self.s4_d_state,
            "s4_dropout": self.s4_dropout,
            "s4_bidirectional": self.s4_bidirectional,
            "s4_layernorm": self.s4_layernorm,
        }


class SSSDAdapter(ForecastingModel):
    """Adapter wrapping SSSDS4Imputer into :class:`ForecastingModel`.

    Data flow:

    SSSD uses channel-first layout ``[B, K, L]`` and a mask convention
    where ``1 = observed/context`` and ``0 = to generate``.

    Training (uses ``training_loss()`` from util.py):
        x [B, K, L], cond_mask [B, K, L]
          → diffusion forward pass + noise prediction
          → MSE loss at forecast (mask=0) positions

    Inference (uses ``sampling()`` from util.py):
        Full DDPM reverse chain conditioning on observed context.
        Each call to ``sampling()`` returns one independent sample ``[B, K, L]``.
        Repeated ``n_samples`` times → ``[B, n_samples, L, K]``.
    """

    def __init__(self, config: SSSDConfig = SSSDConfig()) -> None:
        super().__init__(config)
        self.config: SSSDConfig = config

        # SSSD source — unchanged except relative imports
        self.model = SSSDS4Imputer(**config.imputer_kwargs()).to(self.device)

        # Pre-compute diffusion schedule and move to CUDA
        _hp = calc_diffusion_hyperparams(
            T=config.diff_steps,
            beta_0=config.beta_start,
            beta_T=config.beta_end,
        )
        self.diffusion_hyperparams: dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in _hp.items()
        }

    # ------------------------------------------------------------------
    # ForecastingModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: ForecastingData) -> torch.Tensor:
        x, cond_mask = self._adapt_batch(batch)  # [B, K, L]

        # loss_mask: True at forecast positions (where we compute the loss)
        loss_mask = ~cond_mask.bool()

        # training_loss() from util.py handles the full DDPM training step:
        # random t, forward diffusion, noise prediction, masked MSE
        loss = training_loss(
            net=self.model,
            loss_fn=nn.MSELoss(),
            X=[x, x, cond_mask, loss_mask],
            diffusion_hyperparams=self.diffusion_hyperparams,
            only_generate_missing=1,
        )
        return loss

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        x, cond_mask = self._adapt_batch(batch)  # [B, K, L]
        B, K, L = x.shape

        all_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            # sampling() prints a progress line per call; suppress it since
            # we cannot modify the verbatim source file.
            _stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                generated = sampling(
                    net=self.model,
                    size=(B, K, L),
                    diffusion_hyperparams=self.diffusion_hyperparams,
                    cond=x,
                    mask=cond_mask,
                    only_generate_missing=1,
                )  # [B, K, L]
            finally:
                sys.stdout = _stdout

            # Restore conditioning: replace generated context with ground truth
            generated = generated * (1 - cond_mask) + x * cond_mask

            # Back to [B, L, K]
            all_samples.append(generated.permute(0, 2, 1))

        stacked = torch.stack(all_samples, dim=1)  # [B, n_samples, L, K]

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

    def _adapt_batch(self, batch: ForecastingData) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, cond_mask) both in [B, K, L] (channel-first).

        ``cond_mask``:  1 = observed context  (known, condition on this)
                        0 = forecast horizon  (to be generated)
        """
        observed_data = batch.observed_data.to(device=self.device, dtype=torch.float32)
        observed_mask = batch.observed_mask.to(device=self.device, dtype=torch.float32)
        forecast_mask = batch.forecast_mask.to(device=self.device, dtype=torch.float32)

        if self.config.first_differences:
            observed_data = self._first_differences(observed_data)
            observed_mask = self._first_difference_mask(observed_mask)
            forecast_mask = self._first_difference_mask(forecast_mask)

        # SSSD expects [B, K, L]
        x = observed_data.permute(0, 2, 1)
        cond_mask = (observed_mask * (1.0 - forecast_mask)).permute(0, 2, 1)

        return x, cond_mask

    # ------------------------------------------------------------------
    # Helpers (mirror CSDI adapter pattern)
    # ------------------------------------------------------------------

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
            base_levels: [B, 1, K]
        """
        base = base_levels.unsqueeze(1)  # [B, 1, 1, K]
        return base + differences.cumsum(dim=2)
