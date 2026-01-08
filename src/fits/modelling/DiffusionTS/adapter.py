from dataclasses import dataclass
from typing import Literal

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig
from fits.modelling.DiffusionTS.interpretable_diffusion.gaussian_diffusion import (
    Diffusion_TS,
)


@dataclass
class DiffusionTSConfig(ModelConfig):
    """Typed configuration for the :class:`DiffusionTSAdapter`."""

    seq_len: int = 48
    feature_size: int = 36
    n_layer_enc: int = 4
    n_layer_dec: int = 4
    d_model: int = 96  # 4 X 24
    timesteps: int = 500
    sampling_timesteps: int = 500
    # fast_sampling = sampling_timesteps < timesteps; if True, leads to dramatic decrease in sampling quality (500&200 was worse than 200&200)
    loss_type: Literal["l1", "l2"] = "l1"
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    n_heads: int = 4
    mlp_hidden_times: int = 4
    eta: float = 0.0
    attn_pd: float = 0.0
    resid_pd: float = 0.0
    kernel_size: int | None = None
    padding_size: int | None = None
    use_fourier_loss: bool = True
    # train_loss + fourier_loss --- to encourage the model to reproduce similar spectral content (amplitudes/phases across frequencies), which can improve seasonality and smoothness
    reg_weight: float | None = None
    langevin_coef: float = 1e-2
    # langevin_coef = 0.0   -> Replace-only conditional sampling
    # langevin_coef > 0     -> Replace + Guided conditional sampling
    langevin_learning_rate: float = 5e-2

    def diffusion_kwargs(self) -> dict:
        return {
            "seq_length": self.seq_len,
            "feature_size": self.feature_size,
            "n_layer_enc": self.n_layer_enc,
            "n_layer_dec": self.n_layer_dec,
            "d_model": self.d_model,
            "timesteps": self.timesteps,
            "sampling_timesteps": self.sampling_timesteps,
            "loss_type": self.loss_type,
            "beta_schedule": self.beta_schedule,
            "n_heads": self.n_heads,
            "mlp_hidden_times": self.mlp_hidden_times,
            "eta": self.eta,
            "attn_pd": self.attn_pd,
            "resid_pd": self.resid_pd,
            "kernel_size": self.kernel_size,
            "padding_size": self.padding_size,
            "use_ff": self.use_fourier_loss,
            "reg_weight": self.reg_weight,
        }


class DiffusionTSAdapter(ForecastingModel):
    """Wrap the DiffusionTS model with the unified :class:`ForecastingModel` API."""

    def __init__(self, config: DiffusionTSConfig = DiffusionTSConfig()):
        super().__init__(config)
        self.config: DiffusionTSConfig = config
        self.diffusion = Diffusion_TS(**config.diffusion_kwargs()).to(self.device)

    def forward(self, batch: ForecastingData):
        diffusion_batch, padding_mask, _ = self._adapt_batch(batch)
        return self.diffusion(diffusion_batch, padding_masks=padding_mask)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        self.eval()

        diffusion_batch, padding_mask, partial_mask = self._adapt_batch(batch)
        batch_size = diffusion_batch.size(0)
        model_kwargs = {
            "coef": self.config.langevin_coef,
            "learning_rate": self.config.langevin_learning_rate,
        }

        samples = []
        for _ in range(n_samples):
            if self.diffusion.fast_sampling:
                generated = self.diffusion.fast_sample_infill(
                    (batch_size, self.config.seq_len, self.config.feature_size),
                    target=diffusion_batch,
                    sampling_timesteps=self.diffusion.sampling_timesteps,
                    partial_mask=partial_mask,
                    model_kwargs=model_kwargs,
                )
            else:
                generated = self.diffusion.sample_infill(
                    shape=(
                        batch_size,
                        self.config.seq_len,
                        self.config.feature_size,
                    ),
                    target=diffusion_batch,
                    partial_mask=partial_mask,
                    model_kwargs=model_kwargs,
                )

            generated = generated.to(
                diffusion_batch.device,
                dtype=diffusion_batch.dtype,
            )

            # generated[padding_mask] = diffusion_batch[padding_mask]

            samples.append(generated)

        stacked = torch.stack(samples, dim=1)

        return ForecastedData(
            forecasted_data=stacked,
            forecast_mask=batch.forecast_mask.to(self.device, dtype=torch.float32),
            observed_data=batch.observed_data.to(self.device, dtype=torch.float32),
            observed_mask=batch.observed_mask.to(self.device, dtype=torch.float32),
            time_points=batch.time_points[..., 0].to(self.device, dtype=torch.float32),
        )

    def _adapt_batch(
        self, batch: ForecastingData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_data = batch.observed_data.to(dtype=torch.float32, device=self.device)
        padding_mask = None
        # padding_mask = ~batch.observed_mask.bool().any(dim=-1)

        partial_mask = (
            (batch.observed_mask * (1 - batch.forecast_mask)).to(self.device).bool()
        )
        # 0 - to be generated
        # 1 - known at generation

        return observed_data, padding_mask, partial_mask
