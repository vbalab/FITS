from dataclasses import dataclass
from typing import Literal

import torch

from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig
from fits.data.dataset import ForecastingData
from fits.modelling.DiffusionTS.interpretable_diffusion.gaussian_diffusion import (
    Diffusion_TS,
)


@dataclass
class DiffusionTSConfig(ModelConfig):
    """Typed configuration for the :class:`DiffusionTSAdapter`."""

    seq_len: int = 48
    horizon: int = 6
    feature_size: int = 36
    n_layer_enc: int = 3
    n_layer_dec: int = 5
    d_model: int = 64
    timesteps: int = 100
    sampling_timesteps: int = 100
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
    reg_weight: float | None = None
    langevin_coef: float = 1.0
    langevin_learning_rate: float = 0.1

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
        diffusion_batch, padding_mask = self._adapt_batch(batch)
        return self.diffusion(diffusion_batch, padding_masks=padding_mask)

    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        self.eval()

        with torch.no_grad():
            diffusion_batch, padding_mask = self._adapt_batch(batch)
            batch_size = diffusion_batch.size(0)
            partial_mask = batch.forecast_mask.to(self.device).bool()

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
                    diffusion_batch.device, dtype=diffusion_batch.dtype
                )

                if padding_mask is not None:
                    generated[padding_mask] = diffusion_batch[padding_mask]

                samples.append(generated)

            stacked = torch.stack(samples, dim=1)  # (B, nsample, L, K)
            forecast_mask = (
                (~batch.forecast_mask.bool()).permute(0, 2, 1).to(self.device)
            )

            observed_data = batch.observed_data.to(self.device).permute(0, 2, 1)
            observed_mask = batch.observed_mask.to(self.device).permute(0, 2, 1)

            return ForecastedData(
                forecasted_data=stacked.permute(0, 1, 3, 2),
                forecast_mask=forecast_mask,
                observed_data=observed_data,
                observed_mask=observed_mask,
                time_points=batch.time_points.to(self.device),
            )

    def _adapt_batch(
        self, batch: ForecastingData
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(batch, ForecastingData)

        observed_data = batch.observed_data.to(dtype=torch.float32, device=self.device)
        padding_mask = None
        if batch.observed_mask is not None:
            padding_mask = batch.observed_mask.bool().all(dim=-1)
        return observed_data, padding_mask
