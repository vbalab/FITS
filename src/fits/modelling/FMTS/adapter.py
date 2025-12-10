from dataclasses import dataclass

import torch

from fits.dataframes.dataset import ForecastingData
from fits.modelling.FMTS.interpretable_diffusion.FMTS import FM_TS
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class FMTSConfig(ModelConfig):
    """Configuration for the :class:`FMTSAdapter`."""

    seq_len: int = 48
    feature_size: int = 36
    n_layer_enc: int = 2
    n_layer_dec: int = 4
    d_model: int = 64
    n_heads: int = 4
    mlp_hidden_times: int = 3
    attn_pd: float = 0.0
    resid_pd: float = 0.0
    kernel_size: int | None = None
    padding_size: int | None = None

    def fmts_kwargs(self) -> dict:
        return {
            "seq_length": self.seq_len,
            "feature_size": self.feature_size,
            "n_layer_enc": self.n_layer_enc,
            "n_layer_dec": self.n_layer_dec,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "mlp_hidden_times": self.mlp_hidden_times,
            "attn_pd": self.attn_pd,
            "resid_pd": self.resid_pd,
            "kernel_size": self.kernel_size,
            "padding_size": self.padding_size,
        }


class FMTSAdapter(ForecastingModel):
    """Adapter that exposes FM-TS through the unified :class:`ForecastingModel` API."""

    def __init__(self, config: FMTSConfig = FMTSConfig()):
        super().__init__(config)
        self.config: FMTSConfig = config
        self.fmts = FM_TS(**config.fmts_kwargs()).to(self.device)

    def forward(self, batch: ForecastingData):
        diffusion_batch = self._adapt_batch(batch)
        return self.fmts(diffusion_batch)

    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        self.eval()

        with torch.no_grad():
            diffusion_batch = self._adapt_batch(batch)
            batch_size = diffusion_batch.size(0)
            partial_mask = batch.forecast_mask.to(self.device).bool()

            samples = []
            for _ in range(n_samples):
                generated = self.fmts.fast_sample_infill(
                    shape=(batch_size, self.config.seq_len, self.config.feature_size),
                    target=diffusion_batch,
                    partial_mask=partial_mask,
                )
                generated = generated.to(
                    diffusion_batch.device, dtype=diffusion_batch.dtype
                )

                if batch.observed_mask is not None:
                    padding_mask = batch.observed_mask.bool().all(dim=-1)
                    generated[padding_mask] = diffusion_batch[padding_mask]

                samples.append(generated)

            stacked = torch.stack(samples, dim=1)  # (B, nsample, L, K)
            forecast_mask = (
                (~batch.forecast_mask.bool()).permute(0, 2, 1).to(self.device)
            )

            observed_data = batch.observed_data.to(self.device).permute(0, 2, 1)
            observed_mask = batch.observed_mask.to(self.device).permute(0, 2, 1)

            time_points = batch.time_points.to(self.device)
            if time_points.dim() == 3:
                time_points = time_points[..., 0]

            return ForecastedData(
                forecasted_data=stacked.permute(0, 1, 3, 2),
                forecast_mask=forecast_mask,
                observed_data=observed_data,
                observed_mask=observed_mask,
                time_points=time_points,
            )

    def _adapt_batch(self, batch: ForecastingData) -> torch.Tensor:
        assert isinstance(batch, ForecastingData)
        return batch.observed_data.to(dtype=torch.float32, device=self.device)
