from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import reduce

from fits.dataframes.dataset import ForecastingData
from fits.modelling.FITS.transformer import Transformer
from fits.modelling.framework import ForecastedData, ForecastingModel, ModelConfig


@dataclass
class FITSConfig(ModelConfig):
    seq_len: int = 48
    feature_size: int = 36
    n_layer_enc: int = 4
    n_layer_dec: int = 4
    d_model: int = 64
    n_heads: int = 4
    mlp_hidden_times: int = 4
    attn_pd: float = 0.0
    resid_pd: float = 0.0
    kernel_size: int | None = None
    padding_size: int | None = None
    lognormal_hucfg_t_sampling: bool = True  # otherwise, uniform
    hucfg_num_steps: int = 500
    first_differences: bool = True
    conditional: bool = (
        True  # use observed history values for conditioning during training/sampling
    )

    def fits_kwargs(self) -> dict[str, int | float | None]:
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


class FITSModel(ForecastingModel):
    def __init__(self, config: FITSConfig = FITSConfig()):
        super().__init__(config)
        self.config: FITSConfig = config
        self.model = Transformer(
            n_feat=config.feature_size,
            n_channel=config.seq_len,
            n_layer_enc=config.n_layer_enc,
            n_layer_dec=config.n_layer_dec,
            n_heads=config.n_heads,
            attn_pdrop=config.attn_pd,
            resid_pdrop=config.resid_pd,
            mlp_hidden_times=config.mlp_hidden_times,
            max_len=config.seq_len,
            n_embd=config.d_model,
            conv_params=[config.kernel_size, config.padding_size],
        ).to(self.device)

        self.time_scalar = 1000  ## scale 0-1 to 0-1000 for time embedding

    def forward(self, batch: ForecastingData):
        diffusion_batch, partial_mask, _, forecast_mask = self._adapt_batch(batch)
        return self._train_loss(diffusion_batch, forecast_mask, partial_mask)

    @torch.no_grad()
    def evaluate(self, batch: ForecastingData, n_samples: int) -> ForecastedData:
        self.eval()

        diffusion_batch, partial_mask, base_levels, _ = self._adapt_batch(batch)
        batch_size = diffusion_batch.size(0)

        samples = []
        for _ in range(n_samples):
            generated = self.fast_sample_infill(
                shape=(batch_size, self.config.seq_len, self.config.feature_size),
                target=diffusion_batch,
                partial_mask=partial_mask,
            )
            generated = generated.to(
                diffusion_batch.device,
                dtype=diffusion_batch.dtype,
            )
            samples.append(generated)

        stacked = torch.stack(samples, dim=1)
        if self.config.first_differences:
            stacked = self._restore_levels(stacked, base_levels)

        return ForecastedData(
            forecasted_data=stacked,
            forecast_mask=batch.forecast_mask.to(self.device, dtype=torch.float32),
            observed_data=batch.observed_data.to(self.device, dtype=torch.float32),
            observed_mask=batch.observed_mask.to(self.device, dtype=torch.float32),
            time_points=batch.time_points[..., 0].to(self.device, dtype=torch.float32),
        )

    def _output(self, x: torch.Tensor, t: torch.Tensor):
        return self.model(x, t, padding_masks=None)

    def _train_loss(
        self,
        x_start: torch.Tensor,
        loss_mask: torch.Tensor,
        partial_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z0 = torch.randn_like(x_start)
        if self.config.conditional and partial_mask is not None:
            z0 = z0.clone()
            z0[partial_mask] = x_start[partial_mask]
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1, device=z0.device)
        if self.config.lognormal_hucfg_t_sampling:
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1, device=z0.device))

        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0

        model_out = self._output(z_t, t.squeeze() * self.time_scalar)
        train_loss = F.mse_loss(model_out, target, reduction="none")

        loss_mask = loss_mask.to(device=train_loss.device, dtype=train_loss.dtype)
        train_loss = train_loss * loss_mask

        loss_sum = reduce(train_loss, "b ... -> b", "sum")
        mask_sum = reduce(loss_mask, "b ... -> b", "sum").clamp_min(1.0)
        return (loss_sum / mask_sum).mean()

    def fast_sample_infill(
        self,
        shape: tuple[int, int, int],
        target: torch.Tensor,
        partial_mask: torch.Tensor | None = None,
        hucfg_Kscale: float = 0.03,
    ) -> torch.Tensor:
        z0 = torch.randn(shape, device=self.device)
        if self.config.conditional and partial_mask is not None:
            z0 = z0.clone()
            z0[partial_mask] = target[partial_mask]
        z1 = zt = z0

        for step in range(self.config.hucfg_num_steps):
            t = step / self.config.hucfg_num_steps
            t = t**hucfg_Kscale

            z0 = torch.randn(shape, device=self.device)
            if self.config.conditional and partial_mask is not None:
                z0 = z0.clone()
                z0[partial_mask] = target[partial_mask]

            target_t = target * t + z0 * (1 - t)
            zt = z1 * t + z0 * (1 - t)
            if partial_mask is not None:
                zt = zt.clone()
                zt[partial_mask] = target_t[partial_mask]

            v = self._output(
                zt, torch.tensor([t * self.time_scalar], device=self.device)
            )

            z1 = zt.clone() + (1 - t) * v

            if self.config.first_differences:
                z1 = torch.clamp(z1, min=-2, max=2)
                # in first differences [-1, 1] -> [-2, 2]=[(-1) - (1), 1 - (-1)]
            else:
                z1 = torch.clamp(z1, min=-1, max=1)

        return z1

    def _adapt_batch(
        self, batch: ForecastingData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_data = batch.observed_data.to(dtype=torch.float32, device=self.device)
        context_mask = (batch.observed_mask * (1 - batch.forecast_mask)).to(self.device)
        partial_mask = context_mask.bool()
        # 0 - to be generated
        # 1 - known at generation
        forecast_mask = batch.forecast_mask.to(device=self.device, dtype=torch.float32)
        base_levels = observed_data[:, :1]

        if not self.config.first_differences:
            return observed_data, partial_mask, base_levels, forecast_mask

        diff_data = self._first_differences(observed_data)
        diff_partial_mask = self._first_difference_mask(partial_mask)

        return diff_data, diff_partial_mask, base_levels, forecast_mask

    @staticmethod
    def _first_differences(data: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros_like(data)
        diffs[:, 1:] = data[:, 1:] - data[:, :-1]
        return diffs

    @staticmethod
    def _first_difference_mask(context_mask: torch.Tensor) -> torch.Tensor:
        diff_mask = torch.zeros_like(context_mask)
        diff_mask[:, 1:] = context_mask[:, 1:] & context_mask[:, :-1]
        diff_mask[:, 0] = context_mask[:, 0]
        return diff_mask

    @staticmethod
    def _restore_levels(
        differences: torch.Tensor, base_levels: torch.Tensor
    ) -> torch.Tensor:
        base = base_levels.unsqueeze(1)
        return base + differences.cumsum(dim=2)
