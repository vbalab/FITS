"""
Multi-Resolution Diffusion Model (mr-Diff) — implemented from the paper.

Reference:
    Shen, L., Chen, W., & Kwok, J. T. (2024).
    Multi-Resolution Diffusion Models for Time Series Forecasting.
    ICLR 2024. https://openreview.net/forum?id=mmjnr0G8ZY

No official source code exists. This is a faithful implementation of the
architecture described in the paper and its slides.

Design:
  - S independent DDPM denoising stages (coarse → fine).
  - Stage s operates on Y_s, the s-th level seasonal-trend trend of Y.
  - Each stage has its own DenoisingNet (conv encoder + diffusion embedding
    + conv decoder) and conditioning built from X_s (lookback at same level).
  - Forward diffusion is standard DDPM with a linear β schedule.
  - Training loss = sum of per-stage MSE on Y_s^0 prediction.
  - Inference: start from Gaussian noise at stage S, denoise K→0, feed
    the result as coarser condition into stage S-1, and so on.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# ---------------------------------------------------------------------------
# Diffusion schedule helpers
# ---------------------------------------------------------------------------


def _linear_beta_schedule(K: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear beta schedule as used in mr-Diff (β₁=1e-4, β_K=0.1)."""
    return torch.linspace(beta_start, beta_end, K)


def _precompute_schedule(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


def _extract(coeff: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather coefficient at timestep t and reshape for broadcasting."""
    out = coeff.to(t.device)[t]
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


# ---------------------------------------------------------------------------
# Multi-resolution decomposition
# ---------------------------------------------------------------------------


def _avg_pool_trend(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Smooth x ∈ [B, H, N] with symmetric-padding average pool of size kernel_size.
    Returns the same shape — this is X_s = AvgPool(Pad(X_{s-1}), τ_s).
    """
    B, H, N = x.shape
    pad = (kernel_size - 1) // 2
    # Pad along the time (H) dimension
    x_pad = F.pad(x.permute(0, 2, 1), (pad, pad), mode="replicate")  # [B, N, H+2p]
    trend = F.avg_pool1d(
        x_pad, kernel_size=kernel_size, stride=1, padding=0
    )  # [B, N, H]
    return trend.permute(0, 2, 1)  # [B, H, N]


def build_trend_sequence(
    y: torch.Tensor,
    kernel_sizes: list[int],
) -> list[torch.Tensor]:
    """
    Build the multi-resolution trend sequence Y_0, Y_1, ..., Y_{S-1}.

    Y_0 = y (finest, the original forecast target)
    Y_s = AvgPool(Pad(Y_{s-1}), τ_s)  for s = 1, ..., S-1

    Args:
        y:            [B, H, N] — forecast target (finest resolution)
        kernel_sizes: list of length S-1, kernel_sizes[s] = τ_{s+1}

    Returns:
        list of S tensors, each [B, H, N]
    """
    trends = [y]
    for ks in kernel_sizes:
        trends.append(_avg_pool_trend(trends[-1], ks))
    return trends  # [Y_0, Y_1, ..., Y_{S-1}]


# ---------------------------------------------------------------------------
# Diffusion step embedding (sinusoidal, as described in the paper)
# ---------------------------------------------------------------------------


class DiffusionStepEmbedding(nn.Module):
    """
    Sinusoidal embedding for the diffusion step k, followed by two FC+SiLU layers.
    Produces embedding p^k ∈ R^d_emb.
    """

    def __init__(self, d_emb: int) -> None:
        super().__init__()
        assert d_emb % 2 == 0
        self.d_emb = d_emb
        self.fc1 = nn.Linear(d_emb, d_emb)
        self.fc2 = nn.Linear(d_emb, d_emb)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """k: [B] long tensor of timestep indices → [B, d_emb]."""
        half = self.d_emb // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=k.device).float() / (half - 1)
        )
        args = k.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, d_emb]
        emb = F.silu(self.fc1(emb))
        emb = F.silu(self.fc2(emb))
        return emb


# ---------------------------------------------------------------------------
# Per-stage denoising network
# ---------------------------------------------------------------------------


class DenoisingNet(nn.Module):
    """
    Denoising network θ_s for stage s.

    Architecture (from paper §3.2 and Figure 2):
      - Input projection: Y_s^k ∈ R^{N×H} → z̄^k ∈ R^{d'×H}  (linear per-time)
      - Diffusion step embedding p^k ∈ R^{d_emb} added via broadcast
      - Conv encoder: z̄^k → z^k ∈ R^{d''×H}
      - Concatenate z^k with condition c_s ∈ R^{2d×H} → (2d+d'')×H
      - Conv decoder → Y_{θ_s}^0 ∈ R^{N×H}

    The condition c_s is built externally by the adapter (see _build_condition).
    """

    def __init__(
        self,
        n_features: int,  # N — number of variates
        pred_len: int,  # H — forecast horizon
        d_model: int,  # d  (paper uses same d for input projection and condition)
        d_emb: int,  # diffusion step embedding dim
        n_layers: int,  # number of conv layers in encoder/decoder
        kernel_size: int,  # conv kernel size along the time axis
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        d_prime = d_model  # d' = d in paper
        d_double_prime = d_model  # d'' = d in paper

        # Input projection: [B, H, N] → [B, H, d']  (applied per time step)
        self.input_proj = nn.Linear(n_features, d_prime)

        # Diffusion step embedding projection to add to z̄^k
        self.step_emb = DiffusionStepEmbedding(d_emb)
        self.emb_proj = nn.Linear(d_emb, d_prime)

        # Conv encoder: [B, d', H] → [B, d'', H]
        enc_layers: list[nn.Module] = []
        in_ch = d_prime
        for _i in range(n_layers):
            out_ch = d_double_prime
            enc_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.SiLU(),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Condition dim: 2d (z_history + Y_{s+1}), concat with d'' from encoder
        # Decoder: [B, 2d + d'', H] → [B, N, H]
        dec_in = 2 * d_model + d_double_prime
        dec_layers: list[nn.Module] = []
        in_ch = dec_in
        for layer_idx in range(n_layers):
            out_ch = d_model if layer_idx < n_layers - 1 else n_features
            dec_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            ]
            if layer_idx < n_layers - 1:
                dec_layers.append(nn.SiLU())
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)

    def forward(
        self,
        y_k: torch.Tensor,  # [B, H, N]  noisy forecast at step k
        k: torch.Tensor,  # [B]         diffusion step indices
        cond: torch.Tensor,  # [B, H, 2d]  conditioning signal
    ) -> torch.Tensor:
        """Returns Y_{θ_s}^0 ∈ [B, H, N] — predicted clean forecast."""
        B, H, N = y_k.shape

        # Input projection + step embedding  [B, H, d']
        z = self.input_proj(y_k)  # [B, H, d']
        p = self.emb_proj(self.step_emb(k))  # [B, d']
        z = z + p.unsqueeze(1)  # broadcast over H

        # Conv encoder  [B, d'', H]
        z = self.encoder(z.permute(0, 2, 1))  # [B, d'', H]

        # Concat with condition  [B, 2d + d'', H]
        cond_t = cond.permute(0, 2, 1)  # [B, 2d, H]
        z_cat = torch.cat([z, cond_t], dim=1)  # [B, 2d+d'', H]

        # Conv decoder  [B, N, H]
        out = self.decoder(z_cat)  # [B, N, H]
        return out.permute(0, 2, 1)  # [B, H, N]


# ---------------------------------------------------------------------------
# Main mr-Diff model
# ---------------------------------------------------------------------------


class MrDiff(nn.Module):
    """
    Multi-Resolution Diffusion Model for time series forecasting.

    Holds S independent DenoisingNet modules and the shared diffusion schedule.
    The adapter handles the decomposition, conditioning, training loop, and
    inference loop — keeping this module as a thin container that matches
    the paper's description.

    Args:
        n_features:   N — number of variates
        seq_len:      total window length (context + pred)
        pred_len:     H — forecast horizon length
        n_stages:     S — number of resolution stages (default 5)
        diff_steps:   K — diffusion steps per stage (default 100)
        beta_start:   β₁ (default 1e-4)
        beta_end:     β_K (default 0.1)
        kernel_sizes: list of S-1 ints — τ_s for average-pooling decomposition
                      (e.g. [3, 5, 11, 21] for S=5)
        d_model:      channel dimension inside denoising nets (default 64)
        d_emb:        diffusion step embedding dim (default 64)
        n_layers:     conv layers per encoder/decoder block (default 3)
        conv_kernel:  kernel size for conv layers (default 3)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        n_stages: int = 5,
        diff_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        kernel_sizes: list[int] | None = None,
        d_model: int = 64,
        d_emb: int = 64,
        n_layers: int = 3,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_stages = n_stages
        self.diff_steps = diff_steps

        # Kernel sizes for multi-resolution decomposition
        # Default: progressively wider kernels for coarser stages
        if kernel_sizes is None:
            # τ_s doubles roughly with each stage
            kernel_sizes = [2 ** (s + 1) + 1 for s in range(n_stages - 1)]
        assert len(kernel_sizes) == n_stages - 1
        self.kernel_sizes = kernel_sizes

        # Diffusion schedule (registered as buffers → moves with .to(device))
        betas = _linear_beta_schedule(diff_steps, beta_start, beta_end)
        sched = _precompute_schedule(betas)
        for name, val in sched.items():
            self.register_buffer(name, val)

        # One context projection per stage:  X_s [B, ctx, N] → [B, ctx, d]
        self.ctx_projs = nn.ModuleList(
            [nn.Linear(n_features, d_model) for _ in range(n_stages)]
        )

        # S denoising networks — one per stage
        self.denoise_nets = nn.ModuleList(
            [
                DenoisingNet(
                    n_features=n_features,
                    pred_len=pred_len,
                    d_model=d_model,
                    d_emb=d_emb,
                    n_layers=n_layers,
                    kernel_size=conv_kernel,
                )
                for _ in range(n_stages)
            ]
        )

    # ------------------------------------------------------------------
    # Diffusion forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        y0: torch.Tensor,  # [B, H, N]
        k: torch.Tensor,  # [B]
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample Y_s^k from q(Y_s^k | Y_s^0): standard DDPM forward."""
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_ab = _extract(self.sqrt_alpha_bar, k, y0.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alpha_bar, k, y0.shape)
        return sqrt_ab * y0 + sqrt_1mab * noise

    # ------------------------------------------------------------------
    # Per-stage conditioning
    # ------------------------------------------------------------------

    def build_condition(
        self,
        x_s: torch.Tensor,  # [B, ctx, N] — lookback at stage s
        y_coarser: torch.Tensor,  # [B, H, N]  — Y_{s+1}^0 (or estimate)
        stage: int,
        y_s0: torch.Tensor | None = None,  # [B, H, N] — Y_s^0, only during training
        mixup_prob: float = 0.5,
    ) -> torch.Tensor:
        """
        Build conditioning signal c_s ∈ R^{B×H×2d} (paper §3.2).

        Training: future-mixup — z_mix = m ⊙ z_history + (1-m) ⊙ Y_s^0
        Inference: z_mix = z_history  (no ground-truth future available)
        Then c_s = concat(z_mix, coarser_proj) along feature dim.
        """
        # z_history: project X_s from context dimension  [B, H, d]
        # We align ctx to pred_len by taking the last pred_len steps
        x_aligned = x_s[:, -self.pred_len :, :]  # [B, H, N]
        z_hist = self.ctx_projs[stage](x_aligned)  # [B, H, d]

        if y_s0 is not None and self.training:
            # Future mixup: random binary mask m ~ Bernoulli(mixup_prob)
            m = torch.bernoulli(torch.full_like(z_hist, mixup_prob))
            y_proj = self.ctx_projs[stage](y_s0)  # [B, H, d]
            z_mix = m * z_hist + (1 - m) * y_proj
        else:
            z_mix = z_hist

        # Project coarser trend  [B, H, d]
        y_coarser_proj = self.ctx_projs[stage](y_coarser)

        # Concat along feature dim → [B, H, 2d]
        cond = torch.cat([z_mix, y_coarser_proj], dim=-1)
        return cond

    # ------------------------------------------------------------------
    # Single-stage training loss  (paper Eq. 3 / Algorithm 1)
    # ------------------------------------------------------------------

    def stage_loss(
        self,
        y0: torch.Tensor,  # [B, H, N]  — Y_s^0
        cond: torch.Tensor,  # [B, H, 2d] — conditioning
        stage: int,
    ) -> torch.Tensor:
        """MSE loss for stage s: E||Y_s^0 - Y_{θ_s}(Y_s^k, k | c_s)||²."""
        B = y0.shape[0]
        k = torch.randint(0, self.diff_steps, (B,), device=y0.device)
        noise = torch.randn_like(y0)
        y_k = self.q_sample(y0, k, noise)
        y0_pred = self.denoise_nets[stage](y_k, k, cond)
        return F.mse_loss(y0_pred, y0)

    # ------------------------------------------------------------------
    # Single-stage inference  (Algorithm 2, reversed K→0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def stage_sample(
        self,
        cond: torch.Tensor,  # [B, H, 2d]
        stage: int,
    ) -> torch.Tensor:
        """Full DDPM reverse chain for one stage. Returns Ŷ_s^0 ∈ [B, H, N]."""
        B = cond.shape[0]
        shape = (B, self.pred_len, self.n_features)
        y = torch.randn(shape, device=cond.device)

        for k_val in reversed(range(self.diff_steps)):
            k = torch.full((B,), k_val, device=cond.device, dtype=torch.long)

            # Predict Y_s^0
            y0_pred = self.denoise_nets[stage](y, k, cond)

            # DDPM reverse step: Ŷ^{k-1} = posterior_mean + σ_k ε
            beta = _extract(self.betas, k, y.shape)
            alpha = _extract(self.alphas, k, y.shape)
            alpha_bar = _extract(self.alpha_bar, k, y.shape)
            alpha_bar_prev = _extract(self.alpha_bar_prev, k, y.shape)

            # Posterior mean coefficients (standard DDPM)
            coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
            coef2 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean = coef1 * y0_pred + coef2 * y

            if k_val > 0:
                posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                noise = torch.randn_like(y)
                y = mean + torch.sqrt(posterior_var.clamp(min=1e-20)) * noise
            else:
                y = mean

        return y
