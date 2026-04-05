"""
FALDA — Fourier Adaptive Lite Diffusion Architecture.

Reference:
    Wang, X., Dai, R., Liu, K., & Chu, X. (2025).
    Effective Probabilistic Time Series Forecasting with Fourier
    Adaptive Noise-Separated Diffusion.
    arXiv:2505.11306. https://arxiv.org/abs/2505.11306

No official source code has been released. This is a faithful
implementation of the architecture described in the paper.

Design:
  Fourier decomposition splits the forecast target Y into three
  components:
    Y_non   — top-K₁ amplitude frequencies  (non-stationary trend)
    Y_noise — bottom-K₂ amplitude frequencies (noise)
    Y_stat  = Y - Y_non - Y_noise             (stationary patterns)

  Three modules process them:
    f_w  (NSAdapter)  : X_non  → Ŷ_non   (MLP)
    g_φ  (StatBackbone): X_stat → Ŷ_stat  (Linear)
    θ    (DEMA)        : diffusion denoiser over R = Y - Ŷ_non - Ŷ_stat

  Final prediction: Ŷ = Ŷ_non + Ŷ_stat + R̂

  Training uses an alternating optimisation schedule (Eq. 8-11):
    Phase 1 (s < δ):       pretrain f_w and g_φ with L_non + L_point
    Phase 2 (s mod Δ ≠ 0): train DEMA with stop-gradient on R
    Phase 2 (s mod Δ = 0): fine-tune f_w/g_φ with stop-gradient on R̂

  Diffusion: linear schedule K=1000 steps (paper §5); DDIM at inference.
"""

import math

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Diffusion schedule (linear, K=1000 steps)
# ---------------------------------------------------------------------------


def _make_schedule(
    K: int, beta_start: float, beta_end: float
) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, K)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": alpha_bar.sqrt(),
        "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
    }


def _extract(buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    out = buf.to(t.device)[t]
    return out.view(t.shape[0], *([1] * (ndim - 1)))


# ---------------------------------------------------------------------------
# Fourier decomposition (Eq. 4)
# ---------------------------------------------------------------------------


def fourier_decompose(
    y: torch.Tensor,
    k_non: int,
    k_noise: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split y ∈ [B, L, N] into (y_non, y_stat, y_noise) using FFT.

    Y_non   = iFFT( top-k_non  by amplitude )
    Y_noise = iFFT( bottom-k_noise by amplitude )
    Y_stat  = Y - Y_non - Y_noise
    """
    B, L, N = y.shape
    # FFT along the time axis
    Y_f = torch.fft.rfft(y, dim=1)  # [B, L//2+1, N] complex
    amp = Y_f.abs()  # [B, L//2+1, N]
    F_len = amp.shape[1]

    # Clamp to available frequencies
    k_non = min(k_non, F_len)
    k_noise = min(k_noise, F_len - k_non)

    # Indices sorted by amplitude (per sample, per variate)
    sorted_idx = amp.argsort(dim=1, descending=True)  # [B, F_len, N]

    top_idx = sorted_idx[:, :k_non, :]  # [B, k_non, N]
    bot_idx = sorted_idx[:, -k_noise:, :]  # [B, k_noise, N]

    def _mask_and_irfft(indices: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(amp, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        Y_masked = torch.where(mask, Y_f, torch.zeros_like(Y_f))
        return torch.fft.irfft(Y_masked, n=L, dim=1)  # [B, L, N]

    y_non = _mask_and_irfft(top_idx)
    y_noise = _mask_and_irfft(bot_idx)
    y_stat = y - y_non - y_noise
    return y_non, y_stat, y_noise


# ---------------------------------------------------------------------------
# NS-Adapter  f_w  (Appendix E.3: "implemented as an MLP")
# ---------------------------------------------------------------------------


class NSAdapter(nn.Module):
    """
    Multi-layer perceptron mapping X_non [B, ctx, N] → Ŷ_non [B, pred, N].

    Applied independently per variate (channel-independent) then projected
    to the forecast horizon.
    """

    def __init__(
        self, ctx_len: int, pred_len: int, n_features: int, hidden: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x_non: torch.Tensor) -> torch.Tensor:
        """x_non: [B, ctx, N] → [B, pred, N]."""
        # apply per-variate: transpose to [B, N, ctx], linear, back
        return self.net(x_non.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Stationary backbone  g_φ  (paper uses iTransformer as default)
# ---------------------------------------------------------------------------


class StatBackbone(nn.Module):
    """
    Lightweight stationary-component backbone.

    The paper plugs in iTransformer here (hence "plug-and-play"), but for
    self-containedness we use a channel-independent Linear layer, which
    achieves competitive performance on stationary data (cf. DLinear).
    Users can swap in any ForecastingModel as g_φ externally.
    """

    def __init__(self, ctx_len: int, pred_len: int, n_features: int) -> None:
        super().__init__()
        # Channel-independent linear: separate weight per variate
        self.linear = nn.Linear(ctx_len, pred_len)

    def forward(self, x_stat: torch.Tensor) -> torch.Tensor:
        """x_stat: [B, ctx, N] → [B, pred, N]."""
        return self.linear(x_stat.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Diffusion step sinusoidal embedding
# ---------------------------------------------------------------------------


class StepEmbedding(nn.Module):
    def __init__(self, d_emb: int) -> None:
        super().__init__()
        assert d_emb % 2 == 0
        self.d_emb = d_emb
        self.proj = nn.Sequential(nn.Linear(d_emb, d_emb), nn.SiLU())

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """k: [B] → [B, d_emb]."""
        half = self.d_emb // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=k.device).float() / (half - 1)
        )
        args = k.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# DEMA denoiser  (Decomposition MLP with Adaptive Layer Norm)
# Equations 12-16 from the paper
# ---------------------------------------------------------------------------


class DEMALayer(nn.Module):
    """
    One DEMA layer (Eqs. 13-16).

    Step 1: Decompose h into seasonal + trend via moving average (Eq. 13)
    Step 2: Compute AdaLN parameters from the conditioning signal (Eq. 14)
    Step 3: Normalise each component with AdaLN scale/shift (Eq. 15)
    Step 4: Gated linear combination → residual update (Eq. 16)
    """

    def __init__(self, d: int, ma_kernel: int, d_cond: int) -> None:
        super().__init__()
        # Moving-average trend extraction
        pad = (ma_kernel - 1) // 2
        self.ma = nn.AvgPool1d(ma_kernel, stride=1, padding=pad)

        # Layer norms for season and trend
        self.ln_season = nn.LayerNorm(d)
        self.ln_trend = nn.LayerNorm(d)

        # AdaLN parameter projection (Eq. 14): 6 * d parameters (γ, β, o for each of 2 components)
        self.adaLN_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 6 * d))

        # Output mix
        self.out_proj = nn.Linear(d, d)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        h: [B, H_d, d]   — sequence representations
        e: [B, d_cond]   — conditioning (step embedding + x_noise projection)
        Returns [B, H_d, d]
        """
        B, H_d, d = h.shape

        # Eq. 13: decompose into trend and seasonal
        trend = self.ma(h.permute(0, 2, 1)).permute(0, 2, 1)  # [B, H_d, d]
        season = h - trend  # [B, H_d, d]

        # Eq. 14: AdaLN parameters from conditioning
        params = self.adaLN_proj(e)  # [B, 6d]
        params = params.view(B, 1, 6, d)  # [B, 1, 6, d]
        gamma_s, beta_s, o_s = params[:, :, 0], params[:, :, 1], params[:, :, 2]
        gamma_t, beta_t, o_t = params[:, :, 3], params[:, :, 4], params[:, :, 5]

        # Eq. 15: adaptive layer norm
        tau_s = (gamma_s + 1) * self.ln_season(season) + beta_s  # [B, H_d, d]
        tau_t = (gamma_t + 1) * self.ln_trend(trend) + beta_t  # [B, H_d, d]

        # Eq. 16: gated output update
        h = h + (o_s + o_t) * self.out_proj(tau_s + tau_t)
        return h


class DEMA(nn.Module):
    """
    Full DEMA denoiser (Eqs. 12-16).

    Directly reconstructs R̂⁰ from R^(k) conditioned on (k, X_noise).
    """

    def __init__(
        self,
        pred_len: int,
        n_features: int,
        d_model: int,
        n_layers: int,
        ma_kernel: int,
        d_noise_ctx: int,  # dimension of X_noise projection
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        d_emb = d_model

        # Eq. 12: input embedding  R^(k) → h^[0]
        self.input_proj = nn.Linear(n_features, d_model)

        # Diffusion step + context noise conditioning → e_k (Eq. 12)
        self.step_emb = StepEmbedding(d_emb)
        self.noise_ctx_proj = nn.Linear(
            n_features, d_emb
        )  # project x_noise per time step
        d_cond = d_emb  # combined conditioning dim

        # L DEMA layers
        self.layers = nn.ModuleList(
            [
                DEMALayer(d=d_model, ma_kernel=ma_kernel, d_cond=d_cond)
                for _ in range(n_layers)
            ]
        )

        # Output projection → R̂⁰  [B, H_d, N]
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(
        self,
        r_k: torch.Tensor,  # [B, pred, N] — noisy residual at step k
        k: torch.Tensor,  # [B] — diffusion step
        x_noise: torch.Tensor,  # [B, pred, N] — historical noise forecast
    ) -> torch.Tensor:
        """Returns R̂⁰ ∈ [B, pred, N]."""
        # Eq. 12: input embedding
        h = self.input_proj(r_k)  # [B, pred, d]

        # Eq. 12: conditioning signal e_k
        p_k = self.step_emb(k)  # [B, d_emb]
        c = self.noise_ctx_proj(x_noise).mean(dim=1)  # [B, d_emb] — avg over time
        e = p_k + c  # [B, d_cond]

        # L DEMA layers
        for layer in self.layers:
            h = layer(h, e)

        return self.out_proj(h)  # [B, pred, N]


# ---------------------------------------------------------------------------
# Main FALDA model
# ---------------------------------------------------------------------------


class FALDAModel(nn.Module):
    """
    FALDA: Fourier Adaptive Lite Diffusion Architecture.

    Holds all three sub-modules and the diffusion schedule.
    The adapter drives training (alternating schedule) and inference (DDIM).
    """

    def __init__(
        self,
        ctx_len: int,
        pred_len: int,
        n_features: int,
        k_non: int = 5,
        k_noise: int = 5,
        ns_hidden: int = 128,
        dema_d_model: int = 64,
        dema_n_layers: int = 3,
        dema_ma_kernel: int = 25,
        diff_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.k_non = k_non
        self.k_noise = k_noise
        self.diff_steps = diff_steps

        # Sub-modules
        self.ns_adapter = NSAdapter(ctx_len, pred_len, n_features, ns_hidden)
        self.stat_backbone = StatBackbone(ctx_len, pred_len, n_features)
        self.dema = DEMA(
            pred_len=pred_len,
            n_features=n_features,
            d_model=dema_d_model,
            n_layers=dema_n_layers,
            ma_kernel=dema_ma_kernel,
            d_noise_ctx=n_features,
        )

        # Diffusion schedule buffers
        sched = _make_schedule(diff_steps, beta_start, beta_end)
        for name, val in sched.items():
            self.register_buffer(name, val)

    # ------------------------------------------------------------------
    # DDPM forward  q(R^(k) | R)
    # ------------------------------------------------------------------

    def q_sample(
        self, r: torch.Tensor, k: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(r)
        sqrt_ab = _extract(self.sqrt_alpha_bar, k, r.ndim)
        sqrt_1mab = _extract(self.sqrt_one_minus_alpha_bar, k, r.ndim)
        return sqrt_ab * r + sqrt_1mab * noise

    # ------------------------------------------------------------------
    # DDIM inference  (accelerated sampling, ~50 steps)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        x_noise: torch.Tensor,  # [B, pred, N] — noise ctx
        n_sample_steps: int = 50,
    ) -> torch.Tensor:
        """Run DDIM reverse chain, returns R̂⁰ ∈ [B, pred, N]."""
        device = x_noise.device
        B = x_noise.shape[0]

        # Evenly-spaced DDIM timestep subsequence
        step_size = self.diff_steps // n_sample_steps
        timesteps = list(reversed(range(0, self.diff_steps, step_size)))

        r = torch.randn(B, self.pred_len, self.n_features, device=device)

        for i, t_val in enumerate(timesteps):
            k = torch.full((B,), t_val, device=device, dtype=torch.long)
            r0_pred = self.dema(r, k, x_noise)

            ab_t = _extract(self.sqrt_alpha_bar, k, r.ndim) ** 2
            ab_tm1 = (
                _extract(
                    self.sqrt_alpha_bar,
                    torch.full((B,), timesteps[i + 1], device=device, dtype=torch.long),
                    r.ndim,
                )
                ** 2
                if i + 1 < len(timesteps)
                else torch.ones_like(ab_t)
            )

            # DDIM update (η=0: deterministic)
            r = ab_tm1.sqrt() * r0_pred + (1.0 - ab_tm1).sqrt() * (
                r - ab_t.sqrt() * r0_pred
            ) / (1.0 - ab_t).sqrt().clamp(min=1e-8)

        return r
