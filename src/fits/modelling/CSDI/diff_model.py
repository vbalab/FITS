import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(
        in_channels, out_channels, kernel_size
    )  # already initialized by Kaiming (He)
    nn.init.kaiming_normal_(
        layer.weight
    )  # changes the distribution shape: uniform -> normal
    # -> a bit of overcomplication
    return layer


class SinusoidalDiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()

        if projection_dim is None:
            projection_dim = embedding_dim

        assert embedding_dim % 2 == 0

        # non-trainable, sinusoidal position encoding
        self.register_buffer(
            "embedding",
            # "embedding / 2", because we'll get [sin, ..., sin, cos, ..., cos] for each diffusion timestep, that later will be used in projection.
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )

        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        # diffusion_step: torch.LongTensor[B] — advanced indexing: one index per batch element.
        x = self.embedding[diffusion_step]  # [B, Te]

        # "timestep MLP": SiLU(W_2 * SiLU(W_1 * e))
        x = self.projection1(x)  # [B, Te]
        x = F.silu(x)  # [B, Te]
        x = self.projection2(x)  # [B, Te]
        x = F.silu(x)  # [B, Te]
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # [T, 1]
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # [1, dim]
        table = steps * frequencies  # [T, dim]
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1
        )  # [T, Te = dim * 2]

        return table


# TODO: try; also think if we need "timestep MLP" here
class LearnableDiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, emb_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_steps, emb_dim)

    def forward(self, diffusion_step):
        x = self.embedding(diffusion_step)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        cond_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
    ):
        super().__init__()

        self.cond_dim = cond_dim

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(cond_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape

        if L == 1:
            return y

        y = (
            y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        )  # [B * K, C, L]

        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        y = (
            y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        )  # [B, C, K * L]

        return y

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape

        if K == 1:
            return y

        y = (
            y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        )  # [B * L, C, K]

        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        y = (
            y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        )  # [B, C, K * L]

        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, C, K, L = x.shape
        base_shape = x.shape

        x = x.reshape(B, C, K * L)  # [B, C, K * L]

        # [B, Te] -> [B, C] -> [B, C, 1]
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # [B, C, 1]
        y = x + diffusion_emb  # [B, C, K * L]; broadcasted `1` to `K * L`

        # 1. In CSDI feature-attention is applied on top of the output of the time-attention, not as they used separately (TODO: mb try separate & concatenated)
        # 2. Why this order (time -> feature)?
        #   Empirically stable for ts: temporal smoothing first, then fuse variables at each time.
        y = self.forward_time(y, base_shape)  # [B, C, K * L]
        y = self.forward_feature(y, base_shape)  # [B, C, K * L]
        y = self.mid_projection(
            y
        )  # [B, 2 * C, K * L] <- basically could do separate&concat instead

        _, cond_dim, _, _ = cond_info.shape
        assert cond_dim == self.cond_dim

        # S = timeemb + featureemb [+1]
        cond_info = cond_info.reshape(B, self.cond_dim, K * L)  # [B, S, K * L]
        cond_info = self.cond_projection(cond_info)  # [B, 2 * C, K * L]

        # 1. Why it’s placed after both attentions?
        #   This is analogous to how conditional diffusion models (e.g. DDPM for class-conditional image generation) inject conditioning features into intermediate layers.
        # 2. Why do we add conditional info additively instead of concatenating?
        #   Additive conditioning like this is a FiLM-style modulation (Feature-wise Linear Modulation).
        #   It says: “shift the hidden representation based on the conditioning signal.”
        y = y + cond_info  # [B, 2 * C, K * L]

        # “Gated Linear Unit” from WaveNet & PixelCNN:
        gate, filtr = torch.chunk(y, 2, dim=1)  # [B, C, K * L], [B, C, K * L]
        y = torch.sigmoid(gate) * torch.tanh(filtr)  # [B, C, K * L]
        y = self.output_projection(y)  # [B, 2 * C, K * L]

        residual, skip = torch.chunk(y, 2, dim=1)  # [B, C, K * L], [B, C, K * L]

        x = x.reshape(base_shape)  # [B, C, K, L]
        residual = residual.reshape(base_shape)  # [B, C, K, L]
        x = (x + residual) / math.sqrt(2.0)  # [B, C, K, L]; keeps variance stable

        skip = skip.reshape(base_shape)  # [B, C, K, L]

        return x, skip


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = SinusoidalDiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # Why we use `Conv1d` for projection instead of `Linear`?
        #    Conv1d broadcasts weights along all positions (here: all K × L grid points).
        #    We could use Linear actually (doing the same broadcasting), just with reshaped data:
        #        self.linear = nn.Linear(inputdim, C)
        #        ...
        #        x = x.reshape(B, inputdim, K * L).permute(0, 2, 1)    # [B, K * L, inputdim]
        #        x = self.linear(x)                                    # [B, K * L, C]
        #    so it's just convinient to use Conv1d, that's all
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.n_layers = config["layers"]
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    cond_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self,
        x,
        cond_info,
        diffusion_step,  # diffusion_step: torch.LongTensor[B]
    ):
        # B = batch
        # K = number of ts variables (a.k.a. features)
        # L = time
        # I = (noise) or (observed parts & missing parts noised) depending on self.is_unconditional - raw ts dimensionality - number of raw input channels per (feature, time)
        # C = internal latent dimensionality (hidden “channel” width of the model)
        # D = diffusion embedding dimensionality

        B, I, K, L = x.shape

        x = x.reshape(B, I, K * L)  # [B, I, K * L], I is 2nd for Conv1d
        x = self.input_projection(x)  # [B, C, K * L]
        x = F.relu(x)  # [B, C, K * L]
        x = x.reshape(B, self.channels, K, L)  # [B, C, K, L]

        diffusion_emb = self.diffusion_embedding(diffusion_step)  # [B, Te]

        skips = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skips.append(skip_connection)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(
            self.n_layers
        )  # [B, C, K, L]
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # [B, C, K * L]
        x = F.relu(x)  # [B, C, K * L]
        x = self.output_projection2(x)  # [B, 1, K * L]
        x = x.reshape(B, K, L)  # [B, K, L]

        return x
