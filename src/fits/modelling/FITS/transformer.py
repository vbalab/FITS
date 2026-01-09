import torch
import torch.nn.functional as F
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        hidden_dim: int = 64,  # C
    ):
        super().__init__()

        self.ln = nn.LayerNorm(hidden_dim)
        ...

    def forward(self, x, timestep, mask=None, label_emb=None): ...


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 8,
        hidden_dim: int = 64,  # C
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(...) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        ...
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        ...

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        ...
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderBlock(...) for _ in range(n_layer)])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        ...
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        ...

    def forward(
        self,
    ):
        ...

        return out
