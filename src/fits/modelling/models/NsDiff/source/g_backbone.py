# ruff: noqa
# Source: https://github.com/wwy155/NsDiff
# File: src/layer/g_backbone.py
# Reproduced verbatim except one import line converted to relative import
# so the file works inside a Python package (no algorithmic changes).
# Requires: torch-timeseries==0.1.10
import torch
import torch.nn as nn
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .SelfAttention_Family import DSAttention, AttentionLayer
from .embedding import DataEmbedding
import torch.nn.functional as F
from .sigma import wv_sigma, wv_sigma_trailing   # changed: was `from src.utils.sigma import ...`


class SigmaEstimation(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, seq_len, pred_len, enc_in, hidden_size=512, kernel_size=24):
        super(SigmaEstimation, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.hidden_size = hidden_size


        # Define 2-layer MLP for predicting future sigmas
        self.mlp = nn.Sequential(
            nn.Linear(seq_len -  kernel_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Output size should match enc_in
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len)  # Output size should match enc_in
        )

        # Moving average kernel (can adjust the window size if needed)
        self.kernel_size = kernel_size  # You can modify this to adjust the window size
        self.padding = self.kernel_size // 2  # To ensure the output length matches the input length



    def forward(self, x_enc):
        # x_enc: B, T, N
        # return sigmas: B, O, N
        B, T, N = x_enc.shape

        sigma = wv_sigma_trailing(x_enc, self.kernel_size, discard_rep=True)
        # 2. Use MLP to predict future sigma values
        sigma = sigma[:, -(T - self.kernel_size):, :] + 10e-8
        pred_sigma = self.mlp(sigma.permute(0, 2, 1))  # (B, T, N) -> (B, N, T)

        # 3. Extract the last `pred_len` time steps for prediction
        pred_sigma = F.softplus(pred_sigma).permute(0, 2, 1)  # (B, O, N) where O = pred_len
        return pred_sigma[:, -self.pred_len:, :]
