"""Frequency-axis Conformer used after spatial set aggregation."""
from __future__ import annotations

from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FrequencyConformerBlock(nn.Module):
    """Macaron block whose self-attention and convolution run along frequency."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_in = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.conv_out = nn.Linear(d_model, d_model)
        self.conv_dropout = nn.Dropout(dropout)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + 0.5 * self.ffn1(x)
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.attention_dropout(attended)
        convolved = self.conv_in(self.conv_norm(x))
        value, gate = convolved.chunk(2, dim=-1)
        convolved = value * gate.sigmoid()
        convolved = self.depthwise(convolved.transpose(1, 2)).transpose(1, 2)
        x = x + self.conv_dropout(self.conv_out(convolved * convolved.sigmoid()))
        x = x + 0.5 * self.ffn2(x)
        return self.output_norm(x)


class FrequencyConformer(nn.Module):
    def __init__(self, n_freq: int, d_model: int, d_ff: int, n_heads: int, n_layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.position = nn.Parameter(Tensor(1, n_freq, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.position_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            FrequencyConformerBlock(d_model, d_ff, n_heads, kernel_size, dropout)
            for _ in range(n_layers)
        )

    def forward(self, frequency_tokens: Tensor) -> Tensor:
        x = self.position_dropout(frequency_tokens + self.position)
        for layer in self.layers:
            x = layer(x)
        return x
