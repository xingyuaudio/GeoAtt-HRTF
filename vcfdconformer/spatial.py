"""Variable-size spatial set encoding and geometry-aware cross-attention."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FourierDirectionEncoding(nn.Module):
    """Fourier features for absolute head-centred unit directions."""

    def __init__(self, bands: int = 4) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands))

    @property
    def output_dim(self) -> int:
        return 3 + 2 * 3 * len(self.frequencies)

    def forward(self, directions: Tensor) -> Tensor:
        angles = directions[..., None] * self.frequencies * torch.pi
        return torch.cat(
            (directions, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1
        )


class GeometryAwareSpatialAttention(nn.Module):
    """Cross-attention from target directions to a padded set of context HRTFs.

    Attention is independently applied at every frequency. Geometry contributes
    an additive, head-specific bias computed from delta direction, dot product,
    and geodesic angle.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, direction_bands: int = 4,
                 use_geometry_bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_geometry_bias = use_geometry_bias
        direction = FourierDirectionEncoding(direction_bands)
        self.direction = direction
        self.context_position = nn.Sequential(
            nn.Linear(direction.output_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.query_position = nn.Sequential(
            nn.Linear(direction.output_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.spectrum_value = nn.Sequential(
            nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(5, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, n_heads)
        ) if use_geometry_bias else None

    def forward(
        self,
        context_hrtf: Tensor,
        context_dirs: Tensor,
        context_mask: Tensor,
        query_dirs: Tensor,
    ) -> Tensor:
        """Return spatial tokens with shape `(B, Q, F, C)`."""
        batch, n_context, _, n_freq = context_hrtf.shape
        n_query = query_dirs.size(1)

        spectra = context_hrtf.permute(0, 1, 3, 2)  # B, N, F, 2
        binaural_features = torch.cat(
            (spectra[..., :1], spectra[..., 1:2], spectra[..., :1] - spectra[..., 1:2]),
            dim=-1,
        )
        context_tokens = self.spectrum_value(binaural_features)
        context_tokens = context_tokens + self.context_position(self.direction(context_dirs))[:, :, None]
        query_tokens = self.query_position(self.direction(query_dirs))

        keys_and_values = context_tokens.permute(0, 2, 1, 3).reshape(
            batch * n_freq, n_context, self.d_model
        )
        queries = query_tokens[:, :, None].expand(-1, -1, n_freq, -1)
        queries = queries.permute(0, 2, 1, 3).reshape(batch * n_freq, n_query, self.d_model)

        if self.use_geometry_bias:
            dot = torch.einsum("bqc,bnc->bqn", query_dirs, context_dirs).clamp(-1.0, 1.0)
            delta = query_dirs[:, :, None] - context_dirs[:, None, :]
            relative_geometry = torch.cat(
                (delta, dot[..., None], torch.acos(dot)[..., None]), dim=-1
            )
            bias = self.geometry_bias(relative_geometry).permute(0, 3, 1, 2)
            attention_mask = bias[:, None].expand(-1, n_freq, -1, -1, -1)
            attention_mask = attention_mask.masked_fill(
                ~context_mask[:, None, None, None], float("-inf")
            ).reshape(batch * n_freq * self.n_heads, n_query, n_context)
            attended, _ = self.attention(
                queries, keys_and_values, keys_and_values,
                attn_mask=attention_mask, need_weights=False,
            )
        else:
            key_padding_mask = (~context_mask)[:, None].expand(-1, n_freq, -1)
            key_padding_mask = key_padding_mask.reshape(batch * n_freq, n_context)
            attended, _ = self.attention(
                queries, keys_and_values, keys_and_values,
                key_padding_mask=key_padding_mask, need_weights=False,
            )
        return attended.reshape(batch, n_freq, n_query, self.d_model).permute(0, 2, 1, 3)
