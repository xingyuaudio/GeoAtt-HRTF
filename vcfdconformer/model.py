"""GeoAtt: direct-prediction variable-context FD-Conformer."""
from __future__ import annotations

from torch import Tensor, nn

from .frequency import FrequencyConformer
from .spatial import GeometryAwareSpatialAttention


class VariableContextFDConformer(nn.Module):
    """Predict binaural HRTF magnitudes at arbitrary query directions.

    The spatial module aggregates an arbitrary number of known measurements.
    The frequency module then captures global and local spectral dependencies.
    """

    def __init__(
        self,
        n_freq: int = 106,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        conv_kernel: int = 7,
        dropout: float = 0.1,
        direction_bands: int = 4,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.spatial = GeometryAwareSpatialAttention(d_model, n_heads, dropout, direction_bands)
        self.frequency = FrequencyConformer(
            n_freq, d_model, d_ff, n_heads, n_layers, conv_kernel, dropout
        )
        self.spectrum_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(
        self,
        context_hrtf: Tensor,
        context_dirs: Tensor,
        context_mask: Tensor,
        query_dirs: Tensor,
    ) -> Tensor:
        """Directly predict HRTF log magnitudes, shape `(B, Q, 2, F)`."""
        if context_hrtf.ndim != 4 or context_hrtf.size(2) != 2:
            raise ValueError("context_hrtf must have shape (B, N, 2, F)")
        if context_hrtf.size(-1) != self.n_freq:
            raise ValueError(f"expected {self.n_freq} frequency bins")
        spatial_tokens = self.spatial(context_hrtf, context_dirs, context_mask, query_dirs)
        batch, n_query, n_freq, channels = spatial_tokens.shape
        spectral_tokens = self.frequency(spatial_tokens.reshape(batch * n_query, n_freq, channels))
        output = self.spectrum_head(spectral_tokens)
        return output.reshape(batch, n_query, n_freq, 2).permute(0, 1, 3, 2).contiguous()
