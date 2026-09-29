"""Probabilistic multi-horizon influent forecaster.

Architecture: a dilated causal temporal convolutional encoder over the 24 h
of inlet history, followed by

* a **quantile head** producing the 10/50/90 % quantiles of every observable
  inlet channel for each of the next eight 15-minute steps, and
* an auxiliary **regime head** classifying the current weather condition
  (dry / rain / storm).

The quantile spread of the flow channel is the uncertainty signal that gates
the anticipatory term of the controller, and the regime posterior is a
directly interpretable storm alarm for the operator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import CHANNELS, HORIZON, LOOKBACK, QUANTILES, REGIMES


class CausalBlock(nn.Module):
    """Dilated causal convolution with a residual connection."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = F.gelu(self.conv1(h))
        h = F.pad(h, (self.pad, 0))
        h = self.drop(F.gelu(self.conv2(h)))
        return self.norm(x + h)


class InfluentForecaster(nn.Module):
    """TCN encoder + quantile head + weather-regime head."""

    def __init__(self, n_channels: int = len(CHANNELS), width: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                 n_quantiles: int = len(QUANTILES), horizon: int = HORIZON,
                 n_regimes: int = len(REGIMES), dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.horizon = horizon
        self.n_quantiles = n_quantiles

        self.stem = nn.Conv1d(n_channels, width, 1)
        self.blocks = nn.ModuleList(
            [CausalBlock(width, d, dropout=dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, horizon * n_channels * n_quantiles))
        self.regime_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_regimes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, C) -> context vector (B, width) at the last step."""
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor]:
        ctx = self.encode(x)
        q = self.head(ctx).view(-1, self.horizon, self.n_channels,
                                self.n_quantiles)
        # enforce monotone quantiles: q10 <= q50 <= q90
        base = q[..., :1]
        widths = F.softplus(q[..., 1:])
        q = torch.cat([base, base + torch.cumsum(widths, dim=-1)], dim=-1)
        return q, self.regime_head(ctx), ctx


def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles=QUANTILES) -> torch.Tensor:
    """Pinball loss averaged over horizon, channels and quantile levels."""
    taus = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()
