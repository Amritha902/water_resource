"""Effluent-ammonium early-warning network.

A causal TCN encoder over 6 h of plant signals with two heads:

* **level head** -- 10/50/90 % quantiles of effluent ammonium at each
  horizon (12, 30, 60, 90, 120 min), monotone by construction;
* **hazard head** -- a discrete-time survival model.  For each horizon
  interval it emits the conditional probability ``lambda_j`` that the first
  exceedance happens in that interval, and

      P(violation within h_j) = 1 - prod_{l <= j} (1 - lambda_l).

  The warning probability can therefore never *decrease* with the horizon
  ("likely within 30 min but unlikely within 2 h" is impossible by
  construction), and the head reads naturally as a time-to-violation
  distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..forecast.models import CausalBlock
from .dataset import CHANNELS, HORIZONS, QUANTILES

#: ammonium scale used inside the level head [g N/m3]
NH_SCALE = 5.0


class EarlyWarningNet(nn.Module):
    def __init__(self, n_channels: int = len(CHANNELS), width: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 n_horizons: int = len(HORIZONS),
                 n_quantiles: int = len(QUANTILES), dropout: float = 0.1):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.stem = nn.Conv1d(n_channels, width, 1)
        self.blocks = nn.ModuleList(
            [CausalBlock(width, d, dropout=dropout) for d in dilations])
        self.level_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, n_horizons * n_quantiles))
        self.hazard_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_horizons))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor):
        """``x``: (B, T, C) -> (levels (B,H,Q) [g N/m3], P(event by h) (B,H))."""
        z = self.encode(x)
        q = self.level_head(z).view(-1, self.n_horizons, self.n_quantiles)
        q = torch.cat([q[..., :1],
                       q[..., :1] + torch.cumsum(F.softplus(q[..., 1:]), -1)], -1)
        levels = q * NH_SCALE
        # log-survival accumulates, so P(event by h) is monotone in h
        log_surv = torch.cumsum(F.logsigmoid(-self.hazard_head(z)), dim=-1)
        p_event = 1.0 - torch.exp(log_surv)
        return levels, p_event.clamp(1e-6, 1 - 1e-6)


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    taus = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()
