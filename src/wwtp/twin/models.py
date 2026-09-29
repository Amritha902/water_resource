"""Neural digital twin of the BSM1 loop.

A recurrent residual model

    h_t      = GRU([s_t, u_t, d_t], h_{t-1})
    s_{t+1}  = s_t + g(h_t)          (controlled variables)
    NH_t     = q(h_t)                (effluent ammonium)

The recurrent state stands in for everything the controller cannot measure
(biomass inventories, sludge blanket, reactors 1-4), which is exactly the
unknown interconnection term ``G_i(s_k)`` of the reference paper.  The model
is differentiable in ``u``, so the controller can obtain the sensitivity of a
predicted trajectory to a control move without any hand-derived plant model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DigitalTwin(nn.Module):
    def __init__(self, n_state: int = 2, n_control: int = 2,
                 n_disturbance: int = 3, hidden: int = 96):
        super().__init__()
        self.n_state = n_state
        self.n_control = n_control
        self.n_disturbance = n_disturbance
        self.hidden = hidden

        n_in = n_state + n_control + n_disturbance
        self.embed = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU())
        self.cell = nn.GRUCell(hidden, hidden)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_state))
        self.nh_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def init_hidden(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def step(self, s: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
             h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One coarse prediction step. All tensors are normalised."""
        h = self.cell(self.embed(torch.cat([s, u, d], dim=-1)), h)
        return s + self.delta_head(h), self.nh_head(h), h

    def forward(self, s: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
                h: torch.Tensor | None = None):
        """Teacher-forced pass over a sequence; ``s``: (B, T, n_state)."""
        b, t, _ = s.shape
        h = self.init_hidden(b, s.device) if h is None else h
        s_out, nh_out = [], []
        for k in range(t):
            s_next, nh, h = self.step(s[:, k], u[:, k], d[:, k], h)
            s_out.append(s_next)
            nh_out.append(nh)
        return torch.stack(s_out, 1), torch.stack(nh_out, 1), h

    def rollout(self, s0: torch.Tensor, u_seq: torch.Tensor,
                d_seq: torch.Tensor, h0: torch.Tensor):
        """Free-running prediction used for control lookahead.

        ``u_seq``/``d_seq``: (B, H, n) -- returns the predicted state and
        ammonium trajectories, differentiable with respect to ``u_seq``.
        """
        s, h = s0, h0
        states, nhs = [], []
        for k in range(u_seq.shape[1]):
            s, nh, h = self.step(s, u_seq[:, k], d_seq[:, k], h)
            states.append(s)
            nhs.append(nh)
        return torch.stack(states, 1), torch.stack(nhs, 1), h
