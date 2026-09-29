"""PANDA-MAACC -- Predictive ANticipatory Disturbance-Aware multiagent
adaptive critic control.

PANDA keeps the whole control-theoretic skeleton of the reference paper --
decentralised agents, a Q-function per subsystem, an incremental adaptive
term added to an expert prior -- and adds three learned components that
together turn a purely *reactive* controller into an *anticipatory* one:

``u_{k,i} = b_{k,i} + g_{k,i} a_{k,i} + f_{k,i}``

``b``  expert prior (incremental PID), exactly as in the paper;
``a``  adaptive critic term, but with the agents' action and critic networks
       additionally conditioned on a forecast context vector ``c_k``;
``g``  an uncertainty gate in [0, 1] derived from the width of the forecast
       predictive interval -- when the forecaster is unsure the controller
       retreats towards the industrially validated prior;
``f``  an anticipatory feed-forward move computed by differentiating a neural
       digital twin through the *predicted* influent trajectory.

The feed-forward solve is **closed-loop aware and offset-free**, which took
two corrections to get right (both documented in ``docs/03_novelty.md``).

*Closed-loop aware*: the rollout carries a differentiable copy of the
incremental PID, so the twin predicts what the **closed loop** will do under
the forecast influent.  Cancelling the open-loop disturbance impact instead
grossly over-compensates -- the PID would have reacted long before the
horizon ends -- and measured worse than plain PID.

*Offset-free*: the objective compares **two** closed-loop rollouts over the
same horizon, one driven by the forecast influent and one with the influent
frozen at its current value.  Only their difference is penalised.  Any
standing bias of the twin appears in both and cancels, so ``f`` is exactly
zero whenever the forecast says nothing is about to change, and the
feed-forward can never fight the integral action of the prior.  Scoring the
absolute predicted error instead makes the solver spend the actuator on
correcting model bias, which again measured worse than plain PID.

An effluent-ammonium head on the twin turns the same pair of rollouts into a
predictive discharge-violation early warning -- the *predicted rise* in
effluent ammonium caused by the incoming load -- which enters the objective
as a one-sided risk penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..forecast.dataset import FORECAST_DT_SECONDS
from ..forecast.predictor import InfluentPredictor
from ..maacc.agent import MAACCController, default_agent_configs
from ..twin.dataset import COARSE_STRIDE, D_SCALE, NH_SCALE, S_SCALE, U_SCALE
from ..twin.models import DigitalTwin

#: length of the control context vector produced by the predictor
CONTEXT_DIM = 6


@dataclass
class PandaConfig:
    """Hyper-parameters of the anticipatory layer."""

    horizon: int = 8                      # coarse steps (8 x 6 min = 48 min)
    solver_steps: int = 6                 # gradient steps per feed-forward solve
    solver_lr: float = 0.03               # step size in normalised control units
    discount: float = 0.90                # horizon weighting
    effort: float = 0.5                   # rho, penalty on the feed-forward move
    ff_limit: np.ndarray = None           # max |f| in physical units
    weight: np.ndarray = None             # W in the feed-forward objective
    nh_margin: float = 0.2                # tolerated predicted rise in NH4-N [g N/m3]
    nh_weight: float = 0.6                # risk penalty on a predicted rise
    gate_sensitivity: float = 4.0         # kappa of the uncertainty gate
    ff_confidence: float = 3.0            # uncertainty discount of the feed-forward
    recompute_every: int = 8              # control periods between solves

    def __post_init__(self):
        if self.ff_limit is None:
            self.ff_limit = np.array([40.0, 15000.0])
        if self.weight is None:
            self.weight = np.array([1.0, 0.6])


class PANDAController(MAACCController):
    """MAACC plus the forecast-driven anticipatory layer."""

    name = "PANDA"

    def __init__(self, forecaster: str | Path, twin: str | Path,
                 configs=None, seed: int = 0, cfg: PandaConfig | None = None,
                 use_context: bool = True, use_gate: bool = True,
                 use_feedforward: bool = True, device: str = "cpu",
                 dt_seconds: float = 45.0, **kwargs):
        self.cfg = cfg or PandaConfig()
        self.use_context = use_context
        self.use_gate = use_gate
        self.use_feedforward = use_feedforward
        self.device = device
        self.dt_seconds = dt_seconds

        self.predictor = InfluentPredictor(forecaster, dt_seconds=dt_seconds,
                                           device=device)
        blob = torch.load(twin, map_location=device, weights_only=False)
        self.twin = DigitalTwin().to(device)
        self.twin.load_state_dict(blob["state_dict"])
        self.twin.eval()
        for p in self.twin.parameters():
            p.requires_grad_(False)

        super().__init__(configs=configs or default_agent_configs(), seed=seed,
                         extra_context=CONTEXT_DIM if use_context else 0,
                         **kwargs)

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        if not hasattr(self, "predictor"):       # called from the base __init__
            return
        self.predictor.reset()
        self.h_twin = self.twin.init_hidden(1, torch.device(self.device))
        self.ff = np.zeros(2)
        self._delta = torch.zeros(1, 2)
        self._last_u = np.array([84.0, 55338.0])
        self._ctx = np.zeros(CONTEXT_DIM)
        self.diagnostics.update({"ff": [], "gate": [], "p_wet": [],
                                 "nh_risk": [], "ctx": []})
        self._nh_risk = 0.0

    # -- MAACC hooks -------------------------------------------------------
    def _context(self, context: dict) -> np.ndarray | None:
        return self._ctx if self.use_context else None

    def _gate(self, context: dict) -> np.ndarray:
        if not self.use_gate or not self.predictor.ready:
            return np.ones(self.n_agents)
        unc = float(self.predictor.relative_uncertainty(None)[0])
        return np.full(self.n_agents, 1.0 / (1.0 + self.cfg.gate_sensitivity * unc))

    def _feedforward(self, context: dict) -> np.ndarray:
        return self.ff if self.use_feedforward else np.zeros(self.n_agents)

    # -- anticipatory layer ------------------------------------------------
    def _forecast_disturbance(self, inlet: np.ndarray) -> np.ndarray:
        """Forecast influent on the twin's coarse grid, shape (H, 3)."""
        h = self.cfg.horizon
        coarse_dt = COARSE_STRIDE * self.dt_seconds
        t_coarse = (np.arange(1, h + 1) * coarse_dt) / FORECAST_DT_SECONDS
        med = self.predictor.median                       # (H_f, 3), 15-min grid
        t_src = np.arange(1, med.shape[0] + 1, dtype=float)
        out = np.empty((h, med.shape[1]))
        for j in range(med.shape[1]):
            out[:, j] = np.interp(t_coarse, t_src, med[:, j],
                                  left=med[0, j], right=med[-1, j])
        return out

    def _closed_loop_rollout(self, s0, r_t, d_seq, delta, pid_state, dev):
        """Roll the twin forward with the expert prior in the loop.

        ``delta`` is the constant anticipatory offset added on top of the
        prior's own move.  Returns the predicted state and effluent-ammonium
        trajectories; gradients flow to ``delta``.
        """
        kp, ki, kd, u_lo, u_hi, s_scale = pid_state["gains"]
        u_pid, e1, e2 = pid_state["u"], pid_state["e1"], pid_state["e2"]
        state, hid = s0, self.h_twin
        states, nhs = [], []
        for k in range(d_seq.shape[1]):
            e = (r_t - state) * s_scale
            du = kp * (e - e1) + ki * e + kd * (e - 2.0 * e1 + e2)
            u_pid = torch.clamp(u_pid + du, u_lo, u_hi)
            e2, e1 = e1, e
            u_k = torch.clamp(u_pid + delta, 0.0, 1.0)
            state, nh, hid = self.twin.step(state, u_k, d_seq[:, k], hid)
            states.append(state)
            nhs.append(nh)
        return torch.stack(states, 1), torch.stack(nhs, 1)

    def _solve_feedforward(self, s: np.ndarray, r: np.ndarray,
                           inlet: np.ndarray) -> np.ndarray:
        """Residual anticipatory move, solved through the neural twin.

        Two closed-loop rollouts are compared -- forecast influent versus
        influent frozen at its current value -- and only their difference is
        penalised, so the solve is offset-free with respect to twin bias.
        """
        cfg = self.cfg
        h = cfg.horizon
        dev = torch.device(self.device)
        f32 = torch.float32

        d_hat = torch.tensor(self._forecast_disturbance(inlet) / D_SCALE,
                             dtype=f32, device=dev)[None]
        d_now = torch.tensor(inlet / D_SCALE, dtype=f32,
                             device=dev)[None, None].expand(1, h, 3).contiguous()
        s0 = torch.tensor(s / S_SCALE, dtype=f32, device=dev)[None]
        r_t = torch.tensor(r / S_SCALE, dtype=f32, device=dev)[None]

        u_scale = torch.tensor(U_SCALE, dtype=f32, device=dev)
        s_scale = torch.tensor(S_SCALE, dtype=f32, device=dev)
        limit = torch.tensor(cfg.ff_limit / U_SCALE, dtype=f32, device=dev)
        weight = torch.tensor(cfg.weight, dtype=f32, device=dev)

        pid = self.pid
        pid_state = {
            "gains": (torch.tensor(pid.kp, dtype=f32, device=dev) / u_scale,
                      torch.tensor(pid.ki * COARSE_STRIDE, dtype=f32,
                                   device=dev) / u_scale,
                      torch.tensor(pid.kd, dtype=f32, device=dev) / u_scale,
                      torch.tensor(pid.u_min / U_SCALE, dtype=f32, device=dev),
                      torch.tensor(pid.u_max / U_SCALE, dtype=f32, device=dev),
                      s_scale),
            "u": torch.tensor(pid.u / U_SCALE, dtype=f32, device=dev)[None],
            "e1": torch.tensor(pid.e1, dtype=f32, device=dev)[None],
            "e2": torch.tensor(pid.e2, dtype=f32, device=dev)[None],
        }
        zero = torch.zeros(1, 2, dtype=f32, device=dev)

        # reference rollout: no anticipatory move, influent frozen.  Twin bias
        # is common to both branches and therefore cancels below.
        with torch.no_grad():
            s_ref, nh_ref = self._closed_loop_rollout(s0, r_t, d_now, zero,
                                                      pid_state, dev)

        gamma = torch.tensor([cfg.discount ** k for k in range(h)],
                             dtype=f32, device=dev)[None, :, None]

        delta = self._delta.detach().clone().to(dev).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=cfg.solver_lr)
        loss_val, rise = 0.0, torch.zeros((), device=dev)
        for it in range(cfg.solver_steps + 1):
            d_clamped = torch.tanh(delta / limit) * limit
            s_pred, nh_pred = self._closed_loop_rollout(s0, r_t, d_hat,
                                                        d_clamped, pid_state, dev)
            residual = (s_pred - s_ref) * s_scale * weight
            track = (gamma * residual ** 2).sum()
            rise = (nh_pred - nh_ref) * NH_SCALE
            risk = (gamma * torch.relu(rise - cfg.nh_margin) ** 2).sum()
            effort = cfg.effort * float(h) * (d_clamped ** 2).sum()
            loss = track + cfg.nh_weight * risk + effort
            if it == cfg.solver_steps:
                loss_val = float(loss.detach())
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            self._nh_risk = float(rise.detach().max())
            self._ff_loss = loss_val
            self._delta = delta.detach().cpu()
            ff = (torch.tanh(delta.detach() / limit) * limit
                  * u_scale)[0].cpu().numpy()
        return ff

    @torch.no_grad()
    def _advance_twin(self, s: np.ndarray, u: np.ndarray,
                      inlet: np.ndarray) -> None:
        dev = torch.device(self.device)
        s_t = torch.tensor(s / S_SCALE, dtype=torch.float32, device=dev)[None]
        u_t = torch.tensor(u / U_SCALE, dtype=torch.float32, device=dev)[None]
        d_t = torch.tensor(inlet / D_SCALE, dtype=torch.float32, device=dev)[None]
        _, _, self.h_twin = self.twin.step(s_t, u_t, d_t, self.h_twin)

    # -- control law -------------------------------------------------------
    def act(self, k: int, s: np.ndarray, r: np.ndarray,
            context: dict) -> np.ndarray:
        inlet = np.asarray(context.get("inlet", np.array([18446.0, 31.6, 69.5])),
                           dtype=float)
        self.predictor.update(k, inlet)
        if self.use_context:
            self._ctx = self.predictor.context()

        # keep the twin's recurrent state synchronised with the real plant
        if k % COARSE_STRIDE == 0:
            self._advance_twin(s, self._last_u, inlet)

        if (self.use_feedforward and self.predictor.ready
                and k % self.cfg.recompute_every == 0):
            raw = self._solve_feedforward(s, r, inlet)
            conf = float(np.exp(-self.cfg.ff_confidence
                                * self.predictor.relative_uncertainty(None)[0]))
            self.ff = np.clip(conf * raw, -self.cfg.ff_limit, self.cfg.ff_limit)

        u = super().act(k, s, r, context)
        self._last_u = np.clip(u, [0.0, 0.0], [240.0, 92230.0])

        if k % 400 == 0:
            self.diagnostics["ff"].append(self.ff.tolist())
            self.diagnostics["p_wet"].append(float(self.predictor.regime[1:].sum()))
            self.diagnostics["nh_risk"].append(self._nh_risk)
            self.diagnostics["ctx"].append(self._ctx.tolist())
        return u
