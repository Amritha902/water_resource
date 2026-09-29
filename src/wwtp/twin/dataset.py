"""Excitation rollouts used to identify the neural digital twin.

A twin that is to be differentiated for control must have seen the plant
respond to *control moves*, not only to influent disturbances.  Data taken
under feedback is not enough: with the PID in the loop, ``KLa_5`` rises
exactly when the oxygen demand rises, so the apparent correlation between
aeration and dissolved oxygen is close to zero -- or inverted.  A twin fitted
on such data reports a near-zero aeration gain and the anticipatory solver
then moves the actuator the wrong way (see ``docs/03_novelty.md``).

The collector therefore mixes two excitation modes:

``prbs``  open-loop amplitude-modulated pseudo-random binary sequences.  The
          control is held constant for a random dwell time and resampled
          across the whole admissible range, independently of the plant
          state, so the causal gain is identifiable.
``pid``   the expert PID plus a bounded random-walk dither, which covers the
          narrow region around the set point where the plant actually runs.

Records are stored on a coarse grid (``COARSE_STRIDE`` control periods) that
matches the prediction step of the twin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..baselines.pid import IncrementalPID, PAPER_KP, PAPER_KI, PAPER_KD
from ..bsm1 import asm1
from ..bsm1.influent import random_scenario
from ..bsm1.plant import BSM1Plant, KLA5_BOUNDS, QA_BOUNDS, PAPER_SETPOINTS

#: control periods per twin prediction step (8 x 45 s = 6 min)
COARSE_STRIDE = 8
#: physical scales used to normalise the twin's inputs and outputs
S_SCALE = np.array([2.0, 3.0])            # [S_O,5, S_NO,2]
U_SCALE = np.array([240.0, 92230.0])      # [KLa_5, Q_a]
D_SCALE = np.array([20000.0, 40.0, 90.0])  # [Q_in, S_NH_in, S_S_in]
NH_SCALE = 5.0                            # effluent ammonium


@dataclass
class TwinCorpus:
    """Sequences of (state, control, disturbance) and the next states."""

    s: np.ndarray        # (N, T, 2) normalised controlled variables
    u: np.ndarray        # (N, T, 2) normalised manipulated variables
    d: np.ndarray        # (N, T, 3) normalised inlet observations
    s_next: np.ndarray   # (N, T, 2) normalised state one coarse step later
    nh: np.ndarray       # (N, T, 1) normalised effluent ammonium

    def __len__(self) -> int:
        return self.s.shape[0]


def collect_rollout(rng: np.random.Generator, days: float = 14.0,
                    dt_seconds: float = 45.0,
                    setpoints=PAPER_SETPOINTS,
                    mode: str = "pid",
                    dither: float = 0.18) -> dict:
    """One excitation rollout; returns coarse-grid arrays.

    ``mode`` is ``"pid"`` (expert prior plus dither) or ``"prbs"``
    (open-loop amplitude-modulated pseudo-random control).
    """
    series = random_scenario(rng, days=days, dt_seconds=dt_seconds)
    plant = BSM1Plant()
    pid = IncrementalPID(kp=PAPER_KP, ki=PAPER_KI, kd=PAPER_KD,
                         u_min=np.array([KLA5_BOUNDS[0], QA_BOUNDS[0]]),
                         u_max=np.array([KLA5_BOUNDS[1], QA_BOUNDS[1]]),
                         u0=np.array([84.0, 55338.0]))
    r = np.asarray(setpoints, dtype=float)
    u_lo = np.array([KLA5_BOUNDS[0], QA_BOUNDS[0]])
    u_hi = np.array([KLA5_BOUNDS[1], QA_BOUNDS[1]])
    u_range = u_hi - u_lo

    # PRBS schedule: dwell times of 5-60 min drawn independently per channel
    hold = np.zeros(2, dtype=int)
    level = np.array([84.0, 55338.0])
    walk = np.zeros(2)

    s = plant.measurement()
    rec_s, rec_u, rec_d, rec_nh = [], [], [], []
    for k in range(series.n_steps):
        if mode == "prbs":
            for j in range(2):
                if hold[j] <= 0:
                    hold[j] = int(rng.integers(int(300 / dt_seconds),
                                               int(3600 / dt_seconds)))
                    lo = (0.04 if j == 0 else 0.05) * u_range[j]
                    level[j] = rng.uniform(lo, 0.95 * u_range[j])
                hold[j] -= 1
            u = level.copy()
            pid(r - s)                      # keep the prior's state warm
        else:
            walk = np.clip(0.995 * walk + rng.normal(0.0, dither * 0.1, 2),
                           -dither, dither)
            u = pid(r - s) * (1.0 + walk) + walk * 0.35 * u_range
        u = np.clip(u, u_lo, u_hi)

        if k % COARSE_STRIDE == 0:
            rec_s.append(s.copy())
            rec_u.append(u.copy())
            rec_d.append([series.flow[k], series.composition[k, asm1.S_NH],
                          series.composition[k, asm1.S_S]])
            rec_nh.append(float(plant.z_effluent[asm1.S_NH]))
        s = plant.step(dt_seconds, series.composition[k], float(series.flow[k]), u)

    return {"s": np.array(rec_s), "u": np.array(rec_u), "d": np.array(rec_d),
            "nh": np.array(rec_nh)[:, None], "weather": series.weather,
            "mode": mode}


def build_corpus(n_rollouts: int = 24, seed: int = 0, seq_len: int = 64,
                 days: float = 14.0, prbs_fraction: float = 0.6) -> TwinCorpus:
    """Collect rollouts and cut them into fixed-length training sequences."""
    rng = np.random.default_rng(seed)
    S, U, D, SN, NH = [], [], [], [], []
    for i in range(n_rollouts):
        mode = "prbs" if i < int(round(prbs_fraction * n_rollouts)) else "pid"
        roll = collect_rollout(rng, days=days, mode=mode)
        s = roll["s"] / S_SCALE
        u = roll["u"] / U_SCALE
        d = roll["d"] / D_SCALE
        nh = roll["nh"] / NH_SCALE
        n = (len(s) - 1) // seq_len
        for j in range(n):
            a, b = j * seq_len, (j + 1) * seq_len
            S.append(s[a:b]); U.append(u[a:b]); D.append(d[a:b])
            SN.append(s[a + 1:b + 1]); NH.append(nh[a:b])
        print(f"  rollout {i + 1}/{n_rollouts} ({roll['mode']}, "
              f"{roll['weather']})", flush=True)

    return TwinCorpus(np.asarray(S, np.float32), np.asarray(U, np.float32),
                      np.asarray(D, np.float32), np.asarray(SN, np.float32),
                      np.asarray(NH, np.float32))
