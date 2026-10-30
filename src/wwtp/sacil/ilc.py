"""Iterative learning control (ILC) of the daily load cycle, for paper 1's two loops.

The BSM1 influent repeats a daily pattern.  A feedback PID only reacts once
the morning load peak has already pushed the oxygen and nitrate away from
their set points.  ILC learns, day after day, a feedforward input profile over
the time of day:

    u_k = u_PID,k + f(phase_k)
    f(phase) <- Q[ f(phase) + beta * (u_PID(phase + lead) - daily mean) ]   after each day

* ``u_PID`` is what the feedback PID had to supply at that time of day;
* ``beta`` is the share of it the feedforward takes over per day, and
  ``lead`` how much earlier it is applied;
* ``Q`` smooths the profile over neighbouring phases (a moving average) and
  slightly forgets (factor ``q``), which keeps the learning robust to noise
  and to days that differ from the others (weekends, rain, storms).

The profile is updated online at the end of each day of the run; it starts at
zero unless a profile learned beforehand is given.
"""

from __future__ import annotations

import numpy as np

from ..official.plant import DT_DAYS

PERIOD = int(round(1.0 / DT_DAYS))           # control periods per day (1920)


class DailyILC:
    """Feedback-error learning of the daily cycle.

    The PID's own input ``u_fb`` (its integrated state, without the
    feedforward) is recorded by time of day.  After each day the feedforward
    takes over a share ``beta`` of the PID's daily variation, applied ``lead``
    periods earlier:

        f(phase) <- Q[ f(phase) + beta * (u_fb(phase + lead) - mean_day u_fb) ]

    Over the days the PID has to correct less and less, and the input arrives
    ahead of the load instead of after it."""

    def __init__(self, beta, lead, smooth=15, forget=1.0, profile=None):
        self.beta = np.asarray(beta, float)
        self.lead = np.asarray(lead, int)
        self.w, self.q = int(smooth), float(forget)
        n = len(self.beta)
        self.f = np.zeros((PERIOD, n)) if profile is None else np.array(profile, float)
        self._u = np.zeros((PERIOD, n))

    def ff(self, k):
        return self.f[k % PERIOD]

    def record(self, k, e, u_fb):
        self._u[k % PERIOD] = u_fb
        if (k + 1) % PERIOD == 0:
            self._learn()

    def _learn(self):
        new = self.f.copy()
        kern = np.ones(2 * self.w + 1) / (2 * self.w + 1)
        for i in range(len(self.beta)):
            dev = self._u[:, i] - self._u[:, i].mean()
            upd = self.f[:, i] + self.beta[i] * np.roll(dev, -self.lead[i])
            pad = np.concatenate([upd[-self.w:], upd, upd[: self.w]])     # the day is periodic
            new[:, i] = self.q * np.convolve(pad, kern, mode="valid")
        self.f = new
