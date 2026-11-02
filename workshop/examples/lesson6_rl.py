"""Lesson 6, hands-on: reinforcement learning, from reward alone.

Run it:   python workshop/examples/lesson6_rl.py      (about 15 s)

  1 ENVIRONMENT  reset() and step(action) → state, reward, done   (as src/wwtp/rl/env.py)
  2 REWARD       one number per step: what we want, written down  (as src/wwtp/rl/rewards.py)
  3 ACT          the policy picks an action, sometimes a random one to explore
  4 LEARN        update from what just happened (Q-learning's Bellman update)
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/davidfernand/BSM1_in_Python/5dadb209a5a34ecb0e2f83a2e79c060bf3470582/code/Influent%20files/"
if not (DATA / "Inf_dry_2006.xlsx").exists():
    urllib.request.urlretrieve(URL + "Inf_dry_2006.xlsx", DATA / "Inf_dry_2006.xlsx")
d = pd.read_excel(DATA / "Inf_dry_2006.xlsx").astype(float).values[:-1]
load = d[:, 14] * (d[:, 10] + 0.2 * d[:, 2])                            # flow × (ammonium + some COD)
LOAD = np.repeat(load / load.mean(), 3)                                  # 15 min → 5 min steps


# ── 1 ENVIRONMENT: one aerated tank. Oxygen in from the air, oxygen used by the bacteria.
class OxygenTank:
    ACTIONS = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])                    # change of the air (KLa)

    def __init__(self, day, energy_weight=0.1, seed=0):
        self.day, self.w, self.rng = day, energy_weight, np.random.default_rng(seed)

    def reset(self):
        self.k0, self.k, self.kla, self.S, self.e1 = int(self.day * 288), 0, 120.0, 2.0, 0.0
        return self.state()

    def state(self):
        return np.array([self.S - 2.0, self.e1, self.kla / 240.0, LOAD[self.k0 + min(self.k, 287)]])

    def step(self, dkla):
        self.e1 = self.S - 2.0
        self.kla = float(np.clip(self.kla + dkla, 0.0, 240.0))
        our = 700.0 * LOAD[self.k0 + self.k] * (1 + 0.05 * self.rng.normal())
        s_eq = 8.0 - our / max(self.kla, 1e-6)
        self.S = max(0.0, s_eq + (self.S - s_eq) * np.exp(-self.kla * 5 / 1440))
        self.k += 1
        # ── 2 REWARD: be at 2 mg/L, don't waste air
        reward = -abs(self.S - 2.0) - self.w * self.kla / 240.0
        return self.state(), reward, self.k >= 288


def run(env, policy):
    s, done, R, err = env.reset(), False, 0.0, []
    while not done:
        s, r, done = env.step(policy(s)); R += r; err.append(abs(env.S - 2))
    return R, np.mean(err), env.kla


pid = lambda s: float(np.clip(80 * (-s[0] + s[1]) + 25 * -s[0], -20, 20))   # the expert: a PID, no learning
print(f"1 ENVIRONMENT  PID on an unseen day: error {run(OxygenTank(10), pid)[1]:.3f} mg/L")

# ── 3 + 4: Q-learning. A table Q[state, action] = how good this move is here, as far as we know.
E, K, L = np.array([-.5, -.25, -.1, .1, .25, .5]), np.linspace(0, 1, 17)[1:-1], np.array([.6, .8, 1., 1.2, 1.4])
cell = lambda s: (np.digitize(s[0], E), np.digitize(s[2], K), np.digitize(s[3], L))
greedy = lambda Q: (lambda s: OxygenTank.ACTIONS[int(np.argmax(Q[cell(s)]))])


def q_learning(energy_weight=0.1, episodes=1500, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((7, 16, 6, 5))
    best = (-np.inf, None)
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (0.6 * episodes))                      # explore a lot at first, then less
        env = OxygenTank(int(rng.integers(0, 7)), energy_weight, seed=ep)
        s, done = env.reset(), False
        while not done:
            c = cell(s)
            a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[c]))   # ── 3 ACT
            s, r, done = env.step(OxygenTank.ACTIONS[a])
            target = r + (0.0 if done else 0.9 * Q[cell(s)].max())
            Q[c + (a,)] += 0.2 * (target - Q[c + (a,)])                  # ── 4 LEARN (Bellman update)
        if ep % 100 == 99:
            v = run(OxygenTank(6, energy_weight), greedy(Q))[0]
            if v > best[0]:
                best = (v, Q.copy())
    return best[1]


Q = q_learning()
print(f"3-4 Q-LEARNING trained on days 0-6; on unseen day 10: error {run(OxygenTank(10), greedy(Q))[1]:.3f} mg/L")

ff = {"last": None}
def pid_ff(s):                                                           # PID + one line of physics
    target = 700.0 * s[3] / 6.0                                          # air that holds 2 mg/L at this load
    d = 0.0 if ff["last"] is None else target - ff["last"]
    ff["last"] = target
    return float(np.clip(pid(s) + d, -20, 20))
print(f"    BASELINE   PID + physics feedforward, no training: error {run(OxygenTank(10), pid_ff)[1]:.3f} mg/L")

Q5 = q_learning(energy_weight=5.0)
R, err, kla = run(OxygenTank(10, 5.0), greedy(Q5))
print(f"2 REWARD       air made 50× dearer → the agent sets the air to {kla:.0f} (PID: about 120); oxygen error {err:.2f} mg/L")

# ✏️ Try: energy_weight=2.0, then 3.0. Where does the agent's behaviour flip?
