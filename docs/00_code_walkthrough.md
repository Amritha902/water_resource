# Code walkthrough — how to read this repository

This is the guide to read **before** opening the code. It explains every
module in plain language: what it takes in, what it gives back, and why it
was built that way. It is ordered the way the data flows — plant first, then
the controllers, then the learned models that sit on top.

```
                    ┌──────────────────────────────────────────────┐
 influent ─────────►│  BSM1 plant  (bsm1/)                         │──► effluent
 (bsm1/influent.py) │  5 reactors + clarifier, ASM1 biology        │
                    └──────────────┬───────────────────────▲───────┘
                     measurements  │                       │ KLa_5, Q_a
                                   ▼                       │
                    ┌──────────────────────────────────────┴───────┐
                    │  controller:  PID  |  MAACC  |  PANDA        │
                    └──────────────────────────────────────────────┘
                                   ▲
         learned models ───────────┤
         forecast/  influent forecaster (what is about to arrive)
         twin/      digital twin (how the plant will respond)
         warning/   effluent-ammonium early warning (will we violate?)
         rl/        aeration agents: how much oxygen to use at all
```

`envs.py` is the harness that wires one controller to the plant, runs the
simulation and records everything; `metrics.py` turns the recording into
numbers.

---

## 1. The plant — `src/wwtp/bsm1/`

BSM1 (Benchmark Simulation Model no. 1) is the standard test plant of the
wastewater-control community. Every paper in the field, including the base
paper, reports results on it.

| file | what it is | input → output |
|---|---|---|
| `asm1.py` | **The biology.** Activated Sludge Model no. 1: 13 concentrations (substrate, biomass, oxygen, nitrate, ammonium, ...) and 8 reactions (growth of heterotrophs and nitrifiers, decay, hydrolysis, ammonification). `reaction_rates(z)` returns d(concentration)/dt caused by the bacteria. | state of 5 reactors (5×13) → rates (5×13) |
| `settler.py` | **The clarifier.** Ten horizontal layers; sludge sinks with the Takács double-exponential settling velocity. Splits the flow into clean effluent (top) and thick return sludge (bottom). | feed solids, flows → effluent and underflow concentrations |
| `plant.py` | **The whole plant.** Reactors 1–2 anoxic (no air, bacteria use nitrate → denitrification), 3–5 aerated (nitrification of ammonium). Mass balances + biology + aeration term `KLa·(S_O,sat − S_O)`. `step(dt, influent, flow, u)` advances time. | control `u = [KLa_5, Q_a]` → measurement `s = [S_O,5, S_NO,2]` |
| `influent.py` | **What arrives at the plant.** Dry / rain / storm 14-day profiles built from the documented BSM1 statistics (diurnal double peak, weekend drop, rain dilution, storm first-flush). `random_scenario(rng)` gives randomised versions for training ML models. | seed → flow + 13 concentrations every 45 s |

**The two control loops** (both from the paper):

* `KLa_5` (air blown into reactor 5) controls `S_O,5` (dissolved oxygen in
  reactor 5). More air → more oxygen.
* `Q_a` (internal recycle pump, reactor 5 → reactor 1) controls `S_NO,2`
  (nitrate in reactor 2). More recycle → more nitrate carried back.

These two loops are **coupled**: more air in reactor 5 also means more
dissolved oxygen recycled to the anoxic zone, which disturbs denitrification.
That coupling is the whole reason the paper uses a *multi-agent* design.

**How it was validated:** `tests/test_bsm1.py` checks the open-loop steady
state against the published BSM1 numbers, that concentrations stay
non-negative, that the settler thickens sludge >50×, and that both control
channels move the plant in the right direction.

---

## 2. The harness and the metrics — `envs.py`, `metrics.py`

`run_closed_loop(controller, influent_series, ...)`:

1. warms the plant up for one day under PID so every controller starts from
   the same state;
2. every 45 s: read the (optionally noisy / lagged / stale) measurement →
   ask the controller for `u` → optionally rate-limit and add actuator
   noise → step the plant;
3. returns a `RunResult` holding every trajectory.

The optional arguments are the **instrumentation realism knobs**:

| argument | real-world meaning |
|---|---|
| `actuator_noise_std`, `sensor_noise_std` | the paper's noise protocol |
| `slew_rate` | blowers and pumps cannot jump instantly |
| `sensor_tau_seconds` | DO probes respond with a 1–2 min lag |
| `analyzer_period_seconds` | nitrate analysers only update every few minutes |

`metrics.py` computes the paper's tracking indices (IAE, ISE, DEVmax — eq. 36)
and the BSM1 plant indices: effluent quality **EQ**, energy (**AE**, **PE**,
**OCI**), and **violation fractions** against the discharge limits
(NH₄-N 4, TN 18, COD 100, BOD₅ 10, TSS 30 g/m³).

---

## 3. The controllers

### 3.1 Expert prior — `baselines/pid.py`, `baselines/pid_controller.py`

Incremental (velocity-form) PID with the paper's gains
`Kp = diag{100, 100000}, Ki = diag{20, 30000}, Kd = diag{10, 1000}`.
It outputs a *change* `Δu` each step, so it has integral action built in.
This is `b_k` in the paper's control law.

### 3.2 MAACC, the base paper — `maacc/agent.py`, `maacc/networks.py`

`u_k = b_k + a_k`: the PID's move plus a learned correction.

One **agent per loop**. Each agent owns

* an **action network** (actor): own tracking error → increment `Δa`;
* a **critic network**: full error vector + `Δa` + `a` → estimated
  cost-to-go `Q`.

The utility (eq. 10) charges each agent for its **own** error, the
**other** loop's error (the coupling term), and its control effort. The
critic learns `Q` from the Bellman equation; the actor moves in the
direction that lowers `Q`. `a₀ = 0`, so on day one the controller *is* the
PID — safe from the start.

`networks.py` holds the small single-hidden-layer `tanh` nets the paper uses,
trained online with plain gradient steps (no PyTorch — matches the paper).

**Correction made (documented in `docs/02_reproduction.md` §3):** the
paper's critic update eq. (15) diverges on slowly varying states. We use
the standard TD(0) direction on the same residual.
`tests/test_critic_update.py` shows both.

### 3.3 PANDA, the prediction-driven extension — `panda/controller.py`

`u_k = b_k + g_k·a_k + f_k`

* `b_k` — PID prior (unchanged);
* `a_k` — MAACC agents, but their networks also see a 6-number summary of
  the influent forecast;
* `g_k` — **uncertainty gate** in (0, 1]: if the forecast is unsure, shrink
  the learned term and trust the PID;
* `f_k` — **anticipatory feed-forward**: a move computed *before* the
  disturbance arrives, by differentiating the digital twin through the
  forecast influent.

Setting `g = 1, f = 0` gives back exactly the paper's controller.

---

## 4. The learned models

### 4.1 Influent forecaster — `forecast/`

* `dataset.py` — builds training data from 160 random influent series:
  inlet flow, ammonium, readily-biodegradable COD on a 15-min grid; 24 h of
  history in, 2 h out.
* `models.py` — a **dilated causal TCN** (convolutions that only look
  backwards in time, with dilations 1…32 so it sees the full 24 h). Two
  heads: 10/50/90 % quantiles (uncertainty) and a dry/rain/storm classifier.
* `predictor.py` — the online wrapper: feed one measurement per step, get
  the median forecast, the spread, and the 6-number `context()` vector.
* **Result:** regime accuracy 95.3 %; checkpoint `artifacts/forecaster.pt`.

### 4.2 Digital twin — `twin/`

* A GRU network that predicts the next `[S_O,5, S_NO,2]` and effluent
  ammonium from the current state, control and inlet.
* **Correction made:** trained first on PID-controlled data it learned that
  more air barely raises oxygen (the PID raises air *exactly when* oxygen
  is being consumed, which hides the true effect). Fixed by adding 60 %
  open-loop random excitation data. `tests/test_learned_models.py` asserts
  the gain signs.

### 4.3 Effluent-ammonium early warning — `warning/` (new)

See `docs/05_early_warning.md` for the full write-up. In short: from signals
the plant already measures, predict **whether** effluent ammonium will
break the 4 g N/m³ limit in the next 12–120 min and **how high** it will go,
early enough for an operator to act.

| file | role |
|---|---|
| `dataset.py` | simulates plants at randomised operating points (aeration, temperature, DO set point), records 17 measured signals on a 6-min grid, and builds look-ahead labels |
| `models.py` | causal TCN + quantile head + discrete-time hazard head |
| `baselines.py` | analyser alarm (today's practice), trend extrapolation, gradient boosting |
| `evaluate.py` | AUROC/AUPRC/calibration, lead time, false alarms per day |
| `train.py` | training loop, save/load |

---

## 5. The aeration RL layer — `src/wwtp/rl/`

This is paper 2's problem and our modified agent. Paper 1 asks "hold DO at
its set point"; paper 2 asks "how much oxygen should we be using at all",
which is where the energy is. Nothing here tracks a set point.

| file | what it is | input → output |
|---|---|---|
| `env.py` | **The aeration MDP.** State `[S_S,5, S_O,5, S_NH,5]`, action writes `KLa_5`, one step advances the plant and returns the effluent and energy over the interval. `Q_a` stays on paper 1's nitrate controller, identically for every method, so only aeration is being judged. Switches for the three corrections: `obs_mode` (`"paper"` or `"augmented"`, which appends `KLa_5/240` to restore the Markov property), `action_mode` (`"incremental"` as published, or `"absolute"`), and `action_interval` (control periods per decision). | action in [-1, 1] → next observation, info dict |
| `rewards.py` | **The two published reward functions.** `reward_eq` is their eq. (19), effluent quality only; `reward_ae_eq` is eq. (23), energy plus ammonium. Coefficients are the published `0.38` and `0.42`. The one judgement call — what units `AE` enters the reward in — is argued in the module docstring. | info dict → scalar |
| `baselines.py` | **The comparators.** Incremental PID with their gains (Kp 200, Ki 15, Kd 2) and a Mamdani fuzzy controller, both holding DO at 2 mg/L. Both land within 0.6% of the published aeration energy. | observation → action |
| `ddpg.py` | **The published agent.** Actor `3-256-256-1` (ReLU, tanh), critic `4-256-256-1`, γ 0.99, batch 256, buffer 30 000, soft update 0.001. Plus the three fixes the reproduction needed: reward scaling by `1-γ`, actor weight decay, and a penalty on the pre-tanh output so a saturated policy is not stuck at an actuator rail. | transitions → policy |
| `panda_rl.py` | **Our agent.** Same skeleton, four changes: the actor and critic also read the 6-D influent forecast; the critic emits 32 return quantiles and the actor maximises their CVaR instead of the mean; each discharge limit gets a cost critic and a Lagrange multiplier driven by dual ascent on the measured violation rate; and a hook for twin-generated synthetic updates. Reduces exactly to DDPG when all four are switched off. | transitions → policy |
| `train.py` | **The protocol.** Learn on week 1 of an influent file, evaluate with exploration off on week 2, report means over that week — Du et al. Sec. 4.1. Carries the forecast context and per-constraint costs through the buffer and runs the dual ascent at the end of each episode. | agent + influent → history, summary |

**The three corrections** are the substance of the reproduction and are
explained in `docs/04_second_paper.md` §5: the published observation is not
Markov because the actuator is a hidden integrator; a 45 s decision interval
puts the ammonium consequence outside the agent's horizon while the energy
saving is immediate; and at γ = 0.99 the critic's `dQ/da` has the wrong sign
for four simulated weeks.

## 6. Scripts — `experiments/`

| script | does | time (4 CPU) |
|---|---|---|
| `01_train_forecaster.py` | trains the influent forecaster | ~10 min |
| `02_train_twin.py` | collects excitation data, trains the twin | ~40 min |
| `03_run_control.py` | PID vs MAACC vs PANDA benchmark | long |
| `04_collect_warning_data.py` | simulates the early-warning corpus | ~15 min |
| `05_warning_benchmark.py` | trains + evaluates the early warning, writes figures | ~30 min |
| `06_aeration_benchmark.py` | **the main one.** PID / fuzzy / DDPG-A / DDPG-B / PANDA-RL (+ ablations) over dry, rain and storm | ~2 h |
| `07_figures.py` | figures from the benchmark results | ~1 min |

## 7. Tests — `tests/`

Run `python -m pytest tests -q`. Each file guards one layer: plant physics,
control loop, critic convergence, learned-model sanity, early-warning labels
and metrics, and the aeration MDP plus both RL agents (`test_rl.py`).
