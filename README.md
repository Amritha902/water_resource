# Water resource project — RL control of a wastewater treatment plant

Two papers reproduced on a BSM1 plant built from scratch, plus our own change
to the RL network. Every number below comes from `results/`, which is in the
repo; `docs/05_results.md` is generated from those files by
`experiments/09_write_results.py`, so nothing here is typed by hand.

---

## The plant, in one paragraph

BSM1 is the standard benchmark plant: five tanks (two anoxic, three aerated)
and a settling clarifier. Two things are controlled:

- **dissolved oxygen in tank 5** (`S_O,5`), by the air transfer rate `KLa_5`
- **nitrate in tank 2** (`S_NO,2`), by the internal recycle flow `Q_a`

They interfere. More recycle brings nitrate back for denitrification but also
drags oxygen-rich water backwards; more air changes how much nitrate exists to
recycle in the first place.

---

## Base paper 1 — Wang, Li, Ren, Qiao (IEEE T-II, 2026)

*Multiagent Adaptive Critic Control With Expert Knowledge for WWTPs*

What we understood it to do:

- Treat the plant as two coupled subsystems, one agent each, so each agent
  learns one simple control law instead of a joint one.
- Give each agent a cost that includes **everyone's** tracking error — that is
  how the coupling is handled without ever modelling it.
- Apply `u = a + b`, where `b` is a normal PID (the "expert knowledge") and `a`
  is learned. The learned part starts at exactly zero, so day one is just PID
  and nothing risky happens.
- Emit an *increment* to the control rather than the control itself, so a
  disturbance corrupts only one small step.

## Base paper 2 — Du, Chen, Han, Qiao (Sci China Tech Sci, 2023)

*Dissolved oxygen concentration control in WWTP based on reinforcement learning*

A different question, and the more useful one. Everyone else holds DO at
2 mg/L and tries to track it accurately. Du et al. point out that tracking a
fixed number is the wrong goal — the influent changes all day, so sometimes
2 mg/L is more air than you need and you are simply paying for it.

So they throw the setpoint away. A DDPG agent writes increments straight onto
`KLa_5` and lets DO float wherever the trade-off puts it:

- **state** `[S_S,5, S_O,5, S_NH,5]`, **action** the increment on `KLa_5`
- **reward A** quality only; **reward B** adds the air: `-(AE + 0.42·max(NH_e − 4, 0))`
- train on week 1 of the influent file, test on week 2

It works because of a conflict in the plant: **more oxygen lowers effluent
ammonia but raises effluent total nitrogen.** You cannot minimise both. We
reproduced that conflict on our plant before trusting anything else.

---

## What we are trying to do

Paper 1 gives a safe *controller*. Paper 2 gives the right *goal*. Our
contribution is to the RL network itself:

1. **Forecast input** — a separate network predicts the next 2 h of influent
   with uncertainty; the actor and critic read it. The agent can cut air
   *before* the load drops instead of after.
2. **Risk-aware critic** — predicts the whole distribution of returns and
   optimises the bad tail (CVaR), not the mean. A discharge limit is about
   occurrences, not averages.
3. **Constraints instead of guessed weights** — paper 2 hand-tunes `0.38` and
   `0.42`. We replace them with Lagrange multipliers driven by a violation
   rate the operator actually specifies.
4. **Learned plant model** for sample efficiency (Dyna).

---

## Verified results

**Reproduction.** Our comparators sit within 1.2 % of the published aeration
energy, and paper 2's central claim holds:

| dry weather | paper 2 | ours |
|---|---|---|
| PID aeration energy | 3698.2 | 3719.7 |
| Fuzzy aeration energy | 3697.1 | 3719.9 |
| DDPG-B energy saving | −7.24 % | −6.8 % |
| (rain) DDPG-B saving | −5.52 % | −4.9 % |
| (storm) DDPG-B saving | −6.06 % | −5.0 % |

**But the saving is not free — this is the main finding.** The papers report
energy and effluent averages, not how often the limit is broken:

| weather | DDPG-B saving | DDPG-B time above 4 g N/m³ | PID |
|---|---|---|---|
| dry | +6.8 % | 38.1 % | 12.2 % |
| rain | +4.9 % | 35.2 % | 19.6 % |
| storm | +5.0 % | 37.0 % | 23.2 % |

**Our network change earns its place.** Same objective, same swept weight,
arms differing *only* in the network. Each arm's Pareto front measured against
DDPG-B's at matched violation rate (positive = cheaper at the same risk):

| arm | what it adds | mean gap |
|---|---|---|
| forecast only | modification 1 | +19.5 kWh/d |
| CVaR only | modification 2 | +25.9 kWh/d |
| **both** | | **+28.6 kWh/d** |

**Paper 1 works where the paper says it does.** Under its own noise protocol
MAACC improves nitrate tracking by **+28.9 %** (dry) and **+22.8 %** (rain)
against the PID prior.

Full tables, all three weathers and every seed: `docs/05_results.md`.

---

## What is *not* proven

- **Every sweep point is one seed.** Per-point gaps are published so you can
  see whether a mean rests on one run.
- **The network comparison is not parameter-matched** — our arm reads six
  extra inputs and has 32 critic outputs against 1, so some of the gap could
  be capacity.
- **The Dyna fix failed.** Model-based updates do buy sample efficiency
  (3749 → 3205 kWh/d at equal real experience) but broke the constraint
  (14 % → 67 % violations), and the intra-episode dual we wrote to fix it did
  not (68 %). Diagnosis in `docs/03_novelty.md` §4.
- **Two earlier claims were withdrawn** when the full tracking benchmark ran:
  that MAACC never beats a well-tuned PID (it does, under noise), and that a
  de-tuned prior gives the learned term room (it does not — MAACC is worse
  everywhere in that condition).
- The plant is a faithful re-implementation, not the official BSM1 code, and
  the influent is statistically equivalent rather than identical. Absolute
  values are not directly comparable with either paper's tables; every claim
  here is a comparison *between methods on the same simulator*.

---

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # 72 tests, ~1 min
python experiments/09_write_results.py         # regenerate docs/05_results.md
python experiments/07_figures.py               # regenerate the figures
```

Trained checkpoints are in `artifacts/`, so the experiment scripts
(`01`–`12`, run in order) only need re-running to reproduce from scratch.

| doc | what it covers |
|---|---|
| `docs/00_code_walkthrough.md` | every module, in plain language |
| `docs/01_paper_summary.md` | base paper 1 in detail |
| `docs/02_reproduction.md` | reproducing it, and a correction to its eq. (15) |
| `docs/03_novelty.md` | our contribution, with a per-modification verdict table |
| `docs/04_second_paper.md` | base paper 2, and three corrections its recipe needs |
| `docs/05_results.md` | all results (generated) |
| `docs/06_tracking_layer_feedforward.md` | a negative result we kept |
| `STATUS.md` | what is done, what is not, what to do next |
