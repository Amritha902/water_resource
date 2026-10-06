# Water resource project — RL control of a wastewater treatment plant

This repo has three parts:

1. A simulation of the BSM1 wastewater treatment plant, written from scratch.
2. Reproductions of the two papers we are building on.
3. Our own change to the RL network, which is the new contribution.

Nothing here is finished yet. `STATUS.md` says exactly what works, what the
numbers are so far, and what is left.

---

## The two papers

### Paper 1 — Wang, Li, Ren, Qiao (IEEE T-II, 2026)
*Multiagent Adaptive Critic Control With Expert Knowledge for WWTPs*

The plant has two things that must be held steady:

- dissolved oxygen in tank 5 (`S_O,5`), adjusted with the air transfer rate `KLa_5`
- nitrate in tank 2 (`S_NO,2`), adjusted with the internal recycle flow `Q_a`

These two loops interfere with each other. More recycle flow brings nitrate
back for denitrification, but it also drags oxygen-rich water backwards. More
air changes how much nitrate there is to recycle in the first place.

What the paper does about it:

- Treat the plant as two connected subsystems and give each one its own agent,
  so each agent only has to learn one simple control law instead of a joint one.
- Each agent's cost function includes **everyone's** tracking error, not just
  its own. That is how the coupling is handled without ever having to model it.
- The applied control is `u = a + b`, where `b` is a normal PID controller
  (the "expert knowledge") and `a` is what the agent learns. The learned part
  starts at exactly zero, so on day one the plant just runs on PID and nothing
  risky happens.
- The agent outputs an *increment* to the control, not the control itself, so a
  disturbance only corrupts one small step.

### Paper 2 — Du, Chen, Han, Qiao (Sci China Tech Sci, 2023)
*Dissolved oxygen concentration control in WWTP based on reinforcement learning*

This one asks a different question. Everyone else fixes the DO setpoint at
2 mg/L and tries to track it accurately. Du et al. point out that tracking a
fixed number is the wrong goal — the influent changes all day, so sometimes
2 mg/L is more oxygen than you need and you are just paying for air.

So they throw the setpoint away. A DDPG agent writes increments straight onto
`KLa_5` and lets DO float wherever the trade-off puts it:

- **state** `[S_S,5, S_O,5, S_NH,5]` — the three tank-5 concentrations that
  actually drive the oxygen balance
- **action** the increment on `KLa_5`
- **reward A** (they call it DDPG-A): `-(max(NH_e - 4, 0) + 0.38 max(N_tot,e - 18, 0))`
  — only cares about effluent quality
- **reward B** (DDPG-B): `-(AE + 0.42 max(NH_e - 4, 0))` — also pays for the air
- network: actor 3→256→256→1, critic 4→256→256→1, γ=0.99, batch 256,
  buffer 30000, soft update 0.001

They train on week 1 of the BSM1 influent file and test on week 2. DDPG-B cuts
aeration energy by 5–7% versus PID while effluent still passes.

The reason this works is a conflict in the plant: **more oxygen lowers effluent
ammonia but raises effluent total nitrogen.** You cannot minimise both. Our
simulator reproduces this (see `docs/04_second_paper.md`).

---

## What we are trying to do

Paper 1 gives a good *tracking* controller. Paper 2 gives a good *goal* —
spend less energy, keep the effluent legal. Put together:

- paper 2's job (decide how much air to use), run with
- paper 1's ideas (expert prior so it is safe from step one, incremental action,
  cost that accounts for the other loop), and
- our own change to the RL network, which is the actual novelty.

Our change, in one line: **the agent in paper 2 can only react to what has
already happened to the plant. We give it a forecast of what is coming, and we
make it care about the worst case instead of only the average.**

Concretely, four modifications (details in `docs/03_novelty.md`):

1. **Forecast input.** A separate network predicts the next 2 hours of influent
   (flow, ammonia, COD) with uncertainty, from 24 h of inlet history. Those
   predictions go into the actor and critic. So the agent can cut air *before*
   the load drops, instead of after.
2. **Risk-aware critic.** Instead of one Q-value, the critic predicts the whole
   distribution of returns (quantile regression) and the policy optimises the
   bad tail (CVaR). Discharge limits are legal limits — the tail is what gets
   you fined, not the mean.
3. **Constraint instead of a guessed penalty weight.** Paper 2 hand-tunes
   α2 = 0.38 and β2 = 0.42 by trial and error. We replace that with a Lagrange
   multiplier that adapts itself until the violation rate hits a target you
   actually specify.
4. **Learned plant model for sample efficiency.** A one-step model of the plant
   generates extra training data (Dyna), so the agent needs fewer weeks of real
   interaction. The model is accurate to 0.11 mg/L on DO and 0.05 mg/L on
   effluent ammonia per 15-minute step, and it recovered the Figure-4 trade-off
   from data rather than being told it.

"PANDA-MAACC" is our name for the combined thing — paper 1's controller plus
these prediction-driven changes. It is ours, not from either paper.

---

## Reproduce first

Before claiming anything new we have to match what they published. Where we
stand:

| dry weather, evaluation week | paper 2 | ours |
|---|---|---|
| PID aeration energy | 3698.2 | 3719.7 |
| Fuzzy aeration energy | 3697.1 | 3719.9 |
| DDPG-B aeration energy | 3433 (−7.2%) | 3512 (−5.6%) |
| DDPG-B mean `S_O,5` | 1.032 | 1.074 |

The comparators are 0.6% apart on a plant rebuilt from scratch, and the central
claim — let DO float and you save 5–7% of the aeration energy — reproduces.
Effluent quality moves the way they report too: our DDPG-B gives EQ 5530
against PID's 5411, the EQ/AE coupling their Table 2 shows.

We also reproduced their Figure 4 on our plant: effluent total nitrogen rises
with DO while ammonia falls, and COD/BOD/TSS stay flat. That conflict is the
thing the whole RL design exists to resolve.

**But the published recipe does not learn as written.** It collapses to zero
aeration and stays there. Three reasons, all found by instrumenting it and all
written up in `docs/04_second_paper.md` §5:

1. the state `[S_S,5, S_O,5, S_NH,5]` plus an *incremental* action makes the
   actuator position a hidden integrator, so the MDP is not Markov;
2. at 45 s per decision with γ=0.99 the horizon is 75 min, but ammonia responds
   to aeration over hours — so the energy saving is immediate and the penalty
   lands outside the horizon;
3. with γ=0.99 the critic must learn values ~100× the reward, and its `dQ/da`
   has the **wrong sign** for four simulated weeks, by which time the tanh actor
   has saturated where its gradient vanishes.

Fixing those three is what makes the numbers above possible, and each one also
points at one of our modifications. Results for every method and weather are in
`docs/05_results.md`.

---

## Layout

```
src/wwtp/bsm1/        the plant: ASM1 biology, Takacs settler, 5 tanks, influent
src/wwtp/baselines/   PID (expert prior from paper 1)
src/wwtp/maacc/       paper 1 reproduced
src/wwtp/rl/          paper 2 reproduced: environment, rewards, DDPG, PID/fuzzy
src/wwtp/forecast/    the influent forecaster
src/wwtp/twin/        the digital twin
src/wwtp/panda/       our controller
experiments/          numbered scripts, run them in order
docs/                 00 code walkthrough · 01 paper 1 · 02 paper 1 reproduction
                      03 our contribution · 04 paper 2 + reproduction findings
                      06 a negative result we kept
tests/                28 tests
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q

python experiments/01_train_forecaster.py     # ~10 min
python experiments/02_train_twin.py           # ~40 min
python experiments/03_run_control.py          # paper 1 benchmark
python experiments/06_aeration_benchmark.py   # paper 2 + ours  (the main one)
python experiments/07_figures.py              # figures from the results
```

Checkpoints for the forecaster and the twin are committed in `artifacts/`, so
you can skip steps 01 and 02.
