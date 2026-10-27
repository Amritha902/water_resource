# Paper 2 — Du et al. (2023), and what reproducing it taught us

> S. L. Du, P. X. Chen, H. G. Han, J. F. Qiao, **"Dissolved oxygen
> concentration control in wastewater treatment process based on
> reinforcement learning"**, *Science China Technological Sciences*, 66(9),
> 2549–2560, 2023. DOI 10.1007/s11431-022-2403-8

## 1. What the paper does

Everyone else holds the dissolved-oxygen concentration in tank 5 at a fixed
2 mg/L. Du et al. argue that is the wrong objective: the influent changes
through the day, so a constant set point is sometimes more oxygen than the
plant needs, and the extra air is simply paid for.

So they remove the set point. A DDPG agent writes increments onto the oxygen
transfer coefficient and lets `S_O,5` float:

| | |
|---|---|
| state | `s = [S_S,5, S_O,5, S_NH,5]` — the three tank-5 concentrations that appear in the oxygen balance (their eqs. 12–15) |
| action | `a` = increment, `KLa_5,t = KLa_5,t-1 + a_t` |
| reward A (**DDPG-A**), eq. (19) | `-( max(S_NH,e - 4, 0) + 0.38 max(N_tot,e - 18, 0) )` |
| reward B (**DDPG-B**), eq. (23) | `-( AE + 0.42 max(S_NH,e - 4, 0) )` |
| networks | actor `3-256-256-1` (ReLU, tanh out), critic `4-256-256-1` |
| hyper-parameters | `gamma = 0.99`, batch 256, buffer 30 000, soft update 0.001, lr 1e-4 / 3e-4 |
| protocol | learn on week 1 of the influent file, evaluate on week 2, 45 s steps |

Reported: DDPG-B cuts aeration energy by 5.5 % (rain), 7.2 % (dry) and 6.1 %
(storm) against PID while the effluent still passes; DDPG-A instead lowers the
effluent quality index. Mean `S_O,5` drops from 2.000 (PID) to 1.415 (DDPG-A)
and 1.032 (DDPG-B).

## 2. The physics the whole design rests on

Their Figure 4 sweeps the DO set point from 0 to 4 mg/L and shows that
effluent **total nitrogen rises** with DO while **ammonium falls**, and that
COD, BOD and TSS barely move. The two nitrogen limits therefore pull in
opposite directions and there is a genuine optimum to find.

Our simulator reproduces this. Steady influent, DO held at each level until
the plant settles:

| DO set point | `KLa_5` | `S_NH,e` | `N_tot,e` | COD | BOD5 | TSS | AE |
|---|---|---|---|---|---|---|---|
| 0.5 | 84.6 | 1.64 | 13.86 | 45.3 | 2.32 | 10.84 | 3345 |
| 1.0 | 106.8 | 1.08 | 15.04 | 45.3 | 2.32 | 10.84 | 3476 |
| 2.0 | 141.3 | 0.82 | 16.33 | 45.3 | 2.32 | 10.84 | 3681 |
| 3.0 | 183.6 | 0.72 | 17.27 | 45.3 | 2.31 | 10.84 | 3931 |
| 4.0 | 240.0 | 0.67 | 18.05 | 45.3 | 2.31 | 10.84 | 4266 |

Ammonium down, total nitrogen up, the other three flat, energy up — the same
shape as their Figure 4.

**One fix to the simulator was needed to get here.** Our first influent
generator had a flat pollutograph: the ammonium load peak was only 1.25× the
mean, so effluent ammonium never came close to the 4 mg/L limit and the
constraint in both reward functions was permanently inactive. A real municipal
pollutograph is much peakier than the hydrograph. After sharpening it to the
documented BSM1 statistics (`S_NH` influent 9–44 mg/L, load peak 1.5× mean),
the limits bind properly:

| | DO 0.5 | DO 1.0 | DO 2.0 | DO 3.0 |
|---|---|---|---|---|
| fraction of time `S_NH,e > 4` | 0.51 | 0.34 | 0.08 | 0.00 |
| fraction of time `N_tot,e > 18` | 0.07 | 0.16 | 0.47 | 0.55 |

That is the trade-off the RL agent is supposed to resolve.

## 3. Baselines match the paper

Both comparators hold DO at 2 mg/L. Means over the evaluation week, dry
weather:

| | paper | ours |
|---|---|---|
| PID aeration energy | 3698.2 | 3719.7 |
| fuzzy aeration energy | 3697.1 | 3719.9 |
| DO held at | 2.00 | 2.00 |

0.6 % apart, on a plant rebuilt from the model definition rather than from
their code. Good enough to trust relative comparisons.

Note on the energy definitions: our first implementation had the wrong
constants. BSM1 and this paper's eq. (20) define
`AE = (S_O_sat / 1800) * sum_i V_i KLa_i` and eq. (21)
`PE = 0.004 Q_a + 0.008 Q_r + 0.05 Q_w`. Both are now implemented as written.

## 4. One scaling decision the paper leaves open

Equation (23) is `-(b1 * AE + b2 * max(S_NH,e - 4, 0))` with `b1 = 1`,
`b2 = 0.42`. Taken literally, `AE` is of order 3500 kWh/d and would swamp the
ammonium term completely; the paper does not say what units it feeds the
reward. `AE` is affine in `KLa_5` (BSM1 fixes the other four transfer
coefficients), so we use the normalised aeration level `KLa_5 / 240` in
`[0, 1]`. That is the same quantity up to an affine transform and it makes the
published coefficients meaningful: a penalty of 1.0 for full aeration against
0.42 per mg/L of ammonium over the limit. Under this reading PID scores
`-0.63` and zero aeration scores `-2.48`, so the reward does prefer sensible
operation by a wide margin.

## 5. Why the reproduction is hard — three findings

These are the substance of the reproduction, and all three motivate the
modifications in `docs/03_novelty.md`.

### 5.1 The observation is not Markov

The state is `[S_S,5, S_O,5, S_NH,5]` and the action is an *increment* on
`KLa_5`. The actuator position is therefore a hidden integrator: the agent
cannot tell `KLa_5 = 0` from `KLa_5 = 20` once `S_O,5` has saturated near
zero, so it cannot know how far it has to move back. Any small bias in the
policy — or in the exploration noise — integrates the actuator to a rail and
the agent has no observation that says so.

The Ornstein–Uhlenbeck exploration the paper inherits from DDPG makes this
worse. With `theta = 0.15`, `sigma = 0.2` the noise is correlated over about
seven steps, so the *integrated* actuator excursion over a 13 440-step
training week is far larger than the 0–240 range. `KLa_5` is pinned at a rail
within the first few hundred steps.

Appending `KLa_5 / 240` to the observation restores the Markov property. It is
a one-line change and the first thing we would recommend to the authors.

### 5.2 The decision interval fights the plant's time constants

At 45 s per decision with `gamma = 0.99`, the agent's effective horizon is
about 100 steps — **75 minutes**. The effluent-ammonium response to an
aeration change takes several hours. So the reward that should punish cutting
the air arrives well outside the horizon the agent optimises, while the saving
in `AE` is immediate. Cutting aeration looks free.

Holding each action for 20 control periods (15 min) shortens the horizon in
*agent steps* by the same factor without changing the physics, and is how an
aeration set point is actually revised on a plant. This is a deviation from
the paper's stated protocol and we report it as one.

### 5.3 The value scale makes the critic slow, and the sign is wrong for weeks

With `gamma = 0.99` the undiscounted return is roughly 100× the per-step
reward, so the critic has to learn values near `-250` from rewards near
`-2.5`. Instrumenting the critic shows `dQ/da` has the **wrong sign** for the
first four simulated weeks of training and only becomes correct in the fifth —
and by then the tanh actor has saturated at its rail, where its gradient
nearly vanishes, so it escapes only slowly.

Two standard remedies, both now in `src/wwtp/rl/ddpg.py` and both documented
there: scale the reward by `1 - gamma` so the critic works on the reward
scale (this does not change the optimal policy), and put weight decay on the
actor so a saturated output can relax back.

### 5.4 What this means for the contribution

Paper 2's result — that letting DO float saves 5–7 % of aeration energy — is
sound and our plant reproduces the physics that makes it possible. What the
reproduction exposes is that **getting there by trial and error on the real
plant is the expensive part**: the reward is delayed by hours, the state is
incomplete, and the agent spends simulated weeks exploring at the actuator
rails, where a real plant would be discharging illegal effluent the whole
time.

That is exactly what prediction fixes, and it is why the modifications in
`docs/03_novelty.md` are the ones worth making:

* a **forecast input** tells the agent what load is coming, so it does not have
  to discover the consequence of its own actions by waiting hours for it;
* a **digital twin** supplies the thousands of extra transitions the critic
  needs without running them on the plant;
* a **distributional critic with a CVaR objective** targets the violation tail
  rather than the mean, which is what a discharge consent actually constrains;
* a **Lagrange multiplier** replaces the hand-tuned `0.38` and `0.42`, so the
  operator specifies an allowed violation rate instead of guessing a weight.
