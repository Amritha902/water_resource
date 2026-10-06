# The contribution: PANDA-RL

Short version: **paper 2's agent can only react to what has already happened
to the plant, it is told how to weigh energy against discharge limits by two
hand-tuned numbers, and it optimises the average outcome. We give it a
forecast, let it discover the weights itself from a violation budget, and make
it optimise the bad tail instead of the average.**

Everything here is in `src/wwtp/rl/panda_rl.py` and is tested in
`tests/test_rl.py`.

## Why these four changes and not others

The modifications are not a wish list. Each one is aimed at a specific problem
that reproducing paper 2 actually exposed (`docs/04_second_paper.md` §5):

| problem found in reproduction | modification |
|---|---|
| the reward that should punish under-aeration arrives hours after the action, so cutting air looks free | **forecast input** — the agent is told what load is coming instead of discovering it by waiting |
| `alpha2 = 0.38` and `beta2 = 0.42` are set "by trial and error", and reward B had to drop total nitrogen entirely to stay tractable | **Lagrange multipliers** — the operator states a violation budget, the weights are learned |
| a scalar critic maximises the mean return, but a discharge consent is about the tail | **distributional critic + CVaR** |
| the agent needs ~15 simulated weeks of interaction; the paper allows 7 days | **twin-assisted updates** (implemented, not yet evaluated) |

The control law keeps paper 1's safety idea intact: the policy's final layer
is initialised small so the untrained agent holds the actuator where it is,
which is the same safeguard paper 1 gets by zeroing its action-network outer
weights (its eqs. 7–8).

## 1. Forecast-conditioned actor and critic

A separate network (`src/wwtp/forecast/`) reads 24 h of inlet history — flow,
ammonium and readily biodegradable COD, the three measurements a real plant
already has at the inlet — and predicts the next 2 h. It is a dilated causal
TCN with two heads:

- a **quantile head** giving the 10/50/90 % quantiles of every channel at each
  of eight 15-minute horizons, trained with the pinball loss, monotone by
  construction;
- a **weather-regime head** classifying dry / rain / storm, which doubles as an
  interpretable storm alarm for the operator.

Both the actor and the critic additionally receive

```
c_k = [ dQ/Q at 30 min, dQ/Q at 2 h, dNH/NH at 2 h, dCOD/COD at 2 h,
        relative flow uncertainty, P(rain) + P(storm) ]
```

so the policy is `mu(s_k, c_k)` — a *disturbance-scheduled* law rather than
pure state feedback. It can cut aeration before the load falls and raise it
before the load arrives, instead of finding out hours later.

Trained on 160 randomised 14-day influent realisations. The three canonical
BSM1 profiles are held out, so every control result is an out-of-sample test
of the forecaster. Validation weather-regime accuracy is 95%.

## 2. Distributional critic with a CVaR objective

The critic outputs 32 quantiles of the return distribution `Z(s, a)` instead
of a single expected value, trained with the quantile Huber loss against the
distributional Bellman target `r + gamma Z'(s', mu'(s'))`. The actor then
maximises

```
CVaR_alpha[ Z(s, mu(s)) ]      alpha = 0.3
```

the mean of the worst 30% of returns, rather than the expectation.

Why this matters here and not everywhere: a discharge consent is a limit on
*occurrences*, not on an average. A policy that maximises the mean will
happily accept a rare large exceedance if it buys enough energy on the typical
day. Optimising the tail does not. The expectation is recovered exactly at
`alpha = 1`, which is the `PANDA-noCVaR` ablation.

## 3. Discharge limits as constraints, not guessed weights

The reward keeps only the term that is genuinely an objective:

```
r_k = - KLa_5 / 240
```

(`AE` is affine in `KLa_5` because BSM1 fixes the other four transfer
coefficients, so this *is* aeration energy up to an affine transform.)

Each discharge limit becomes a constraint with its own cost critic `Q_c` and
multiplier `lambda`, and the actor maximises

```
CVaR[Z(s, mu(s))]  -  sum_k lambda_k Q_c,k(s, mu(s))
```

Each multiplier is updated by dual ascent on the measured violation rate at
the end of every training week:

```
lambda  <-  clip( lambda + eta (rate - budget),  0,  lambda_max )
```

Because the cost critic is scaled by `1 - gamma`, its output reads directly as
a violation rate, which keeps `lambda` interpretable. The operator specifies
*"ammonium above 4 g N/m³ at most 5% of the time, total nitrogen above
18 g N/m³ at most 15%"* and the trade-off weights follow. Two consequences:

- the `0.38` and `0.42` of eqs. (19) and (23) disappear;
- several limits can be active at once, where paper 2's reward B had to drop
  the total-nitrogen term because a single scalar reward could not carry both.

We can watch the mechanism work. On dry weather the agent first goes after
energy hard and lands at aeration energy 3289 with mean effluent ammonium
4.68 g N/m³ — infeasible — and the multiplier then climbs until it pulls the
policy back inside. Sweeping the budget traces an energy/violation frontier
(`experiments/11_budget_frontier.py`); at a 40 % budget the agent lands at
40.2 %, i.e. it hits what it was asked for, and the resulting frontier passes
below both PID and the reproduced DDPG-B. The numbers are in
`docs/05_results.md` and are regenerated from the stored results rather than
quoted here, so they cannot go stale.

Two honest qualifications, both also in the results document. At a 5 % budget
the agent **cannot** reach its target: PID itself exceeds the ammonium limit
12 % of the time at a fixed 2 mg/L, so a 5 % budget needs *more* aeration
than PID and there is no energy to save there at all. And at one seed and
twenty training weeks the budget does not reliably index the operating point —
one of the four sweep runs is dominated by another, which is why the
comparison is also run against a *stationary* objective in
`experiments/12_weight_sweep.py`.

## 4. Twin-assisted (Dyna) updates

`wwtp.rl.twin.AerationTwin` is a one-step model of the aeration MDP, fitted
on 16 128 open-loop transitions and accurate to 0.114 mg/L on dissolved
oxygen and 0.052 mg/L on effluent ammonium per 15-minute step. It recovered
the Figure-4 trade-off from data rather than being told it — more air, more
oxygen, less effluent ammonium, more total nitrogen — and a test asserts all
three signs. With `dyna_ratio > 0` it supplies synthetic batches alongside
the real ones, each carrying its own reward and constraint costs.

It does buy sample efficiency, and that is also how it bites. At the same
number of *real* interactions the model-based arm reached aeration energy
3004 against 3272 for the model-free arm — but it ran effluent ammonium to
8.0 g N/m³ against 4.8, because three synthetic batches per real one gives
the policy roughly four times the gradient steps while the Lagrange
multiplier was still being stepped once per episode. The policy outruns the
constraint. Fixed with an intra-episode dual (`lambda_every`), which is a
general point about constrained model-based RL rather than a quirk of this
plant: accelerating the policy without accelerating the dual turns a
constrained problem into an unconstrained one for as long as the lag lasts.

One hard lesson from building the existing twin is worth repeating, because it
applies to any model fitted here: a twin identified on data collected **under
feedback** learns a near-zero aeration gain. With the PID in the loop, `KLa_5`
rises exactly when the oxygen demand rises, so aeration and dissolved oxygen
are almost uncorrelated in the data. Our first twin changed its predicted
`S_O,5` by 0.05 mg/L between `KLa_5 = 40` and `KLa_5 = 220`, where the real
plant moves by several mg/L; differentiating it moved the actuator the wrong
way. The fix is 60% open-loop PRBS excitation, and the gain signs are now
asserted in `tests/test_learned_models.py`.

## Everything is ablatable

`PandaRLAgent` reduces exactly to the paper's DDPG with `n_quantiles = 1`,
`cvar_alpha = 1`, no constraints and no forecast — asserted by
`test_panda_reduces_to_ddpg_when_every_component_is_ablated`.
`experiments/06_aeration_benchmark.py --ablations` runs the three
single-component removals.

## Honest limitations

- The forecaster is trained on synthetic influent from the same generator that
  produces the evaluation profiles. Held-out profiles are the strongest claim
  this repo can make; accuracy on real plant data is unknown.
- No stability theorem is offered. Paper 1's Theorem 1 covers its own shallow
  networks; nothing here extends it to a deep distributional critic.
- The fourth modification is not yet evaluated (above).
- DDPG on this problem is seed-sensitive, so every seed is reported rather
  than the best one.

## The other thing we tried, which did not work

`docs/06_tracking_layer_feedforward.md` describes an anticipatory feed-forward
term added to paper 1's *tracking* controller, where the twin is
differentiated through the forecast to pre-empt disturbances. It did **not**
beat a well-tuned PID. The write-up explains why — with an incremental PID
prior in the loop, integral action absorbs any slowly varying learned
correction, and in the idealised tracking setting the remaining error is
dominated by actuator noise that no predictor can help with. That negative
result is what pushed the work to paper 2's framing, where the headroom is
real.
