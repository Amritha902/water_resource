# Status

Read this before continuing. It says what works, what the numbers are, and
what is left.

## Where things stand

| Piece | State |
|---|---|
| BSM1 plant (ASM1 biology, Takács clarifier, 5 tanks, influent generator) | **done**, validated against published BSM1 steady state |
| Paper 1 (MAACC) reproduced | **done** |
| Paper 1 — correction to the critic update in eq. (15) | **done**, reproduced by a test |
| Influent forecaster (TCN, quantiles + weather regime) | **trained** |
| Digital twin (GRU) | **trained** |
| Twin — correction for closed-loop identification bias | **done**, asserted by a test |
| Effluent-ammonium early warning | **done** (added in a parallel session) |
| Paper 2 — Figure 4 physics reproduced | **done** |
| Paper 2 — PID and fuzzy comparators | **done**, within 0.6% of published |
| Paper 2 — DDPG reproduced | **done**, after three corrections (below) |
| PANDA-RL (our modified network) | **implemented, learning** |
| Full benchmark across all weathers and seeds | **not run yet** |
| Figures | **not done** |
| `docs/05_results.md` | **not written** |

33 tests pass, 1 skipped.

## Reproducing paper 2 — the numbers

Dry weather, evaluation week, means. PID and fuzzy hold DO at 2 mg/L.

| | paper | ours |
|---|---|---|
| PID aeration energy | 3698.2 | 3719.7 |
| Fuzzy aeration energy | 3697.1 | 3719.9 |
| DDPG-B aeration energy | 3433 (−7.2%) | 3512 (−5.6%) |
| DDPG-B mean `S_O,5` | 1.032 | 1.074 |

So the central claim of paper 2 — let DO float and you save 5–7% of the
aeration energy — reproduces on an independently built plant. Effluent quality
moves the way they report too: our DDPG-B gives EQ 5530 against PID's 5411, a
slight increase, which is their finding that EQ and AE are coupled.

## Three corrections needed to get there

The published recipe does not learn in our reproduction — it collapses to zero
aeration and stays there. Each of these is documented in
`docs/04_second_paper.md` §5 and implemented in `src/wwtp/rl/`:

1. **The observation is not Markov.** State is `[S_S,5, S_O,5, S_NH,5]` but the
   action is an *increment* on `KLa_5`, so the actuator position is a hidden
   integrator — once DO saturates near zero the agent cannot tell `KLa_5 = 0`
   from `KLa_5 = 20`. The correlated OU exploration noise it inherits from
   DDPG pins the actuator at a rail within the first few hundred steps.
   *Fix:* append `KLa_5 / 240` to the observation, or use an absolute action.

2. **The decision interval fights the plant's time constants.** At 45 s with
   γ = 0.99 the horizon is ~75 min. Effluent ammonium responds to an aeration
   change over hours, so the penalty for cutting air falls outside the horizon
   while the energy saving is immediate — cutting aeration looks free.
   *Fix:* hold each action for 20 periods (15 min).

3. **The value scale makes the critic slow and its gradient wrong for weeks.**
   With γ = 0.99 the critic must learn values near −250 from rewards near
   −2.5. Instrumenting it shows `dQ/da` has the **wrong sign** for four
   simulated weeks; by then the tanh actor has saturated at a rail where its
   gradient nearly vanishes, and the evaluation freezes — we observed the exact
   same evaluation at episodes 4, 9 and 14.
   *Fix:* scale the reward by `1 − γ`, weight-decay the actor, and penalise the
   actor's pre-tanh output so a saturated policy can relax back.

With all three applied the agent converges in about 15 simulated weeks of
interaction. That is far more than the "7 days of online interactive learning"
the paper describes, and the gap is the honest motivation for the twin-assisted
updates in our own agent.

## PANDA-RL — what it changes

`src/wwtp/rl/panda_rl.py`. Four changes to the network, each aimed at one of
the problems above:

1. **Forecast input.** Actor and critic also see a 6-D context from the
   influent forecaster (predicted relative change in flow / ammonia / COD,
   forecast uncertainty, P(wet weather)). The policy becomes
   disturbance-scheduled instead of pure state feedback, so the agent does not
   have to wait hours to find out what its own action did.
2. **Distributional critic + CVaR.** 32 return quantiles, quantile Huber loss,
   and the actor maximises the mean of the worst 30% of returns instead of the
   expectation. A discharge consent constrains the tail, not the average.
3. **Constraints instead of guessed weights.** Reward keeps only
   `−KLa_5/240`. Each discharge limit gets its own cost critic and Lagrange
   multiplier, updated by dual ascent on the measured violation rate. The
   operator says "ammonia over 4 mg/L at most 5% of the time" instead of
   guessing paper 2's `0.38` and `0.42`. More than one limit can be active —
   paper 2's reward B had to drop total nitrogen to stay tractable.
4. **Twin-assisted updates.** Hook present, off by default, not yet evaluated.

It reduces exactly to DDPG with one quantile, CVaR α = 1, no constraints and
no forecast, so every component can be ablated. `experiments/06_aeration_benchmark.py`
has the ablations wired up (`--ablations`).

## Next steps, in order

1. Run `experiments/06_aeration_benchmark.py --ablations --seeds 3` across all
   three weathers. ~2 h. This is the headline table and it does not exist yet.
2. Write `docs/05_results.md` from it.
3. Figures: learning curves, a DO/`KLa_5` trajectory with the forecast overlaid,
   the AE/EQ trade-off, and the ablation bars. A validated colour palette is
   already in `src/wwtp/utils/style.py`.
4. Build the aeration twin (`AerationTwin`) and turn on `dyna_ratio` so the
   sample-efficiency claim in point 4 above can actually be tested. Without
   this the fourth modification is unproven.

## Honest caveats

* The plant is a faithful re-implementation, not the official BSM1 code, and
  the influent is statistically equivalent rather than identical. Absolute
  index values are not directly comparable with either paper's tables — every
  claim here is a comparison between methods on the *same* simulator.
* The forecaster is trained on synthetic influent from the same generator that
  produces the evaluation profiles. The three canonical profiles are held out,
  which is the strongest statement this repo can make; real-plant accuracy is
  unknown.
* DDPG on this problem is seed-sensitive. Every seed gets reported, not the
  best one.
* The paper-1 anticipatory feed-forward (`src/wwtp/panda/`) did **not** beat
  PID in the idealised tracking setting. That is written up as a negative
  result in `docs/02_reproduction.md` and `docs/03_novelty.md`; paper 2's
  framing is where the real headroom turned out to be.
