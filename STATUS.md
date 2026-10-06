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
| Effluent-ammonium early warning | **code done** (added in a parallel session); its network is still untrained, so `test_trained_warning_responds_to_the_analyser` skips. Training was started and stopped at epoch 7/12 to free CPU for the forecast-vs-CVaR split, which was the higher-value run. To finish: `python experiments/04_collect_warning_data.py && python experiments/05_warning_benchmark.py` (~45 min on an idle box). |
| Paper 2 — Figure 4 physics reproduced | **done** |
| Paper 2 — PID and fuzzy comparators | **done**, within 0.6% of published |
| Paper 2 — DDPG reproduced | **done**, after three corrections (below) |
| PANDA-RL (our modified network) | **implemented and evaluated** |
| Aeration twin (for Dyna) | **trained** — DO MAE 0.114 mg/L, effluent NH MAE 0.052 mg/L per 15-min step, and it recovered the Figure-4 trade-off from data |
| Benchmark, dry weather (all 5 methods, 2 seeds) | **done** |
| Benchmark, rain and storm | **partial** — the run was killed at job 13/24 by background-task cleanup; rain has PID/fuzzy/DDPG-A, storm not started |
| Energy/violation frontier | **running** |
| Dyna sample-efficiency ablation | **script written, not yet run** |
| Figures | **not done** |
| `docs/05_results.md` | **generated** from the benchmark output |

57 tests pass, 1 skipped (34 of them in `tests/test_rl.py`).

## Reproducing paper 2 — the numbers

Dry weather, evaluation week, means. PID and fuzzy hold DO at 2 mg/L.

| | paper | ours |
|---|---|---|
| PID aeration energy | 3698.2 | 3719.7 |
| Fuzzy aeration energy | 3697.1 | 3719.9 |
| DDPG-B aeration energy | 3433 (−7.2%) | 3429–3504 (−5.8 to −7.8%) |
| DDPG-B mean `S_O,5` | 1.032 | 1.10–1.16 |

The comparators are 0.6% apart and the central claim reproduces: let DO float
and 5–8% of the aeration energy is available.

## But the saving is not free, and that is the real result

The benchmark also records what the paper's tables do not: how often the
effluent is out of consent.

| dry weather | AE | saving vs PID | time above 4 g N/m³ |
|---|---|---|---|
| PID (fixed 2 mg/L) | 3719.7 | — | **12.2%** |
| DDPG-B seed 0 | 3429.5 | +7.8% | **45.9%** |
| DDPG-B seed 1 | 3503.8 | +5.8% | 30.4% |
| PANDA-RL seed 0 (5% budget) | 3597.9 | +3.3% | 23.1% |
| PANDA-RL seed 1 (5% budget) | 3754.4 | −0.9% | 14.0% |

DDPG-B buys its energy by exceeding the ammonium limit three to four times as
often as PID. That is not a reproduction artefact — it is what eq. (23) asks
for: at `β₂ = 0.42` per mg/L of excess against a full-aeration penalty of 1.0,
a long run of small exceedances is cheaper than the air it saves.

PANDA-RL at a 5% budget halves the violation rate and saves correspondingly
less, and it does **not** reach 5% — the multiplier was still climbing at
12–14 of a 30 cap. That is informative, not broken: PID itself violates 12% of
the time, so a 5% budget needs *more* aeration than PID and there is no energy
to save there at all.

So a single number is the wrong deliverable. The question a constrained
formulation can answer, and a fixed penalty weight cannot, is: **at a violation
rate the operator will accept, how much energy is actually available?**
`experiments/11_budget_frontier.py` sweeps the budget and traces that curve.
First point in: at a 5% budget the agent settles at 12.3% violations — PID's
own rate — with aeration energy 3664.5, i.e. **1.5% cheaper than PID at
matched effluent risk**. The rest of the curve is in `docs/05_results.md`.

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
4. **Twin-assisted updates (Dyna).** Evaluated, with a negative result on the
   fix. At equal real interactions the model-based arm cuts aeration from
   3749 to 3205 kWh/d — real sample efficiency — but takes the ammonium
   violation rate from 14.1% to 66.8%, because four times the gradient steps
   are not matched by four times the dual steps. The intra-episode dual was
   meant to correct that and **did not** (68.0%): its 268 steps of 0.05 give
   a smaller total movement budget than the episode dual's 8 steps of 2.0,
   and the rate it reacts to is built from real transitions only. Scaling the
   dual step with `dyna_ratio` is the proposed correction and is untested.
   See `docs/03_novelty.md` §4.

It reduces exactly to DDPG with one quantile, CVaR α = 1, no constraints and
no forecast, so every component can be ablated. `experiments/06_aeration_benchmark.py`
has the ablations wired up (`--ablations`).

## Next steps, in order

1. Finish the frontier sweep (running) and regenerate `docs/05_results.md`.
2. Re-run rain and storm for DDPG-B and PANDA-RL, in chunks small enough to
   survive. Use `--weathers rain storm` with a separate `--tag` and let
   `09_write_results.py` merge; the traces and curves of the killed run are
   gone but every summary row was written incrementally.
3. Run `experiments/10_dyna_ablation.py` for the three-arm sample-efficiency
   table.
4. Figures: `experiments/07_figures.py` (add `--dose-response` for the slow
   Figure-4 reproduction).
5. Full test suite.

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
  PID in the idealised tracking setting. It is written up as a negative result
  in `docs/06_tracking_layer_feedforward.md`; paper 2's framing is where the
  real headroom turned out to be.
