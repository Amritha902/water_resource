# Status — where this repository stands

Last updated at the end of the first working session. Read this before
picking the work back up.

## Done and tested

| Piece | State | Evidence |
|---|---|---|
| ASM1 + Takács clarifier + 5-reactor BSM1 plant | **done** | `tests/test_bsm1.py` — open-loop steady state matches published BSM1 (`S_O,5 ≈ 0.49`, `S_NO,2 ≈ 3.66`), both control gains have the right sign, settler thickens >50× |
| Dry / rain / storm influent generator | **done** | mean flow 17.5–18.3 k m³/d, COD load ≈ 6.8 t/d, storm peak 38 k m³/d — all in the documented BSM1 range |
| Incremental PID expert prior (paper's gains) | **done** | `tests/test_control_loop.py` |
| MAACC reproduction | **done** | `tests/test_control_loop.py` — starts exactly at the prior (`a_0 = 0`), learned term stays bounded |
| **Correction: eq. (15) critic update diverges** | **done** | `tests/test_critic_update.py` — the literal update blows up to >10⁴ on a problem whose true `Q(0) = 0.9`; the TD(0) direction converges. Written up in `docs/02_reproduction.md` |
| Probabilistic influent forecaster | **trained** | 99 200 windows, validation pinball 0.0418, weather-regime accuracy **95.3 %**; `artifacts/forecaster.pt` |
| Neural digital twin | **trained** | free-running RMSE **0.28 mg/L** over a 4.8 h horizon; `artifacts/twin.pt` |
| **Correction: closed-loop identification bias in the twin** | **done** | first twin had a near-zero aeration gain (40 → 220 1/d moved DO by 0.05 mg/L). Refit with 60 % open-loop PRBS excitation; now 20 → 240 1/d moves DO 0.24 → 1.69 mg/L and the cross-couplings have the right signs. Asserted in `tests/test_learned_models.py` |
| PANDA controller (all four components) | **implemented, runs end to end** | `src/wwtp/panda/controller.py` |
| Docs: paper summary, reproduction notes, contribution | **done** | `docs/01`–`docs/03` |

28 tests pass.

## Not done

1. **The full benchmark sweep** (`experiments/03_run_control.py`) has never
   been run to completion — only spot checks on single scenarios.
2. **Figures** — `src/wwtp/utils/style.py` holds a validated colour palette
   (three-slot categorical, checked against the colour-vision and contrast
   gates), but no figure script has been written.
3. **`docs/04_results.md`** — does not exist yet; it needs the sweep.
4. **Realistic-instrumentation operating condition** — see below. This was
   the next step when the session ended and the code change is *not* in the
   repository.

## What the numbers currently say — read this honestly

Spot checks on the canonical storm profile with the paper's noise protocol
(3 %-of-range actuator noise) plus DO/nitrate sensor noise, 14 days,
burn-in 1 day:

| controller | IAE `S_O,5` | IAE `S_NO,2` | DEVmax `S_NO,2` |
|---|---|---|---|
| PID (expert prior) | 0.0335 | 0.0624 | 0.635 |
| MAACC (reproduction) | 0.0338 | 0.0649 | 0.787 |
| PANDA (current form) | 0.0357 | 0.0666 | 0.629 |

**Neither MAACC nor PANDA currently beats a well-tuned PID on this
simulator.** That is the honest state and it must not be written up as
anything else. Two findings explain it, and both are worth keeping:

1. **MAACC ≈ PID is structural, not a tuning failure.** The expert prior is
   an *incremental* PID, so it has integral action, and it absorbs any
   slowly varying additive correction the actor learns within a few control
   periods. The tracking error therefore carries almost no information about
   `a`, and the only utility term that still depends on `a` is the effort
   penalty, whose minimiser is `a = 0`. The learned term is close to
   unidentifiable from tracking error alone. (With a *de-tuned* prior — 25 %
   of the published gains — MAACC does recover a few percent, ~4 % on both
   IAE channels in a spot check, but seed variance is comparable to the
   effect.)

2. **There is no headroom left to win in the idealised setting.** With
   instantaneous actuation and clean sensors the PID's `IAE_SO5` is dominated
   by white actuator noise, which *no* predictor can help with, and the
   effluent quality index barely moves between weather conditions
   (EQ 5029 storm vs 4770 dry, zero ammonium violations). Anticipation has
   nothing to buy.

## The next concrete step

Add a **realistic-instrumentation** operating condition to
`run_closed_loop` and make it the headline benchmark:

* `slew_rate` — max actuator change per 45 s period (blowers and recycle
  pumps cannot step; suggested `[6 1/d, 2200 m³/d]`, i.e. full travel in
  ~20–30 min);
* `sensor_tau_seconds` — first-order probe lag (DO ≈ 120 s);
* `analyzer_period_seconds` — zero-order hold on the nitrate reading
  (sequential analysers have 5–15 min cycle times, so `S_NO,2` is *stale*
  between cycles).

This is the regime where feedback provably cannot keep up and a forecast can,
and it is what real plants have. The rationale is already written into the
`docs/`; the harness change was drafted and not applied, so
`src/wwtp/envs.py` is at its pre-change state.

Expected shape of the result, to be confirmed rather than assumed: PID
degrades sharply (rate-limited response to storm onset), PANDA degrades much
less because the feed-forward starts moving before the rate limit binds. If
that does **not** materialise, report it — the two structural findings above
stand on their own and are a legitimate contribution.

## Two things already established that should survive any rewrite

Both are corrections to published/standard practice, both are reproduced by
tests, and both cost real debugging time:

* **`docs/02_reproduction.md` §3** — equation (15) of the base paper descends
  the Bellman residual in the direction that corrects the *later* prediction.
  For slowly varying states the recursion has homogeneous gain
  `1 + l_c (1 − λ)‖ϑ‖² > 1` and diverges. Theorem 1 is not contradicted: it
  bounds the weight error (UUB), and a bound can be large.
* **`docs/03_novelty.md` §3.2** — a digital twin identified on closed-loop
  (PID-controlled) data learns a near-zero aeration gain, because the
  controller raises `KLa_5` exactly when demand rises. Differentiating such a
  twin moves the actuator the wrong way. Open-loop PRBS excitation is not
  optional.
