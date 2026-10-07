# Reproducing MAACC, and two things that had to be corrected

Everything here is in `src/wwtp/bsm1/`, `src/wwtp/baselines/` and
`src/wwtp/maacc/`, and is exercised by `tests/`.

## 1. The simulator

The official BSM1 distribution is not redistributable, so the plant is
re-implemented from the published model definition:

* `bsm1/asm1.py` — the 13-component / 8-process ASM1 matrix at 15 °C.
* `bsm1/settler.py` — the ten-layer Takács clarifier (double-exponential
  settling velocity, `v0 = 474`, `v0' = 250`, `r_h = 5.76e-4`,
  `r_p = 2.86e-3`, `f_ns = 2.28e-3`), solubles carried by the flow.
* `bsm1/plant.py` — five reactors (1000/1000/1333/1333/1333 m³), internal
  recycle, return sludge `Q_r = 18 446`, waste `Q_w = 385 m³/d`, initialised
  at the published open-loop steady state.
* `bsm1/influent.py` — dry / rain / storm influent generated from the
  documented BSM1 influent characteristics (18 446 m³/d average, two-peak
  diurnal pattern, weekday/weekend modulation, a sustained rain event, two
  first-flush storm events).

`tests/test_bsm1.py` checks the open-loop steady state against the published
BSM1 values (`S_O,5 ≈ 0.49`, `S_NO,2 ≈ 3.66`, effluent TSS ≈ 12.5 g/m³), that
states stay non-negative, that the settler thickens by more than 50×, and
that both control channels have the correct gain sign.

**Caveat, stated plainly.** This is a faithful re-implementation, not the
official BSM1 code, and the influent files are statistically equivalent
rather than identical. Absolute index values are therefore *not* directly
comparable with Table II of the paper. Every claim in this repository is a
comparison between controllers run on the *same* simulator, which is the
comparison that is actually meaningful.

## 2. Implementation choices

| # | Choice | Why |
|---|---|---|
| 1 | Network inputs and the `Y_i`/`R_i` utility terms are normalised by the actuator ranges. | `KLa_5` spans 0-240 and `Q_a` spans 0-92 230. The paper keeps raw units and absorbs the scale into the weight matrices; normalising is equivalent and lets both agents share one learning rate. |
| 2 | The actor gradient uses `dQ/d(Delta a) + dQ/da`. | `a_k = a_{k-1} + Delta a_k`, so both critic inputs depend on the action-network output. Eq. (27) of the paper only carries the second path. |
| 3 | A constant bias input is appended to both networks. | A `tanh` network without bias is an odd function pinned to zero at the origin, so it cannot represent a Q-function that is strictly positive at the set point. |
| 4 | Weight decay on the action network's outer weights. | A saturated `tanh` output has vanishing gradient. Without decay, the start-up transient latches the policy at its rail permanently — observed directly before the fix. |

## 3. Correction: the critic update of eq. (15) is expansive

Equation (15) defines the approximation error

```
e_k = U_k + lambda Q(x_k) - Q(x_{k-1})
```

and eq. (18) descends `0.5 e_k^2` in `beta_k`, i.e. in the weights that
produce `Q(x_k)`. That corrects the **later** prediction, pulling `Q(x_k)`
towards `(Q(x_{k-1}) - U_k) / lambda`.

Write the resulting recursion for a slowly varying state, where
`x_k ≈ x_{k-1}`:

```
Q_k  <-  Q_{k-1} (1 + l_c (1 - lambda) ||vartheta||^2)  -  l_c ||vartheta||^2 U_k
```

The homogeneous gain is **greater than one** for every admissible learning
rate, so the critic diverges. This is not an artefact of my hyper-parameters:
`tests/test_critic_update.py` reproduces it on a synthetic problem whose true
Q-function is known, and the learned `Q(0)` exceeds 10⁴ instead of the
correct 0.9. A WWTP is precisely the slowly varying regime where this bites.

Theorem 1 is not contradicted. It bounds the weight approximation error
(UUB); a bound can be large, and the condition
`l_c < 1 / (lambda^2 ||vartheta||^2)` is exactly the no-overshoot condition
for that step, not a contraction condition for the recursion.

**The fix** keeps the paper's Bellman equation (9) and changes only the
direction of the descent: correct `Q(x_{k-1})` towards `U_k + lambda Q(x_k)`,
which is the standard TD(0) direction. On the same synthetic problem this
converges to the Bellman fixed point with the correct sign of `dQ/da`, which
is what the actor needs — with the wrong sign the actor integrates a biased
gradient straight into its saturation limit. Both variants are kept in the
test file so the difference is reproducible.

## 4. What the reproduction actually shows

With the fixes in place, MAACC behaves as the paper describes qualitatively:
it starts exactly at the expert prior, the learned correction stays bounded,
and training is stable in most seeds but not all — consistent with the
paper's own "above 90 % success rate".

Quantitatively, the full benchmark (`experiments/03_run_control.py`, 54 runs,
tabulated in `docs/05_results.md`) splits three ways, and two of its readings
corrected what was written here earlier.

**Under the paper's own noise protocol, MAACC works.** With 3 %-of-range
actuator noise plus DO and nitrate sensor noise, MAACC improves the nitrate
loop's IAE by 29 % on dry weather and 23 % on rain against the PID prior, and
lowers `DEVmax` on that loop as well. An earlier draft of this file said
MAACC "does not reliably beat" a well-tuned PID. That was based on spot
checks that happened not to cover this condition, and it was too strong.

**In the clean condition it is worse on nitrate, and that is not interesting.**
PID's IAE there is 0.0021 against MAACC's 0.0032 — both negligible beside the
0.06–0.13 of the noisy runs. With no disturbance to reject there is nothing to
learn and exploration is pure cost.

**The de-tuned prior does not help, which refutes a hypothesis stated here.**
The earlier argument ran: the prior is an *incremental* PID, so it has
integral action, and it absorbs any slowly varying additive correction the
actor learns within a few control periods — therefore weakening the prior
should give the learned term room. It does not. With the prior at 25 % of its
published gains, MAACC is worse than PID on every weather and both loops, by
6 % to 35 %. A weaker prior leaves a larger standing error for the actor to
chase, and chasing it with an incremental policy turns out to be harder than
leaving it alone. The hypothesis is withdrawn.

What survives is the narrower observation, which still motivates the
extension: a learned correction driven by tracking error alone has very
little to work with when the prior is good, and the condition where it does
earn its place is the one with real measurement noise — exactly where
*anticipating* a disturbance should help more than reacting to it.
