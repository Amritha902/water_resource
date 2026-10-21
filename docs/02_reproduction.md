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
Q_k  <-  Q_{k-1} (1 + l_c lambda (1 - lambda) ||vartheta||^2)  -  l_c lambda ||vartheta||^2 U_k
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

**The action network had to be made faithful to eq. (8) first.** An earlier
version gave the actor a bias input and integrated `a` with a leak of 0.999.
With the bias, the increment at zero error, `π(0)`, is not zero, so `a`
drifted steadily into its clip and sat there for 92 % of the run, acting as
a constant offset — a learning-free PID plus that same constant reproduced
its "gain" to the third decimal. Eq. (8) is pure integration, and an
incremental policy must emit nothing at zero error; both are now the
default (`actor_bias=False`, `leak=1.0`) and the second is a test.

**With the faithful actor, MAACC works where the paper says it does**
(`docs/REPRODUCTION.md` §2.3, 54 runs, 2 seeds). Under the paper's
supplementary noise protocol it improves nitrate IAE over the PID prior by
27 % (dry), 21 % (rain) and 10 % (storm), with `DEVmax` lower on dry and
rain, and oxygen within 0.3 % (dry, rain) and 6 % (storm). In the clean
condition it changes nothing (±0.5 %), as it should when there is nothing to
reject. The learned term now moves with the tracking error instead of
sitting at its clip.

**What it learns, measured against learning-free controls**
(`experiments/17_maacc_offset_control.py`). IAE `S_NO,2`, noisy, mean of 2
seeds:

| | PID | MAACC | PID + constant −9 223 | PID, integrator clamp −9 223 |
|---|---|---|---|---|
| dry | 0.0832 | 0.0607 | 0.0590 | 0.0604 |
| rain | 0.0797 | 0.0627 | 0.0614 | 0.0591 |
| storm | 0.1214 | 0.1092 | 0.1110 | 0.1018 |

Part of the gain on the recycle loop is a slowly varying negative correction
that compensates the prior's anti-windup at zero recycle (the PID sits at
zero recycle ~12 % of the time under noise); an anti-windup-tuned prior
gets most of that without learning. MAACC beats the constant offset on
storm, where its correction varies with the state, but not the tuned prior.

**The de-tuned prior is now clearly worse** (our stress test, not the
paper's): oxygen IAE more than doubles. With a weak prior the standing error
drives the purely integrating aeration actor into its clip. Both earlier
withdrawals stand: MAACC does beat a well-tuned PID under noise, and a
de-tuned prior does not give the learned term room.

What survives is the narrower observation, which still motivates the
extension: a learned correction driven by tracking error alone has very
little to work with when the prior is good, and the condition where it does
earn its place is the one with real measurement noise — exactly where
*anticipating* a disturbance should help more than reacting to it.
