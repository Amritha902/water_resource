# Viva preparation — questions to expect, with precise answers

Short answer first, then the detail, then where to point: the website
(`docs/index.html` → Paper 1 / Paper 2 / Prediction models) or the code.
Numbers are the ones the site shows; the site reads them from `results/`.

---

## 0. Say these first, before being asked

1. **Two base papers are reproduced on one BSM1 plant built from scratch.**
   Paper 1 (Wang et al., IEEE TII 2026) — MAACC, a tracking controller.
   Paper 2 (Du et al., Sci China Tech Sci 2023) — DDPG that drops the DO set
   point to save aeration energy. Only *relative* claims are made: the plant
   is a re-implementation and the influent is statistically equivalent.
2. **Paper 1's eq. (15) critic update diverges as printed**; TD(0) on the same
   residual fixes it (test + Monte Carlo ground truth).
3. **Paper 2's recipe does not learn as published**; three corrections were
   needed, and its energy saving comes with **3× more ammonium violations**
   than PID — a number the paper does not report.
4. **Results that failed, and are kept:** the paper-1 forecast feed-forward
   (worse than PID on oxygen); the forecaster's weather-regime head (chance on
   fresh data — the 95.8 % was validation leakage); Dyna (breaks the
   constraint).
5. **MAACC's nitrate gain under noise is not learned**: a constant offset
   (its learned term frozen at the clip) reproduces it — anti-windup on the
   PID prior's recycle loop (A7b).
6. **Everything committed is reproducible:** re-simulating the paper-1
   seed-0 runs gives the committed numbers to 4 decimals; re-training
   DDPG-B (seed 0) gives AE 3429.5 kWh/d and 45.9 % violations, identical to
   the committed run — *at the benchmark's 2 CPU threads*. With 1 thread the
   first training week still matches but the run then diverges (float
   summation order) and lands at 3526.8 kWh/d / 27.3 %, on the same
   trade-off. Know this: "is your RL result reproducible?" → yes, bit-for-bit
   given seed *and* thread count; across thread counts, only statistically.

---

## A. Paper 1 — MAACC

**A1. One sentence?** Two coupled loops (air `KLa_5` → oxygen `S_O,5`;
recycle `Q_a` → nitrate `S_NO,2`), one adaptive-critic agent per loop, each
adding a learned correction to an industrial PID (`u = b + a`) and charged for
the tracking error of *both* loops so it accounts for the coupling it cannot
model.

**A2. Expert knowledge?** The incremental PID prior, `Kp = diag{100, 1e5}`,
`Ki = diag{20, 3e4}`, `Kd = diag{10, 1e3}`. Actor outer weights start at zero
so `a_0 = 0`: day one *is* the PID. `pid.py:50`, `agent.py:109`, test
`test_maacc_starts_from_the_expert_prior`.

**A3. Why increments?** `a_k = a_{k-1} + Δa_k` (eq. 8): a disturbance corrupts
one increment, not the whole input.

**A4. The coupling `G_i(s_k)`?** Never identified. Utility (eq. 10) has
`zᵀ P_i z` with the **full** error vector. `agent.py:139`.

**A5. Theorem 1?** UUB of the *weight* errors under
`l_c < 1/(λ²‖ϑ‖²)`, `l_a < 1/(‖δ‖²(β^M C)²)`. Not closed-loop stability of the
tracking error — that is inherited from the stable prior.

**A6. The eq. (15) problem, precisely.**
`e_k = U_k + λQ(x_k) − Q(x_{k-1})`; eq. (18) descends ½e² in β where
`Q(x_k) = βᵀϑ`, gradient `e·λ·ϑ`, so `Q_k ← Q_k − l_c λ‖ϑ‖² e`. Slowly varying
state, `Q_k ≈ Q_{k-1} = Q`, `e = U − (1−λ)Q`:

```
Q ← Q (1 + l_c λ (1−λ) ‖ϑ‖²) − l_c λ ‖ϑ‖² U
```

Multiplier > 1 for every `l_c > 0` → geometric growth. TD(0) instead corrects
`Q(x_{k-1})` towards `U_k + λQ(x_k)`: a λ-contraction. Evidence: same network,
data, learning rate; true `Q(0) = 0.943` (Monte Carlo, 200 000 paths); eq. (15)
ends near −13 000, TD(0) at 1.10 (the 16 % gap is the 24-unit random-feature
approximation). **Not a contradiction of Theorem 1**: UUB allows a huge bound;
the paper's `l_c` condition is the one-step no-overshoot condition (the
residual changes by `−l_c λ²‖ϑ‖² e` per step), not a contraction over time.

**A7. Results (committed, 54 runs, 2 seeds).** Under the paper's noise
protocol (3 % of each actuator's range + sensor noise): MAACC improves nitrate
IAE by **+28.9 % dry, +22.8 % rain**, −8.0 % storm; oxygen is 2–7 % worse.
Clean condition: MAACC looks −45…−69 % on nitrate, but in absolute terms
0.002 vs 0.003 — negligible; with nothing to reject, exploration is pure cost.
De-tuned prior (25 % gains): MAACC worse everywhere (−6 % to −35 %).

**A7b. Is MAACC's nitrate gain learned control?** **No — and say this
first.** Its learned recycle term `a_2` sits at its clip (−9 223 m³/d) most of
the run. A learning-free PID plus that constant offset gives IAE `S_NO,2`
0.0590 (dry) / 0.0614 (rain) vs MAACC 0.0591 / 0.0616 and PID 0.0832 / 0.0797
(mean of 2 seeds); on storm the offset PID gets 0.1110 where MAACC is *worse*
than PID (0.1311 vs 0.1214). Mechanism: the PID's recycle sits at zero ~12 % of the time under
noise; its integrator clamped at exactly 0 leaves zero recycle too early.
Letting the integrator go to −9 223 (plant still clips at 0) gives 0.0604 /
0.0591 / 0.1018 (dry/rain/storm) — as good as MAACC on dry, better on rain
and storm. Lowering only the upper limit
does not. Script: `experiments/17_maacc_offset_control.py`. What it means:
the noise protocol exposes the prior's anti-windup, MAACC compensates by
accident; this benchmark cannot show the critic learned anything useful.
It does not by itself refute the paper's method.

**A8. "You wrote earlier that MAACC never beats PID."** Withdrawn. It came
from spot checks that did not include the noise protocol; the full benchmark
showed it does on the nitrate loop. Also withdrawn: the prediction that a
de-tuned prior gives the learned term room — the data says the opposite.

**A9. Ten deviations from the paper** (Paper 1 page §9): TD(0) critic; full
actor Jacobian `∂Q/∂Δa + ∂Q/∂a`; bias input; normalised inputs; leaky
integration `0.999·a` and clip ±10 % of range; `Δa = Δa_max·tanh(·)`
(4 1/d, 1500 m³/d per step); actor weight decay 2.0 + gradient clip 10;
our own `W=1, P=0.25, Y=0.02, R=0.05, λ=0.9`, 16/12 hidden; 1-day PID
warm-up; re-implemented BSM1.

**A10. Not reproduced:** Table II absolute numbers; the 7 comparison methods
(RSCMPC, MADRL-PID, SUP-HDP, dHDP, DPPGADP, ACD-VarInf, FAACC); the ">90 %
success over 20 runs".

**A11. Your extension on paper 1 (PANDA, `u = b + g·a + f`)?** Forecast
context into the agents, an uncertainty gate on the learned term, and an
anticipatory feed-forward solved by differentiating a GRU twin over 48 min —
closed-loop aware (a differentiable PID inside the rollout) and offset-free
(two rollouts, forecast vs frozen influent, only their difference
penalised). **Result: worse than PID on oxygen in every condition**, better on
nitrate under noise (+6 to +16 %). Kept as a negative result
(`docs/06_tracking_layer_feedforward.md`). Why: under the idealised and noise
protocols the PID's oxygen error is dominated by white actuator noise that no
predictor can remove; the feed-forward adds model error on top.

---

## B. Paper 2 — DDPG aeration

**B1. One sentence?** Instead of holding DO at 2 mg/L, a DDPG agent sets
`KLa_5` directly and lets DO float, trading aeration energy against effluent
quality.

**B2. Specification.** State `[S_S,5, S_O,5, S_NH,5]`; action an increment on
`KLa_5`; reward A (eq. 19) `−(max(S_NH,e−4,0) + 0.38·max(N_tot,e−18,0))`;
reward B (eq. 23) `−(AE + 0.42·max(S_NH,e−4,0))`; actor 3-256-256-1, critic
4-256-256-1; γ 0.99, batch 256, buffer 30k, τ 0.001, lr 1e-4/3e-4; learn on
week 1, evaluate on week 2.

**B3. Why can floating DO save energy?** Their Figure 4, reproduced: more DO
lowers effluent NH₄ but *raises* total N (less denitrification); COD/BOD/TSS
barely move. The two limits conflict, so there is an optimum.

**B4. Simulator fix needed?** Yes — the first influent generator's ammonium
peak was only 1.25× mean, so NH₄ never neared 4 mg/L and the reward's
constraint never bound. Sharpened to the documented BSM1 statistics (S_NH
9–44 mg/L, load peak 1.5× mean). The forecaster and twins were retrained on
the new generator in the same commit.

**B5. Do the comparators match?** PID 3719.7 vs 3698.2 published, fuzzy
3719.9 vs 3697.1 (+0.6 %).

**B6. Does the saving reproduce?** DDPG-B: 6.8 / 4.9 / 5.0 % (dry/rain/storm)
vs published 7.24 / 5.52 / 6.06 %.

**B7. The cost the paper does not report.** Share of the week with effluent
NH₄ > 4: DDPG-B 38 % (dry) vs PID 12 %. Not a bug — eq. (23) at β₂ = 0.42 per
mg/L excess vs a full-aeration penalty of 1.0 makes long small exceedances
cheaper than the air.

**B8. The three corrections.** (1) *Not Markov*: with an incremental action
the actuator is a hidden integrator; OU noise pins it to a rail → add
`KLa_5/240` to the state, use an absolute action. (2) *Horizon*: 45 s with
γ = 0.99 ≈ 75 min, but effluent NH₄ responds over hours → hold actions 15 min.
(3) *Value scale*: values ~−250 from rewards ~−2.5; `∂Q/∂a` had the wrong sign
for four simulated weeks → reward × (1−γ), actor weight decay, pre-tanh
penalty. Plus: AE normalised as `KLa_5/240` in the reward (the paper does not
state units; raw AE ≈ 3500 would swamp 0.42·excess), and ~20 training weeks
instead of 7 days.

**B9. PANDA-RL — the four changes.** Forecast context into actor and critic;
distributional critic (32 quantiles) with a CVaR(α = 0.3) actor objective;
Lagrange multipliers per discharge limit instead of hand-tuned 0.38/0.42;
twin-assisted Dyna updates. Reduces exactly to DDPG with 1 quantile, α = 1, no
constraints, no forecast.

**B10. Does it help?** Fixed-weight sweep (same reward as paper 2, only the
network differs): Pareto front cheaper than DDPG-B's by +29 kWh/d at matched
violation rate (forecast only +19, CVaR only +26). **Caveats:** one seed per
point; not parameter- or input-matched (6 extra inputs, 32 critic outputs).

**B11. The Lagrangian budget?** Multipliers move the right way; but one of 4
frontier runs is dominated and a 5 % budget is not reached — the policy has
not converged at 20 weeks and one seed. PID itself violates 12 %, so a 5 %
budget needs *more* air than PID.

**B12. Dyna?** At equal real experience AE 3749 → 3205 kWh/d (faster
learning) but violations 14 % → 67 %: four times the gradient steps, dual
stepped once per episode. Intra-episode dual did not fix it (68 %). Proposed
fix (scale the dual step with the Dyna ratio) is untested.

---

## C. Prediction models

**C1. Forecaster.** Dilated causal TCN, 6 residual blocks, dilations 1–32,
width 64; 24 h of log inlet flow/NH₄/S_S at 15 min; 10/50/90 % quantiles over
2 h (monotone via cumulative softplus, pinball loss) + regime head.
**On 24 fresh scenarios:** 2-h median error ≈ 5 % flow / 6 % NH₄ vs 11 % /
30 % for persistence; 10–90 % band covers ≈ 80 % (nominal 80 %).

**C2. The regime head.** 95.8 % on validation, ≈ 36 % on fresh scenarios
(chance). Two reasons: the validation split drew random, overlapping windows
of the same scenarios (leakage); and the label is the *scenario's* weather,
not observable on the calm days of a storm scenario. Its P(wet) feeds PANDA
and PANDA-RL as one of six context inputs. Fix: per-window event labels,
scenario-level split.

**C3. Digital twin (paper 1).** GRU, hidden 96, 6-min steps, residual state
update + effluent-NH₄ head, scheduled sampling. **Closed-loop identification
bias:** trained only on PID data it learned a near-zero aeration gain (the PID
raises air exactly when demand rises), and the feed-forward moved the wrong
way; 60 % open-loop PRBS data fixed the signs. **Fresh-data accuracy:**
free-running 4.8-h RMSE ≈ 0.39 (DO) / 0.48 (nitrate) g/m³ vs ≈ 1.19 / 1.76
for persistence; step-response *direction* always right, *size* good for
some steps and poor for others (e.g. the small KLa decrease).

**C4. Early warning.** Every 6 min: P(effluent NH₄ > 4 within 12/30/60/90/120
min) via a discrete-time hazard head (monotone in horizon by construction),
plus NH₄ quantiles. Inputs: only signals plants measure (inlet, probes,
controller outputs, the stale effluent analyser, temperature, time of day,
forecaster context). Plants randomised (aeration in tanks 3–4 160–240 1/d,
11.5–17 °C, DO set point 0.8–2.0) because at paper settings NH₄ never exceeds
2.4. Train / val / test are different *plants*. Thresholds set on validation
for ≤ 1 false alarm per compliant day; lead time measured on test plants.
Compared against the analyser alarm, trend extrapolation and gradient
boosting. Numbers: Prediction models page §4.

**C5. How do you avoid leakage in the warning?** Labels look strictly ahead
(tested); windows end at the prediction step (tested); plant-level splits;
the analyser input is delayed by one cycle; scores focus on moments when the
plant is still compliant.

---

## D. Questions about method in general

**D1. Why should we believe a re-implemented BSM1?** Steady state matches the
published values; the comparators land within 0.6 % of paper 2's published
energy; paper 2's Figure-4 shape reproduces. And no claim is absolute.

**D2. Seeds?** Paper 1: 2 per configuration. Paper 2: 2 per learner; sweeps
1 per point. Every seed is reported, never the best one.

**D3. What would you do next?** Per-window regime labels for the forecaster;
3–5 seeds on the sweeps; parameter-matched controls for the network
comparison; scale the Dyna dual step; longer training for the Lagrangian arm.
