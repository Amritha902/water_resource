# Viva preparation — questions to expect, with precise answers

Each answer gives the short version first, then the details and where to point
on the website (`docs/index.html`, Part 1 / Part 2) or in the code. Numbers in
*italics* are produced by the experiments; the website always shows the current
values.

---

## A. The base paper

**A1. In one sentence, what does the paper do?**
It controls two coupled loops of the BSM1 wastewater plant (oxygen in reactor 5
via aeration `KLa_5`, nitrate in reactor 2 via internal recycle `Q_a`) with one
adaptive-critic agent per loop, where each agent *adds a learned correction to
an industrial PID* (`u = b + a`) and is charged for the tracking error of
*both* loops, so it accounts for the coupling it cannot model.

**A2. What is the "expert knowledge"?**
The prior control strategy `b_k` — an incremental (velocity-form) PID with
`Kp = diag{100, 1e5}`, `Ki = diag{20, 3e4}`, `Kd = diag{10, 1e3}`. The action
network's outer weights start at zero, so `a_0 = 0`: on the first step the
controller *is* the PID. That removes the trial-and-error phase of RL on a real
plant. Code: `baselines/pid.py:50`, `maacc/agent.py:109` (`zero_outer=True`),
test `test_maacc_starts_from_the_expert_prior`.

**A3. Why an incremental policy (`Δa`) and not `a` directly?**
Eq. (8): `a_k = a_{k-1} + Δa_k`. A disturbance or a bad network output then
corrupts only one *increment*, not the whole input. The paper accepts a slower
response because a WWTP is a slow process plant.

**A4. How does the paper handle the coupling `G_i(s_k)`?**
It never identifies it. The utility of eq. (10) has the term `zᵀ P_i z` with
the **full** error vector, so agent 1 (air) pays for nitrate error and agent 2
(recycle) pays for oxygen error. Each agent optimises a global objective while
moving only its own actuator. Code: `agent.py:139`.

**A5. What exactly does Theorem 1 prove?**
That the critic and actor *weight approximation errors* are uniformly
ultimately bounded (UUB) if `l_c < 1/(λ²‖ϑ‖²)` and
`l_a < 1/(‖δ‖² (β^M C)²)`. It does **not** prove closed-loop stability of the
tracking error — the paper argues that is inherited from the stable expert
prior. UUB says the error stays in *some* bounded set; it says nothing about
how large that set is.

**A6. Which baselines does the paper compare against? Did you reproduce them?**
RSCMPC, MADRL-PID, SUP-HDP, dHDP, DPPGADP, ACD-VarInf, FAACC, plus a
supplementary 3 %-noise experiment. **Not reproduced** — only the PID prior and
MAACC are compared here. Say this plainly.

**A7. Did you reproduce Table II?**
No, and the numbers are not comparable: the official BSM1 code and influent
files are not redistributable, so the plant was re-implemented from its
published definition and the influent generated from the documented
statistics. Every claim is a *relative* comparison on the same simulator.

---

## B. The simulator

**B1. How do you know your BSM1 is right?**
`tests/test_bsm1.py`: the open-loop steady state matches published BSM1
(`S_O,5 ≈ 0.49`, `S_NO,2 ≈ 3.66`, effluent TSS ≈ 12.5 g/m³); concentrations
stay non-negative; the settler thickens > 50×; increasing `KLa_5` raises
`S_O,5` and increasing `Q_a` raises `S_NO,2`.

**B2. What integration scheme?**
Fixed-step explicit Euler with 4 sub-steps per 45 s control period (≈ 11 s),
states clipped at zero (`plant.py:step`). Adequate for the BSM1 time constants;
a stiff solver would be more accurate but was not needed for the comparisons.

**B3. What temperature?**
15 °C (BSM1 default kinetic parameters). In the early-warning study the
nitrifier growth and decay rates are scaled with the BSM2 Arrhenius factors
`exp(0.098(T−15))` and `exp(0.069(T−15))`.

**B4. How are rain and storm generated?**
`bsm1/influent.py`: dry-weather base flow 18 446 m³/d with a two-peak diurnal
pattern and weekend drop; rain = sustained hydrograph (fast rise, slow
recession) that dilutes; storm = short intense hydrograph plus a first-flush
pulse of particulates. Canonical storm events at days 8.6 and 10.4.

---

## C. The reproduction and its deviations (Part 1, §8 on the site)

**C1. What did you change relative to the paper, and why?** — ten items, all
listed in the site table. The ones to know cold:

1. **Critic update direction** (TD(0) instead of eq. 15 as printed) — see C2.
2. **Leaky integration** `a_k = 0.999 a_{k-1} + Δa_k` and a clip at ±10 % of
   the actuator range (±24 1/d, ±9 223 m³/d) — realises the `R a²` penalty and
   keeps `a` a correction, not a replacement.
3. **Bounded increments** `Δa = Δa_max · tanh(·)`, `Δa_max` = 4 1/d and
   1 500 m³/d per 45 s step.
4. **Actor weight decay (2.0)** — without it the start-up transient latched the
   actor at its tanh rail permanently (observed before the fix).
5. **Bias input** on both networks — a tanh net without bias is zero at the
   origin and cannot represent `Q > 0` at the set point.
6. **Full actor Jacobian** `∂Q/∂Δa + ∂Q/∂a` — `a` depends on the actor's
   output through `a_k = a_{k-1} + Δa_k`.
7. **Normalised inputs** — equivalent re-parameterisation of the weight
   matrices.
8. **Hyper-parameters** `W=1, P=0.25, Y=0.02, R=0.05, λ=0.9`, critic 16
   hidden, actor 12, `l_c=0.02`, `l_a=0.002` — chosen by us.

**C2. Explain the eq. (15) problem precisely.**
Eq. (15): `e_k = U_k + λQ(x_k) − Q(x_{k-1})`. Eq. (18) descends ½e² in the
weights β that produce `Q(x_k) = βᵀϑ`. The gradient is `e · λ · ϑ`, so

```
Q_k  ←  Q_k − l_c λ ‖ϑ‖² e
```

For a slowly varying state, `Q_k ≈ Q_{k-1} = Q`, hence `e = U − (1−λ)Q` and

```
Q  ←  Q (1 + l_c λ (1−λ) ‖ϑ‖²)  −  l_c λ ‖ϑ‖² U
```

The multiplier is > 1 for **every** positive `l_c`, so `Q` grows geometrically.
It corrects the *later* prediction towards `(Q(x_{k-1}) − U_k)/λ` — the wrong
end of the Bellman equation. TD(0) instead corrects `Q(x_{k-1})` towards
`U_k + λQ(x_k)`, which is a contraction with factor λ.

*Evidence*: `tests/test_critic_update.py` — same network, same data, same
learning rate; true `Q(0)` = 0.943 (Monte Carlo, 200 000 paths); eq. (15)
ends at about −13 000; TD(0) ends at about 1.10.

**C3. Then is Theorem 1 wrong?**
No. UUB bounds the weight error by *some* constant; a bound can be enormous.
The paper's condition `l_c < 1/(λ²‖ϑ‖²)` is the no-overshoot condition for a
single step (the residual shrinks by `l_c λ² ‖ϑ‖²` per step). It does not make
the recursion over time a contraction.

**C4. Why is TD(0) 1.10 and not 0.943?**
The critic is a 24-unit random-feature network with only outer weights
trained, fitted online on a correlated stream, so it carries approximation
error (~16 %). The test only requires 0.3 < Q(0) < 3 and that Q increases with
|a| — the property the actor actually needs.

**C5. Why doesn't MAACC beat the PID in your runs?**
The prior is an **incremental** PID, i.e. it has integral action. Any slowly
varying correction `a` the actor adds is cancelled within a few control
periods by the integrator moving `b` by `−a` (site Part 1, §7, plots this). So
the tracking error `z` carries almost no information about `a`, and the only
utility term that still depends on `a` is `R a²`, minimised at `a = 0`. The
learned term is close to unidentifiable from tracking error alone. With a
de-tuned prior (25 % of the gains) MAACC recovers a few percent — comparable to
seed variance.

**C6. How many seeds? Training success rate?**
3 seeds per configuration. The paper's ">90 % success over 20 runs" was not
measured.

---

## D. The ML extension

**D1. What is new, in one sentence?**
The controller is given information the PID does not have — a probabilistic
forecast of the influent and a learned model of the plant — and uses it to act
*before* a disturbance reaches the aerated zone: `u = b + g·a + f`.

**D2. Why is it a strict generalisation of the paper?**
`g ≡ 1, f ≡ 0` reproduces eq. (7). `g ∈ (0,1]`, `|f| ≤ f_max`, so the input
stays in the same admissible set.

**D3. Forecaster architecture and training?**
Dilated causal TCN (6 residual blocks, dilations 1–32, kernel 3, width 64,
GroupNorm, GELU, dropout 0.1). Input: log of inlet flow, NH₄, S_S over 24 h at
15 min (96 steps). Output: 10/50/90 % quantiles for 8 steps (2 h); monotone by
cumulative softplus; pinball loss. Plus a 3-way regime head (cross-entropy,
weight 0.2). 160 random 14-day scenarios, AdamW, OneCycle, 12 epochs.

**D4. Is it any good?**
On 24 fresh scenarios: 2-h median error about 6 % (flow) vs 10 % for
persistence; the 10–90 % band covers the truth 79 % of the time (nominal
80 %) — well calibrated. **But** the regime head does not generalise: 37 % on
fresh scenarios (chance 33 %). The earlier 95.3 % came from a random split of
overlapping windows of the same scenarios (leakage), and the label itself is
ill-posed (a "storm scenario" has calm days that are indistinguishable from
dry weather). Fix: per-window event labels and a scenario-level split.

**D5. Digital twin architecture?**
GRU (hidden 96), 6-min step: `h_t = GRU([s_t,u_t,d_t], h_{t-1})`,
`s_{t+1} = s_t + MLP(h_t)` (residual), plus an effluent-NH₄ head. Trained with
scheduled sampling (feeding its own predictions back with probability up to
0.6) so it is fit for free-running rollouts.

**D6. What is closed-loop identification bias?**
Under feedback the input is correlated with the disturbance: the PID raises
`KLa_5` exactly when oxygen demand rises, so in closed-loop data "more air"
does not coincide with "more oxygen". A twin fitted on such data learned a
near-zero aeration gain; differentiating it moved the actuator the wrong way.
Fix: 60 % of training rollouts use open-loop PRBS (random levels held 5–60
min), which makes the causal gain identifiable.

**D7. How accurate is the twin?**
On 192 fresh 4.8-h windows, free-running RMSE ≈ 0.26 g/m³ for DO and ≈ 0.45
g/m³ for nitrate, vs ≈ 0.85 / 1.26 for persistence (about 3× better). The
earlier "0.28" was measured on validation windows from the training rollouts
and is optimistic. In step tests the direction is always right; the
magnitude is 50–130 % of the plant's depending on the step.

**D8. How is the feed-forward `f` computed?**
Every 8 control periods (6 min): roll the twin 8 steps (48 min) ahead with a
**differentiable copy of the PID in the loop**, twice — under the forecast
influent and with the influent frozen at its current value. Penalise only the
*difference* of the two predicted trajectories (discounted, γ = 0.9), plus a
one-sided penalty on a predicted rise in effluent NH₄ above 0.2 g/m³, plus an
effort term. Optimise a constant offset `δ` (tanh-bounded to ±40 1/d,
±15 000 m³/d) with 6 Adam steps, warm-started from the last solution; scale by
`exp(−3 · forecast uncertainty)`.

**D9. Why two rollouts?**
Offset-freeness: any standing bias of the twin appears in both and cancels,
so `f = 0` whenever the forecast says nothing will change and the
feed-forward never fights the PID's integrator. Scoring the absolute predicted
error instead made the solver spend the actuator correcting twin bias, and it
did worse than PID.

**D10. Why the PID inside the rollout?**
The PID will also react during the 48 min. Cancelling the *open-loop*
disturbance impact double-counts that reaction and over-compensates (it was
measured worse than PID).

**D11. Does PANDA beat PID?**
Under the paper's idealised protocol: no — the PID's DO error is dominated by
white actuator noise that no predictor can remove, and there is little
headroom. Under realistic instrumentation (slew limits, probe lag, stale
analyser): see the site's Part 2, §4 table for the current numbers; report
them as they are.

---

## E. The early warning

**E1. What is predicted?**
Every 6 min: P(effluent NH₄ > 4 g N/m³ at any time within 12/30/60/90/120 min)
and 10/50/90 % quantiles of effluent NH₄ at those horizons.

**E2. Why did you change the plant's operating point?**
At the paper's settings effluent NH₄ never exceeds 2.4 g/m³ — nothing to
predict. Real plants trim aeration to save energy and lose nitrification in
cold water. Each simulated plant draws KLa₃,₄ ∈ [160, 240] 1/d, T ∈ [11.5, 17]
°C, DO set point ∈ [0.8, 2.0]. About a third of samples are in violation.

**E3. What is a discrete-time hazard head and why use it?**
For each horizon interval j the network outputs λ_j = P(first exceedance in
interval j | none before). Then P(violation by h_j) = 1 − ∏_{l≤j}(1 − λ_l),
computed in log space. This guarantees the probability is non-decreasing in
the horizon — "likely within 30 min but unlikely within 2 h" cannot be
produced.

**E4. How do you avoid leakage?**
(1) Labels look strictly ahead: `event[i,j] = max(nh[i+1 : i+h_j+1]) > 4`
(tested). (2) Windows end at the prediction step (tested). (3) Train / val /
test are different *plants* (disjoint seeds), never random windows of the same
plant — the mistake that inflated the forecaster's regime accuracy. (4) The
analyser input is delayed by one cycle.

**E5. Why the "onset" subset?**
On an ongoing violation, "alarm if the analyser reads > 4" is trivially right.
The interesting cases are moments when the plant is currently compliant.

**E6. How are methods compared fairly?**
Each method's alarm threshold is chosen on validation plants for ≤ 1 false
alarm per compliant day; lead times are then measured on test plants. A false
alarm is an alarm run with no exceedance within 2 h of its start.

**E7. Is the deep model better than gradient boosting?**
Answer with the site's numbers. If the margin is small, say so: the honest
claim is then that the *formulation* (early warning from existing signals,
equal-nuisance evaluation, hazard head with monotone probabilities) is the
contribution, not the network.

---

## F. Things to say first, before being asked

1. The regime-classifier 95.3 % was leakage; on fresh data it is at chance.
2. The twin's 0.28 mg/L was in-distribution; on fresh data DO 0.26 / NO 0.45,
   still about 3× better than persistence.
3. MAACC does not beat the PID on this simulator, and why (integral action).
4. The eq. (15) critic update diverges; TD(0) fixes it.
5. The simulator is a re-implementation; only relative claims are made.
