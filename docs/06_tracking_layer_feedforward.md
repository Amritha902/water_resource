# The tracking-layer anticipatory feed-forward — a negative result

> **This did not work on the loop it was aimed at.** The full benchmark
> (54 runs, `docs/05_results.md`) has it consistently worse than the PID prior
> on dissolved oxygen, by 4 % to 32 % of IAE across every condition — which is
> the loop the feed-forward was designed for, so the negative result stands.
>
> One nuance the benchmark added: on the *nitrate* loop under measurement
> noise it is better than PID, by 6 % to 16 %. That is a side effect rather
> than the design intent, and it is smaller than what MAACC achieves on the
> same loop (23–29 %), so it does not rescue the approach.
>
> Kept because the reason it failed is the reason the project moved to
> paper 2's framing, and because the two engineering lessons in it
> (closed-loop identification bias, and offset-free feed-forward design) cost
> real time to find. The contribution the project rests on is in
> `docs/03_novelty.md`.

## Original framing: PANDA-MAACC

**P**redictive **AN**ticipatory **D**isturbance-**A**ware multiagent adaptive
critic control.

## 1. The gap, stated precisely

MAACC — and every reinforcement-learning WWTP controller it is compared
against — closes the loop on the tracking error `z_k = s_k - r_k`. A storm
raises the influent flow, the flow washes substrate through the reactors,
the oxygen uptake rate and the nitrate load change, `z_k` moves, and only
*then* does the controller respond. The disturbance is always one plant time
constant ahead of the controller.

But the influent is **measured** at the inlet, and it is **predictable**:
municipal sewage has a strong diurnal and weekly signature, and wet-weather
events announce themselves in the flow signal minutes before their load
reaches the aerated zone. None of that information is in the loop.

Section 4 of `docs/02_reproduction.md` adds a second, sharper reason to care:
because the expert prior has integral action, a learned correction driven by
tracking error alone is close to *unidentifiable* — the prior absorbs it. To
make the learned component earn its place it must be given information the
prior does not have. A forecast is exactly that.

## 2. The control law

```
u_{k,i}  =  b_{k,i}  +  g_{k,i} · a_{k,i}  +  f_{k,i}
            ^^^^^^^     ^^^^^^^^^^^^^^^^     ^^^^^^^
            expert      gated adaptive       anticipatory
            prior       critic term          feed-forward
```

Setting `g ≡ 1` and `f ≡ 0` recovers eq. (7) of the paper exactly, so PANDA
is a strict generalisation and every stability argument that holds for MAACC
holds for the `g = 1, f = 0` restriction. The gate and the feed-forward are
both explicitly bounded (`g ∈ (0, 1]`, `|f| ≤ f_max`), so the applied input
stays within the same admissible set the paper's Lyapunov analysis assumes,
and the prior remains the fallback whenever the learned layer is switched off.

## 3. The four learned components

### 3.1 Probabilistic multi-horizon influent forecaster

`src/wwtp/forecast/` — a dilated causal TCN over 24 h of inlet history
(flow, ammonium, readily-biodegradable COD at 15 min resolution) with

* a **quantile head** emitting the 10/50/90 % quantiles of every channel for
  each of the next eight 15-minute steps, trained with the pinball loss and
  with monotone quantiles enforced by construction (cumulative softplus), and
* an auxiliary **weather-regime head** (dry / rain / storm) which doubles as
  an interpretable storm alarm for the operator.

Trained on 160 randomised 14-day influent realisations; the three canonical
BSM1 profiles are held out, so every control result is an out-of-sample
evaluation of the forecaster.

### 3.2 Neural digital twin of the closed loop

`src/wwtp/twin/` — a GRU residual model on a 6-minute grid

```
h_t = GRU([s_t, u_t, d_t], h_{t-1})
s_{t+1} = s_t + g(h_t)          NH_eff,t = q(h_t)
```

The recurrent state stands in for everything the controller cannot measure —
biomass inventories, sludge blanket, reactors 1-4 — which is precisely the
unknown interconnection term `G_i(s_k)` of eq. (4). The paper charges the
agents for `G_i` through the utility but never predicts it; the twin does.
Trained with scheduled sampling so it is fit for free-running rollouts, not
just one-step prediction.

**This is where the first hard lesson was.** The twin was first identified on
data collected under the PID with a dither. The resulting model had a
near-zero aeration gain: holding `KLa_5` at 40 versus 220 changed the
predicted `S_O,5` by 0.05 mg/L, where the real plant moves by several mg/L.
The cause is classical closed-loop identification bias — with the PID in the
loop, `KLa_5` rises exactly when the oxygen demand rises, so aeration and
dissolved oxygen are almost uncorrelated in the data. Differentiating that
twin moved the actuator the wrong way and the controller was *worse* than
PID. The fix is 60 % open-loop amplitude-modulated PRBS rollouts, where the
control is resampled across the full admissible range independently of the
plant state. The gain sign is asserted in `tests/`.

### 3.3 Anticipatory feed-forward by differentiating the twin

Every 8 control periods the controller solves, by gradient descent through
the twin,

```
min_f  sum_h gamma^h || (s_h - r) ⊙ w ||^2
       +  w_NH sum_h gamma^h relu(NH_h - NH_limit)^2      <- discharge risk
       +  rho || f ||^2                                   <- effort
```

over a 48-minute horizon driven by the **forecast** influent `d̂`.

**This is where the second hard lesson was.** The obvious formulation — roll
the twin forward with the control held constant and cancel the predicted
deviation — grossly over-compensates, because the PID will have reacted long
before the horizon ends. That variant was measurably worse than plain PID.
The solve therefore carries a **differentiable copy of the incremental PID
inside the rollout**, so the twin predicts what the *closed loop* will do,
and `f` is only the residual move feedback cannot deliver in time. With the
effort penalty this drives `f → 0` whenever the prior already suffices, so
the feed-forward never fights the integral action.

The ammonium head makes the same rollout a **predictive discharge-violation
early warning**: `P(NH_eff > 4 g N/m³)` over the next 48 min, which enters the
objective as a one-sided risk penalty and is logged for the operator.

### 3.4 Forecast-conditioned agents and an uncertainty gate

The action and critic networks of both agents are conditioned on a six-
dimensional forecast context

```
c_k = [ dQ/Q at 30 min, dQ/Q at 2 h, dNH/NH at 2 h, dCOD/COD at 2 h,
        relative flow uncertainty, P(rain) + P(storm) ]
```

so the policy `pi_i(z_{k,i}, c_k)` is a *disturbance-scheduled* controller
rather than a pure error feedback. This is the minimal change to the paper's
architecture that makes the learned term identifiable: `c_k` is information
the PID prior genuinely does not have.

Finally the adaptive term is gated by forecast confidence,

```
g_k = 1 / (1 + kappa · (q90 - q10)/q50)
```

so when the forecaster is unsure the controller retreats towards the
industrially validated prior. That is the same design philosophy as the
paper's expert-knowledge term, extended to run-time rather than only to
initialisation.

## 4. Why this is a contribution and not a wrapper

* The paper's utility charges each agent for the coupling term `G_i` but
  never predicts it. The twin predicts it, and its gradient is what sets the
  feed-forward — so the coupling is *used*, not merely paid for.
* The paper's expert prior fixes the controller's behaviour at `k = 0`. The
  uncertainty gate extends that idea to every `k`: expert knowledge becomes
  the run-time fallback, with a learned, calibrated trigger.
* The incremental control strategy of eq. (8) is justified in the paper by
  its robustness to disturbance, while admitting it is slower to respond.
  The feed-forward removes exactly that cost, because the slow channel no
  longer has to wait for the error to appear.
* Nothing here needs a state that a real plant cannot measure. Inlet flow,
  ammonium and COD are standard instruments; the twin's recurrent state is
  driven entirely by measured signals.

## 5. Honest limitations

* The forecaster is trained on synthetic influent from the same generator
  that produces the evaluation profiles. Its accuracy on real plant data is
  unknown; the held-out canonical profiles are the strongest statement this
  repository can make.
* The feed-forward solve adds roughly 8× the compute of MAACC per control
  period — trivial for a 45-second process loop, but it is not free.
* No formal stability proof is offered for the full law. The bounded gate and
  bounded feed-forward keep the input in the admissible set, and the prior
  remains stabilising, but that is an argument, not a theorem. Extending
  Theorem 1 to the context-augmented networks is the obvious next piece of
  work.
