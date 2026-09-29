# The base paper, in my own words

> D. Wang, X. Li, J. Ren, J. Qiao, **"Multiagent Adaptive Critic Control With
> Expert Knowledge for Wastewater Treatment Plants"**, *IEEE Transactions on
> Industrial Informatics*, 2026. DOI 10.1109/TII.2026.3659924

This note records what the paper actually does, because everything in this
repository is either a reproduction of it or an extension of it.

## 1. The plant and the control problem

The plant is the IWA **Benchmark Simulation Model no. 1 (BSM1)**: an
activated-sludge process with a five-unit biological reactor (units 1-2
anoxic, units 3-5 aerated) followed by a ten-layer secondary clarifier.

Two concentrations must be held at their set points:

| controlled variable | symbol | manipulated variable | symbol | range |
|---|---|---|---|---|
| dissolved oxygen in unit 5 | `S_O,5` | oxygen transfer coefficient | `KLa_5` | 0 - 240 1/d |
| nitrate nitrogen in unit 2 | `S_NO,2` | internal recycle flow | `Q_a` | 0 - 92 230 m3/d |

The paper writes the set-point vector as `r = [1, 2]^T` for
`s = [S_O,5, S_NO,2]^T` (Sec. VI).

The difficulty is that the two loops are *coupled*: raising `Q_a` to push
nitrate into the anoxic zone also drags oxygen-rich liquor from unit 5
backwards, and raising `KLa_5` changes the nitrification rate and therefore
the nitrate that `Q_a` recirculates. Most earlier schemes simply ignore this
and tune two independent loops.

## 2. The algorithm (MAACC)

**Decentralised interconnected form.** The plant is written as

```
s_{k+1,i} = F_i(s_{k,i}, u_{k,i}) + G_i(s_k)          (eq. 4)
```

so each subsystem `i` has its own dynamics `F_i` plus an unknown coupling
term `G_i` that depends on the *whole* state. One agent is assigned per
subsystem, which is what keeps the training tractable — each agent optimises
a scalar control law instead of a joint one.

**Expert knowledge (eq. 7).** The applied input is split

```
u_{k,i} = a_{k,i} + b_{k,i}
```

where `b` is a *prior control strategy* — an industrially validated
controller, here an incremental PID with
`Kp = diag{100, 1e5}`, `Ki = diag{20, 3e4}`, `Kd = diag{10, 1e3}` — and `a`
is the learned adaptive term. Because the action network's outer weights are
initialised to zero, `a_0 = 0` and the loop *starts* as the expert
controller. That is the paper's answer to the trial-and-error cost of
reinforcement learning on real industrial plant.

**Incremental policy (eq. 8).** Rather than emitting `a` directly, the actor
emits an increment:

```
Delta a_{k,i} = pi_i(z_{k,i}),    a_{k,i} = a_{k-1,i} + Delta a_{k,i}
```

A disturbance then corrupts only the current increment instead of the whole
control input. The authors accept the slower response because a WWTP is a
process plant where response speed is not the binding constraint.

**Coupling-aware utility (eq. 10).** This is the cleverest part:

```
U_i = z_i^T W_i z_i  +  Delta a_i^T Y_i Delta a_i  +  a_i^T R_i a_i
      +  z^T P_i z                      <-- the FULL error vector
```

The last term charges agent `i` for the tracking error of *every* subsystem.
So although each agent only manipulates one actuator, its objective is
global. That is how the unknown `G_i` is handled without ever identifying it.

**Learning.** Q-function and policy are approximated by single-hidden-layer
networks (CN and AN, eqs. 13-14) with only the outer weights adapted online,
using the critic error of eq. (15) and the actor loss `L_a = 0.5 Q^2`
(eq. 17).

**Stability.** Theorem 1 proves the weight approximation errors are
uniformly ultimately bounded (UUB) for
`l_c < 1 / (lambda^2 ||vartheta||^2)` and `l_a < 1 / (||delta||^2 (beta^M C)^2)`.
Closed-loop stability of the tracking error is *not* proved — it is inherited
from the stable expert prior (Sec. V, final paragraph).

## 3. What the paper reports

Validation on BSM1 under dry, rain and storm influent, 14 days, 45 s
sampling, scored with

```
IAE = mean |z|,    ISE = mean z^2,    DEVmax = max |z|            (eq. 36)
```

MAACC is compared against RSCMPC, MADRL-PID, SUP-HDP, dHDP, DPPGADP,
ACD-VarInf and FAACC, plus a supplementary experiment with 3 %-of-range white
noise on both actuators. Reported training success rate: above 90 % over 20
runs.

## 4. Where I saw room to contribute

Three observations drove the extension in `docs/03_novelty.md`.

1. **MAACC is purely reactive.** Every signal the agents see — `z_k`,
   `Delta a`, `a` — is a consequence of a disturbance that has *already*
   changed the plant. The paper's own Figs. 5-6 show the deviations appear
   when the storm hits. Nothing in the algorithm can act before that.
2. **The influent is the dominant disturbance and it is measurable.** Flow,
   ammonium and readily-biodegradable COD are standard online inlet
   instruments. BSM1 itself supplies them. They are simply not used.
3. **`G_i(s_k)` is left unmodelled.** The utility charges for the coupling
   but never predicts it, so the agents pay for interaction they cannot
   foresee.

The paper's own conclusion points the same way: "future research may extend
its application to ... multiobjective optimization tasks involving both water
quality and energy consumption."
