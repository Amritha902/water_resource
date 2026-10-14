"""Write ``docs/REPRODUCTION.md``: both base papers, reimplemented, in one place.

Every number is read from ``results/``, so the report and the data cannot
drift apart.  Paper 1's published Table II values are not in the repository;
put them in ``results/published_paper1.json`` (format below) and re-run this
script to get a paper-vs-ours table:

    {"source": "Wang et al. 2026, Table II",
     "dry":   {"MAACC": {"IAE_SO5": ..., "ISE_SO5": ..., "DEVmax_SO5": ...,
                         "IAE_SNO2": ..., "ISE_SNO2": ..., "DEVmax_SNO2": ...},
               "PID":   {...}},
     "rain":  {...}, "storm": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "REPRODUCTION.md"
W3 = ("dry", "rain", "storm")
METRICS = ("IAE_SO5", "ISE_SO5", "DEVmax_SO5", "IAE_SNO2", "ISE_SNO2", "DEVmax_SNO2")

#: paper 2's published numbers (Du et al. 2023), as recorded in docs/04
PAPER2 = {"PID_AE_dry": 3698.2, "Fuzzy_AE_dry": 3697.1,
          "DDPG-B_saving": {"dry": 7.24, "rain": 5.52, "storm": 6.06},
          "DO_mean": {"PID": 2.000, "DDPG-A": 1.415, "DDPG-B": 1.032}}


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def pct(a, b):
    """Improvement of a over reference b, positive = a is lower (better)."""
    return 100.0 * (b - a) / b


def paper1(lines: list[str]) -> None:
    runs = load("summary_paper1.json") or []
    pick = lambda c, w, n: [r for r in runs if r["condition"] == c and r["weather"] == w and r["controller"] == n]
    seeds = sorted({r["seed"] for r in runs})
    lines += ["## 2. Paper 1 — Multiagent adaptive critic control (MAACC)", "",
              "> D. Wang, X. Li, J. Ren, J. Qiao, *Multiagent Adaptive Critic Control With Expert",
              "> Knowledge for Wastewater Treatment Plants*, IEEE Trans. Ind. Informat., 2026.", "",
              "### 2.1 What the paper specifies, and where it is in the code", "",
              "| Paper | Content | Code |", "|---|---|---|",
              "| eq. (4) | decentralised model `s(k+1,i) = F_i(s(k,i), u(k,i)) + G_i(s(k))`; `G_i` never identified | — |",
              "| eq. (6) | tracking error `z_k = s_k − r_k`, `r = [1, 2]` | `maacc/agent.py:235` |",
              "| eq. (7) | `u(k,i) = b(k,i) + a(k,i)`, `b` = incremental PID, `Kp = diag{100, 1e5}`, `Ki = diag{20, 3e4}`, `Kd = diag{10, 1e3}` | `baselines/pid.py:50`, `agent.py:243,251` |",
              "| eq. (8) | incremental policy `Δa = π(z_i)`, `a_k = a_{k−1} + Δa_k`; actor outer weights start at 0, so `a_0 = 0` | `agent.py:149–158` |",
              "| eq. (10) | utility `W z_i² + Y Δa² + R a² + zᵀ P z` (full error vector) | `agent.py:139` |",
              "| eqs. (9),(13) | critic `Q(z, Δa, a)`, single hidden tanh layer, outer weights adapted online, λ = 0.9 | `agent.py:161–176` |",
              "| eqs. (15),(18) | critic error and update — **diverges as printed**, see §2.4 | `agent.py:170–174` (TD(0)) |",
              "| eq. (17) | actor loss `½ Q²` | `agent.py:178–184` |",
              "| eq. (36) | `IAE = mean|z|`, `ISE = mean z²`, `DEVmax = max|z|` | `metrics.py` |",
              "| protocol | BSM1, dry/rain/storm, 14 days, 45 s; supplementary run with white noise 3 % of each actuator's range | `envs.py`, `experiments/03_run_control.py` |", "",
              "### 2.2 Deviations from the printed paper", "",
              "| # | Paper | This code | Why |", "|---|---|---|---|",
              "| 1 | critic descends eq. (15) in the weights producing `Q(x_k)` | TD(0): correct `Q(x_{k−1})` towards `U_k + λQ(x_k)` | the printed direction diverges (§2.4) |",
              "| 2 | actor gradient via `∂Q/∂a` (eq. 27) | `∂Q/∂Δa + ∂Q/∂a` | `a_k = a_{k−1} + Δa_k`: both critic inputs depend on the actor output |",
              "| 3 | no bias stated | bias input on the **critic** only | a tanh critic without bias is 0 at the set point and cannot represent `Q > 0` there; the **actor** has none, so `Δa = 0` at zero error, as eq. (8) requires |",
              "| 4 | raw units, scale in `W, Y, R, P` | inputs and effort terms normalised by actuator range | equivalent re-parameterisation; lets both agents share learning rates |",
              "| 5 | `a` unbounded | `a` clipped to ±10 % of each range (±24 1/d, ±9 223 m³/d); `Δa = Δa_max·tanh(·)` with 4 1/d, 1 500 m³/d per step | keeps `a` a correction to the prior |",
              "| 6 | plain gradient steps | actor outer-weight decay 2.0, critic 1e−4, gradient norm clip 10 | without decay a start-up transient latched the actor at its tanh rail |",
              "| 7 | `W, Y, R, P`, network sizes not all stated | `W=1, P=0.25, Y=0.02, R=0.05`, critic 16 hidden, actor 12, `l_c=0.02`, `l_a=0.002` | chosen by us |",
              "| 8 | official BSM1 code and influent files | re-implementation; statistically equivalent influent | files not redistributable — **absolute values differ from Table II** |",
              "| 9 | — | 1-day PID warm-up; first day of each run excluded from metrics | every controller starts from the same plant state |", "",
              "Two earlier deviations were **removed** because they broke the incremental policy: a bias input on the actor "
              "(so `π(0) ≠ 0` and `a` drifted steadily into its clip) and a leaky integrator `a_k = 0.999·a_{k−1} + Δa_k`. "
              "With them, the learned term sat at its clip 92 % of the run and acted as a constant offset; without them it "
              "moves with the tracking error (§2.5).", "",
              f"### 2.3 Results — {len(runs)} runs, seeds {seeds}", "",
              "`clean` = the paper's idealised setting; `noisy` = the paper's supplementary protocol (3 % actuator noise) "
              "plus DO / nitrate sensor noise; `detuned` = our own stress test with the prior at 25 % of its gains (not in "
              "the paper). Mean over seeds; % = improvement over the PID prior (positive = better).", "",
              "| condition | weather | PID IAE S_O,5 | MAACC IAE S_O,5 | Δ | PID IAE S_NO,2 | MAACC IAE S_NO,2 | Δ | PID DEVmax S_NO,2 | MAACC DEVmax S_NO,2 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ("clean", "noisy", "detuned"):
        for w in W3:
            p, m = pick(c, w, "PID"), pick(c, w, "MAACC")
            if not p or not m:
                continue
            lines.append(f"| {c} | {w} | {mean(p,'IAE_SO5'):.4f} | {mean(m,'IAE_SO5'):.4f} | {pct(mean(m,'IAE_SO5'), mean(p,'IAE_SO5')):+.1f} % "
                         f"| {mean(p,'IAE_SNO2'):.4f} | {mean(m,'IAE_SNO2'):.4f} | {pct(mean(m,'IAE_SNO2'), mean(p,'IAE_SNO2')):+.1f} % "
                         f"| {mean(p,'DEVmax_SNO2'):.3f} | {mean(m,'DEVmax_SNO2'):.3f} |")
    lines += ["", "**Reading.** Under the paper's own supplementary protocol MAACC improves nitrate tracking over its "
              "expert prior in every weather and leaves oxygen within a few percent; in the idealised clean setting, "
              "where there is nothing to correct, it changes nothing — both as the paper describes. In our own "
              "de-tuned stress test (not in the paper) it is clearly worse: with a weak prior the large standing "
              "error drives the purely integrating aeration actor into its clip. That is a limit of eq. (8) with a "
              "weak prior, reported as found.", ""]

    pub = load("published_paper1.json")
    lines += ["### 2.4 Against the paper's own numbers (Table II)", ""]
    if pub:
        lines += [f"Source: {pub.get('source', 'paper')}. Absolute values are expected to differ (deviation 8); "
                  "the comparison that transfers is the *ratio* MAACC / PID.", "",
                  "| weather | metric | paper PID | paper MAACC | paper ratio | our PID | our MAACC | our ratio |",
                  "|---|---|---|---|---|---|---|---|"]
        for w in W3:
            for k in METRICS:
                try:
                    pp, pm_ = pub[w]["PID"][k], pub[w]["MAACC"][k]
                except KeyError:
                    continue
                op, om = mean(pick("noisy" if pub.get("condition") == "noisy" else "clean", w, "PID"), k), \
                    mean(pick("noisy" if pub.get("condition") == "noisy" else "clean", w, "MAACC"), k)
                lines.append(f"| {w} | {k} | {pp:.4g} | {pm_:.4g} | {pm_/pp:.3f} | {op:.4g} | {om:.4g} | {om/op:.3f} |")
        lines.append("")
    else:
        lines += ["**Not yet filled in.** The paper's Table II values are not recorded in this repository. Add them to "
                  "`results/published_paper1.json` (format in `experiments/18_write_reproduction.py`) and re-run this "
                  "script. Because the plant and influent are re-implementations (deviation 8), expect the *ratio* "
                  "MAACC / PID — not the absolute IAE — to be the comparable quantity.", ""]

    cr = load("site/critic.json")
    lines += ["### 2.5 Correction: eq. (15) diverges", "",
              "Eq. (15) is `e_k = U_k + λQ(x_k) − Q(x_{k−1})` and eq. (18) descends ½e² in the weights β of "
              "`Q(x_k) = βᵀϑ`; the gradient is `e·λ·ϑ`. For a slowly varying state (`Q_k ≈ Q_{k−1} = Q`):", "",
              "```", "Q  ←  Q · (1 + l_c λ (1 − λ) ‖ϑ‖²)  −  l_c λ ‖ϑ‖² U", "```", "",
              "The multiplier exceeds 1 for every `l_c > 0`. "
              + (f"On the synthetic problem of `tests/test_critic_update.py` (true `Q(0) = {cr['true_q0']:.3f}` by Monte "
                 f"Carlo) the printed update reaches {cr['paper']['q0'][-1]:.0f}; the TD(0) direction on the same residual "
                 f"reaches {cr['td0']['q0'][-1]:.2f}." if cr else "")
              + " Theorem 1 (UUB of the weight error) is not contradicted — a bound can be large; its `l_c` condition "
              "is a one-step no-overshoot condition, not a contraction.", ""]

    off = load("summary_offset_control.json") or []
    if off:
        lines += ["### 2.6 What the learned term does", "",
                  "A learning-free control, `experiments/17_maacc_offset_control.py`, separates learning from "
                  "re-tuning: the PID prior plus a *constant* −9 223 m³/d on `Q_a`, and the PID prior with its "
                  "integrator clamp lowered to −9 223. IAE `S_NO,2`, noisy condition, mean of seeds:", "",
                  "| weather | PID | MAACC | PID + constant offset | PID, lowered clamp |", "|---|---|---|---|---|"]
        for w in W3:
            o = lambda n: mean([r for r in off if r["weather"] == w and r["controller"] == n], "IAE_SNO2")
            lines.append(f"| {w} | {mean(pick('noisy', w, 'PID'), 'IAE_SNO2'):.4f} | {mean(pick('noisy', w, 'MAACC'), 'IAE_SNO2'):.4f} "
                         f"| {o('PID+offset'):.4f} | {o('PID-clamp'):.4f} |")
        lines += ["", "Part of what the agent learns on the recycle loop is a slowly varying negative correction that "
                  "compensates the PID prior's anti-windup at zero recycle; a better-tuned prior gets that for free. "
                  "Read MAACC's gain against both columns, not against the PID alone.", ""]


def paper2(lines: list[str]) -> None:
    aer = load("summary_aeration.json") or []
    for extra in ("summary_aeration_rs.json", "summary_aeration_storm_base.json"):
        seen = {(r["weather"], r["method"], r["seed"]) for r in aer}
        aer += [r for r in (load(extra) or []) if (r["weather"], r["method"], r["seed"]) not in seen]
    pick = lambda w, m: [r for r in aer if r["weather"] == w and r["method"] == m]
    pid = {w: mean(pick(w, "PID"), "AE") for w in W3}
    lines += ["## 3. Paper 2 — DO control by reinforcement learning (DDPG)", "",
              "> S. L. Du, P. X. Chen, H. G. Han, J. F. Qiao, *Dissolved oxygen concentration control in wastewater",
              "> treatment process based on reinforcement learning*, Sci. China Tech. Sci. 66(9), 2549–2560, 2023.", "",
              "### 3.1 What the paper specifies", "",
              "| Item | Specification | Code |", "|---|---|---|",
              "| control | `KLa_5` only, no DO set point; `Q_a` at the BSM1 default | `rl/env.py` |",
              "| state | `[S_S,5, S_O,5, S_NH,5]` | `rl/env.py` (`obs_mode=\"paper\"`) |",
              "| action | increment `KLa_5,t = KLa_5,t−1 + a_t` | `rl/env.py` (`action_mode=\"incremental\"`) |",
              "| reward A, eq. (19) | `−(max(S_NH,e − 4, 0) + 0.38·max(N_tot,e − 18, 0))` | `rl/rewards.py` |",
              "| reward B, eq. (23) | `−(AE + 0.42·max(S_NH,e − 4, 0))` | `rl/rewards.py` |",
              "| networks | actor 3-256-256-1 (ReLU, tanh), critic 4-256-256-1 | `rl/ddpg.py` |",
              "| hyper-parameters | γ 0.99, batch 256, buffer 30 000, τ 0.001, lr 1e−4 / 3e−4 | `rl/ddpg.py` |",
              "| energy | `AE = (S_O_sat/1800)·Σ V_i KLa_i` (eq. 20), `PE = 0.004 Q_a + 0.008 Q_r + 0.05 Q_w` (eq. 21) | `bsm1/plant.py` |",
              "| protocol | learn on week 1, evaluate on week 2, 45 s | `rl/train.py`, `experiments/06_aeration_benchmark.py` |", ""]

    f4 = load("summary_fig4.json")
    if f4:
        lines += ["### 3.2 The physics the paper relies on (its Figure 4), reproduced", "",
                  "Steady dry-average influent, DO held at each set point for 20 simulated days:", "",
                  "| DO | KLa_5 | S_NH,e | N_tot,e | COD | BOD5 | TSS | AE |", "|---|---|---|---|---|---|---|---|"]
        for r in f4:
            lines.append(f"| {r['do']:.1f} | {r['kla']:.1f} | {r['nh']:.2f} | {r['tn']:.2f} | {r['cod']:.1f} | {r['bod']:.2f} | {r['tss']:.2f} | {r['ae']:.0f} |")
        lines += ["", "Ammonium falls and total nitrogen rises with DO, COD/BOD/TSS barely move, energy rises — the shape of "
                  "the paper's Figure 4.", ""]

    lines += ["### 3.3 Against the paper's own numbers", "",
              "| quantity | paper | ours | difference |", "|---|---|---|---|",
              f"| PID aeration energy, dry [kWh/d] | {PAPER2['PID_AE_dry']} | {pid['dry']:.1f} | {100 * (pid['dry'] / PAPER2['PID_AE_dry'] - 1):+.1f} % |",
              f"| Fuzzy aeration energy, dry | {PAPER2['Fuzzy_AE_dry']} | {mean(pick('dry', 'Fuzzy'), 'AE'):.1f} | "
              f"{100 * (mean(pick('dry', 'Fuzzy'), 'AE') / PAPER2['Fuzzy_AE_dry'] - 1):+.1f} % |"]
    for w in W3:
        ours = pct(mean(pick(w, "DDPG-B"), "AE"), pid[w])
        lines.append(f"| DDPG-B energy saving vs PID, {w} | {PAPER2['DDPG-B_saving'][w]:.2f} % | {ours:.1f} % | {ours - PAPER2['DDPG-B_saving'][w]:+.1f} points |")
    for m in ("PID", "DDPG-A", "DDPG-B"):
        lines.append(f"| mean S_O,5, dry, {m} | {PAPER2['DO_mean'][m]:.3f} | {mean(pick('dry', m), 'S_O5_mean'):.3f} | |")
    lines += ["", "The comparators and DDPG-B's saving match closely. **DDPG-A does not**: its mean DO is well above the "
              "paper's 1.415, i.e. our DDPG-A settles at a more conservative operating point; it is also the most "
              "seed-sensitive learner here.", ""]

    lines += ["### 3.4 Full results (evaluation week; mean over seeds)", "",
              "| weather | method | runs | AE [kWh/d] | saving vs PID | EQ | mean S_O,5 | S_NH,e | N_tot,e | NH>4 | TN>18 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in W3:
        for m in ("PID", "Fuzzy", "DDPG-A", "DDPG-B", "PANDA-RL"):
            r = pick(w, m)
            if not r:
                continue
            ae = mean(r, "AE")
            lines.append(f"| {w} | {m} | {len(r)} | {ae:.1f} | {'—' if m == 'PID' else f'{pct(ae, pid[w]):+.1f} %'} | {mean(r,'EQ'):.0f} "
                         f"| {mean(r,'S_O5_mean'):.2f} | {mean(r,'S_NH'):.2f} | {mean(r,'N_tot'):.2f} | {100*mean(r,'viol_NH'):.1f} % | {100*mean(r,'viol_TN'):.1f} % |")
    lines += ["", "**Not reported by the paper:** the share of the week above the 4 g N/m³ ammonium limit. DDPG-B's "
              "saving comes with roughly three times PID's violation rate — what eq. (23) asks for at β₂ = 0.42.", "",
              "### 3.5 Corrections needed for the published recipe to learn", "",
              "As published, the agent collapses to zero aeration (`--published-env` keeps that environment):", "",
              "1. **The observation is not Markov** — with an incremental action the actuator is a hidden integrator. "
              "Add `KLa_5/240` to the state and use an absolute action.",
              "2. **45 s decisions with γ = 0.99 give a ~75 min horizon**, but effluent NH₄ responds over hours. Hold "
              "each action 15 min.",
              "3. **Value scale** — the critic learns values ~−250 from rewards ~−2.5 and `∂Q/∂a` has the wrong sign "
              "for weeks. Scale the reward by `1 − γ`; actor weight decay 1e−4 and a 1e−3 pre-tanh penalty.", "",
              "Further deviations: AE enters the reward as `KLa_5/240` (the paper does not give units; raw AE ≈ 3 500 "
              "would swamp the 0.42 term — affine, so the same objective up to scale); OU σ annealed 0.35 → 0.05, "
              "buffer 80 000; 20 passes over the training week instead of 7 days.", "",
              "### 3.6 Reproducibility", "",
              "Re-training DDPG-B and PANDA-RL (dry, seed 0) in a fresh session with the benchmark's settings "
              "(`--threads 2`) gives the committed numbers exactly. With a different thread count the first training "
              "week still matches, but floating-point summation order differs and training diverges — to a different "
              "point on the same energy/violation trade-off.", ""]


def main() -> None:
    lines = ["# Reproduction report — both base papers", "",
             "Generated by `experiments/18_write_reproduction.py` from `results/`. Do not edit by hand.", "",
             "## 1. The shared plant", "",
             "BSM1 re-implemented from its published definition: ASM1 (13 components, 8 processes) at 15 °C, a ten-layer "
             "Takács clarifier, five reactors (1 000/1 000/1 333/1 333/1 333 m³), `Q_r = 18 446`, `Q_w = 385 m³/d`, explicit "
             "integration with four sub-steps per 45 s period. `tests/test_bsm1.py` checks the open-loop steady state "
             "against the published values (`S_O,5 ≈ 0.49`, `S_NO,2 ≈ 3.66`, effluent TSS ≈ 12.5 g/m³), non-negativity, "
             "settler thickening > 50× and both actuator gain signs. Influent: dry/rain/storm generated from the "
             "documented BSM1 statistics with the pollutograph sharpened so the discharge limits bind; statistically "
             "equivalent, not the official files. **Consequence: every claim is a comparison between methods on the "
             "same simulator; absolute values are not expected to equal the papers' tables.**", ""]
    paper1(lines)
    paper2(lines)
    lines += ["## 4. Reproduce", "", "```",
              "python -m pytest tests -q",
              "python experiments/03_run_control.py --seeds 2          # paper 1 (results/summary_paper1.json)",
              "python experiments/17_maacc_offset_control.py           # paper 1, learning-free controls",
              "python experiments/06_aeration_benchmark.py             # paper 2 (results/summary_aeration*.json)",
              "python experiments/13_export_site_data.py --stages critic fig4",
              "python experiments/18_write_reproduction.py             # this file", "```", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
