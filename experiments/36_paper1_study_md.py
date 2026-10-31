"""Write docs/PAPER1_STUDY.md from results/paper1_study_tables.json (35) and paper1_tuning.json (32),
with round 1 (paper1_study_tables_round1.json, paper1_tuning_round1.json) for comparison.

    python experiments/36_paper1_study_md.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
T = json.loads((RES / "paper1_study_tables.json").read_text())
TU = json.loads((RES / "paper1_tuning.json").read_text())
T1 = json.loads((RES / "paper1_study_tables_round1.json").read_text())
TU1 = json.loads((RES / "paper1_tuning_round1.json").read_text())
W3 = ("dry", "rain", "storm")


def v(case, c, k):
    return T[case][c][k][0]


def red(case, c, k):
    """Reduction against the paper's PID, in %."""
    return 100 * (1 - v(case, c, k) / v(case, "PID", k))


def pct(x):
    if round(x) == 0:
        return "0 %"
    return f"{x:+.0f} %".replace("-", "−")


def row(case, c, k):
    if v(case, c, k) != v(case, c, k):          # not run in this case
        return "–"
    return f"{v(case, c, k):.4f} ({pct(-red(case, c, k))})" if c != "PID" else f"{v(case, c, k):.4f}"


ABL = ["PID", "MAACC", "PID+ILC", "MAACC+ILC", "PIDx3", "PIDx3+ILC", "PIDx3+SAC-IL", "Ours (no gate)", "Ours"]
NAME = {"PID": "Paper's PID prior", "MAACC": "MAACC, paper 1 as printed", "PID+ILC": "Paper's PID + our ILC",
        "MAACC+ILC": "MAACC + our ILC", "PIDx3": "PID ×3", "PIDx3+ILC": "PID ×3 + ILC",
        "PIDx3+SAC-IL": "PID ×3 + SAC-IL (no ILC)", "Ours (no gate)": "Ours without the safety gate",
        "Ours": "**Ours: PID ×3 + ILC + SAC-IL + gate**"}


def table(key, cases, label):
    h = "| Controller | " + " | ".join(label[c] for c in cases) + " |\n|---|" + "---|" * len(cases) + "\n"
    for c in ABL:
        if all(v(cs, c, key) != v(cs, c, key) for cs in cases):
            continue
        h += f"| {NAME[c]} | " + " | ".join(row(cs, c, key) for cs in cases) + " |\n"
    return h


def tradeoff_table():
    items = [("IAE_SO5_test", "Oxygen IAE, test week", "{:.4f}"), ("IAE_SNO2_test", "Nitrate IAE, test week", "{:.4f}"),
             ("DEVmax_SO5", "Largest oxygen deviation (mg/L)", "{:.3f}"), ("DEVmax_SNO2", "Largest nitrate deviation (mg/L)", "{:.3f}"),
             ("AE", "Aeration energy (kWh/d)", "{:.0f}"), ("PE_Qa", "Recycle pumping energy (kWh/d)", "{:.1f}"),
             ("KLa_move_per_day", "Aeration movement, KLa,5 (1/d per day)", "{:.0f}"),
             ("Qa_move_per_day", "Recycle movement, Q_a (m³/d per day)", "{:,.0f}"),
             ("at_limit_KLa", "Time with aeration at a limit", "{:.1%}"), ("at_limit_Qa", "Time with recycle at a limit", "{:.1%}"),
             ("eff_SNH", "Effluent ammonium (g N/m³)", "{:.2f}"), ("eff_TN", "Effluent total nitrogen (g N/m³)", "{:.2f}"),
             ("eff_SNH_over_4", "Time with effluent NH4 > 4", "{:.1%}"), ("eff_TN_over_18", "Time with effluent total N > 18", "{:.1%}"),
             ("gated_share", "Time the safety gate is on", "{:.1%}")]
    out = ""
    for case in W3:
        out += f"\n**{case.capitalize()} weather (14 days)**\n\n| | MAACC (paper) | PID ×3 | PID ×3 + ILC | Ours |\n|---|---|---|---|---|\n"
        for k, name, fmt in items:
            out += f"| {name} | " + " | ".join(fmt.format(v(case, c, k)) for c in ("MAACC", "PIDx3", "PIDx3+ILC", "Ours")) + " |\n"
    return out


C = T["compute"]


def v1(case, c, k):
    return T1[case][c][k][0]


def mid(tab, c, w, where):
    m = tab["midnight"]
    return m.get("round2", m)[f"{c}|{w}|clean"][where]


def dpct(new, old):
    """Change from round 1 to round 2, in %."""
    return pct(100 * (new / old - 1))
lab3 = {w: w.capitalize() for w in W3}
labn = {"noise": "Dry + actuator noise"}
o = {c: [red(w, c, "IAE_SO5_test") for w in W3] for c in ABL}
of = {c: [red(w, c, "IAE_SO5") for w in W3] for c in ABL}
n = {c: [red(w, c, "IAE_SNO2_test") for w in W3] for c in ABL}
ours_vs_ilc = [100 * (1 - v(w, "Ours", "IAE_SO5_test") / v(w, "PIDx3+ILC", "IAE_SO5_test")) for w in W3]
ours_vs_x3 = [100 * (1 - v(w, "Ours", "IAE_SO5_test") / v(w, "PIDx3", "IAE_SO5_test")) for w in W3]
maacc_ilc = [red(w, "MAACC+ILC", "IAE_SO5_test") for w in W3]
rng = lambda xs: (f"{min(xs):.0f}–{max(xs):.0f} %" if round(min(xs)) != round(max(xs)) else f"{round(min(xs)):.0f} %").replace("-0 %", "0 %")
n_seeds = T["dry"]["Ours"]["IAE_SO5"][2]
_best = min(r["IAE_SO5"] for r in TU["ilc_grid"] if r["ilc"] == TU["ilc"])
n_bad = sum(r["IAE_SO5_test"] > 3 * v(r["weather"], "PIDx3", "IAE_SO5_test") for r in
            json.loads((RES / "summary_paper1_study.json").read_text())["runs"]
            if r["controller"] == "Ours (no gate)" and r["weather"] == "rain")
n_near = sum(r["IAE_SO5"] <= 1.05 * _best for r in TU["ilc_grid"])

md = f"""# Paper 1 study: base paper, reimplementation, and our controller

A step-by-step account, with figures, for discussion. Every number comes from
the result files of the repository (`results/summary_paper1_study.json`), and
every figure from the scripts named in each step.

> **Update: read `docs/PAPER1_BASELINES_HELDOUT.md` with this study.** Two
> stronger baselines and a pre-registered held-out test change the
> conclusions: a PID tuned term by term, with the daily ILC, matches or beats
> our controller on the test week; SAC-IL adds nothing on top of it; paper
> 1's MAACC with repaired training does learn (−85 % in dry weather); and on
> the held-out conditions (new influent, official sensors) our controller
> does not generalise (it fails under the official sensor noise).

**In one paragraph.** Paper 1 (MAACC) claims −92 to −95 % oxygen tracking
error against its PID. Rebuilt exactly as printed, MAACC equals its PID. Our
controller keeps the paper's expert PID and adds three things: a stronger
tuning, a daily iterative learning control (ILC) and a soft actor-critic
trained by imitation learning (SAC-IL), behind a safety gate. Tuned and trained
on one week of dry weather only, it reduces the oxygen error on the test
week (see the note in step 9) by **{rng(o['Ours'])}** against the paper's PID (dry, rain, storm),
where MAACC gives {rng(o['MAACC'])}. The reduction is real but smaller than the
paper's claim, and it has costs, listed in step 10.

---

## Step 1. The problem and the data

* **Plant.** BSM1, the IWA benchmark activated-sludge plant: two anoxic tanks,
  three aerated tanks, a secondary settler. We use the official code
  (bsm2-python 0.0.16).
* **Two loops, as in paper 1.** Oxygen in tank 5, S_O,5 → 2 mg/L, through the
  aeration KLa,5 (0–240 1/d). Nitrate in tank 2, S_NO,2 → 1 mg/L, through the
  internal recycle Q_a (0–92 230 m³/d). One control step every 45 s.
* **Data.** The same as paper 1: the official BSM1 influent files for dry,
  rain and storm weather, 14 days each, one sample every 15 minutes
  (paper 1, page 6, Section VI and Fig. 3).
* **Split.** Everything we tune or train uses **dry weather, week 1 only**.
  Week 2 of every weather is the **test**: never seen, and it holds the rain
  and storm events. We also give the paper's own score over 14 days.

![plant](figures/reproduced/p1_fig1.png)

![data split](figures/study/data_split.png)

## Step 2. What paper 1 proposes (MAACC)

Wang, Li, Ren, Qiao, *Multiagent Adaptive Critic Control With Expert Knowledge
for Wastewater Treatment Plants*, IEEE T-II 2026.

1. The plant is treated as two interconnected subsystems, one per loop.
2. One agent per subsystem, each with an action network and a critic network
   (adaptive dynamic programming / heuristic dynamic programming family).
3. The critic is a Q-function that also sees the other loop, so each agent
   accounts for the coupling.
4. **Expert knowledge:** a PID prior is part of the control law,
   u = a (learned) + b (PID), so the controller starts as the PID.
5. **Incremental control:** the action network outputs a change Δa, which is
   integrated, to resist disturbances.
6. A Lyapunov proof of stability; Table I gives every weight and rate.
7. **Claim (Table II):** oxygen error −95 / −95 / −92 % and nitrate error
   −75 / −71 / −76 % against the PID (dry / rain / storm); Table III with
   actuator noise.

![MAACC framework](figures/reproduced/p1_fig2.png)

## Step 3. Our reimplementation of paper 1

* Every equation and the Table I values implemented as printed
  (`src/wwtp/maacc/exact.py`), 20 runs per case.
* **Our PID matches the paper's PID:** dry oxygen IAE 0.0114 vs 0.0114. So the
  plant and the test are the same as the paper's.
* **MAACC as printed equals its PID** in all 80 runs (IAE within 1 %). With the
  printed weights the learning signal is about 10⁻⁴, and with learning rates
  0.01 and 0.005 the networks hardly move; twelve other initial weights either
  stay at the PID or diverge.
* The paper's set points are printed as [1, 2]; its figures and PID numbers
  only match the standard [2, 1], which we use.

![paper 1 Fig. 4 redrawn](figures/reproduced/p1_fig4.png)

## Step 4. Why SAC with imitation learning instead of the paper's training

The paper's learner is an on-line adaptive critic: it updates once per step
from that step alone, through the gradient of its own critic, starting from
small random weights. Our runs show it does not learn on this plant. Soft
actor-critic with imitation learning addresses each of these points.

| | Paper 1: adaptive critic (MAACC) | SAC with imitation learning |
|---|---|---|
| Data use | each step used once, then discarded | all past steps kept and re-used (replay, off-policy) |
| Start | small random weights next to a PID | the actor first copies the expert from recorded data |
| Expert knowledge | added to the output (u = a + b) | learned from demonstrations, and kept as a prior in the loop |
| Exploration | none planned | a small, controlled random part (entropy term) |
| Critic | one network, one-step target | two critics, the smaller value used (less over-estimation) |
| Scaling | sensitive: the learning signal was 10⁻⁴ | inputs and reward normalised |
| Safety on a real plant | learns while controlling from the start | trained on recorded expert data first; a gate falls back to the expert |

These are the reasons this choice fits the problem; check with your guide
which of them he had in mind.

![MAACC vs SAC-IL](figures/study/maacc_vs_sacil.png)

## Step 5. Our controller

![our controller](figures/study/our_controller.png)

1. **PID ×3.** The paper's own PID, all gains ×{TU['gain']} (step 6).
2. **Daily ILC (the feedback-learning part).** About 96 % of the oxygen error
   left by the PID repeats every day: the PID reacts only after the daily load
   peak has moved the oxygen. The ILC stores, for every time of day, the
   correction the PID had to make, and each night takes over a share β of it,
   applied a little earlier (feedback-error learning). It learns on-line, from
   zero, while it controls. It acts on the **oxygen loop only**, and the
   nightly update is **bumpless**: when the new profile takes over part of
   the PID's work, the same amount is taken out of the PID's state, so the
   total aeration does not jump at midnight. Settings from step 6: β =
   {TU['ilc']['beta'][0]}, lead {TU['ilc']['lead'][0]} steps ({TU['ilc']['lead'][0] * 45 / 60:.0f} min), smoothing
   ±{TU['ilc']['smooth']} steps. `src/wwtp/sacil/ilc.py`.
3. **SAC-IL correction (the guide's method).** A small actor network (inputs:
   last three errors of both loops, both inputs, influent flow) adds a bounded
   correction to the aeration. Trained by imitation of PID ×3 + ILC, then by
   soft actor-critic on dry week 1, with the imitation term kept above a floor.
   Only the aeration is corrected: when SAC also moved the recycle, it traded
   nitrate for oxygen. `src/wwtp/sacil/sacil.py`.
4. **Safety gate.** Inputs are clipped to the range seen in training. In any
   step where an input leaves that range by more than 10 %, or where the
   expert needs a larger correction than the learner may give (actuator
   noise), the learned part is switched off and PID ×3 + ILC act alone.

## Step 6. Tuning, on dry week 1 only

* Gain sweep ×1 … ×6: the loop becomes unstable at ×{TU['gain_limit']}; the
  rule (largest whole factor below 70 % of that) gives **×{TU['gain']}**.
* ILC: {len(TU['ilc_grid'])} settings tried (share β, lead, smoothing). Rule fixed
  before the run: the lowest oxygen error among the settings whose largest
  oxygen deviation is no larger than PID ×3's alone, so the ILC may not add
  a worse moment. {TU['ilc_admissible']} of {len(TU['ilc_grid'])} settings pass. The map is flat:
  {n_near} settings are within 5 % of the best, so the result does not depend
  on a lucky setting (cells marked "peak ↑" break the rule).

![tuning](figures/study/tuning.png)

## Step 7. Training

![pipeline](figures/study/pipeline.png)

{n_seeds} independent training runs, each: expert run on dry week 1 → imitation
(behaviour cloning) → critic warm-up → 5 SAC episodes of 7 days. The policy
kept is the best on the training week.

![training](figures/study/training.png)

## Step 8. Results

Oxygen and nitrate IAE. Percentages are against the paper's PID (MAACC as
printed is equal to it). Learned controllers: mean of their runs
({n_seeds} for ours, 3 for MAACC and for SAC-IL without ILC).

**Test week (week 2, never used for tuning or training), oxygen S_O,5**

{table('IAE_SO5_test', W3, lab3)}
**Test week, nitrate S_NO,2**

{table('IAE_SNO2_test', W3, lab3)}
**Paper's protocol (14 days), oxygen**

{table('IAE_SO5', W3, lab3)}
**Dry weather with the paper's actuator noise (14 days), oxygen and nitrate**

{table('IAE_SO5', ('noise',), labn)}
{table('IAE_SNO2', ('noise',), labn)}

![results](figures/study/results.png)

![tracking](figures/study/tracking.png)

**What each part adds (oxygen, test week, dry / rain / storm):**

* PID ×3: {pct(-o['PIDx3'][0])} / {pct(-o['PIDx3'][1])} / {pct(-o['PIDx3'][2])}
* + daily ILC: {pct(-o['PIDx3+ILC'][0])} / {pct(-o['PIDx3+ILC'][1])} / {pct(-o['PIDx3+ILC'][2])}
* + SAC-IL and the gate (ours): {pct(-o['Ours'][0])} / {pct(-o['Ours'][1])} / {pct(-o['Ours'][2])};
  that is {pct(-ours_vs_ilc[0])} / {pct(-ours_vs_ilc[1])} / {pct(-ours_vs_ilc[2])} against PID ×3 + ILC
* For a fair comparison, our ILC added to the paper's own MAACC gives
  {pct(-maacc_ilc[0])} / {pct(-maacc_ilc[1])} / {pct(-maacc_ilc[2])}.

## Step 9. Checking the feedback-learning part (ILC) and the gate

* **Sensitivity.** The tuning map (step 6) is flat around the chosen setting.
* **Learning over the days.** The daily error falls over the first days and
  stays low; weekends (lower load) and the storm days raise it, then it
  recovers. It never diverged in any run.
* **A bug we found and fixed.** In the noise test the earlier code dropped the
  ILC's feedforward before adding the noise; after the fix the ILC acts in
  every case. The earlier noise results were redone.
* **Round 1 → round 2: two flaws found and fixed.** The first full test
  (round 1) showed two problems in the feedback-learning part:
  1. *The nitrate ILC was not robust.* Chosen on dry week 1, it made nitrate
     worse in rain (test week {v1('rain','PIDx3+ILC','IAE_SNO2_test'):.4f} against
     {v('rain','PIDx3','IAE_SNO2_test'):.4f} for PID ×3 alone): the nitrate loop is fast and
     a profile learned on dry days is wrong on rain days.
  2. *A jump at midnight.* The ILC switched its whole profile once a day, so
     the feedforward jumped at midnight. All ten largest oxygen deviations of
     our dry run were within 15 minutes of midnight.
  Round 2 drops the nitrate ILC and makes the nightly update bumpless; the
  ILC was re-tuned and SAC-IL retrained on dry week 1 exactly as before, and
  everything re-tested. **Honest note:** these two changes were decided after
  seeing round 1's test results, so for them the test week is no longer
  fully unseen. Round 1's files are kept (`results/*_round1.json`).

| Dry / rain / storm, test week | Round 1 | Round 2 |
|---|---|---|
| Ours, oxygen IAE | {' / '.join(f"{v1(w,'Ours','IAE_SO5_test'):.5f}" for w in W3)} | {' / '.join(f"{v(w,'Ours','IAE_SO5_test'):.5f}" for w in W3)} |
| Ours, nitrate IAE | {' / '.join(f"{v1(w,'Ours','IAE_SNO2_test'):.5f}" for w in W3)} | {' / '.join(f"{v(w,'Ours','IAE_SNO2_test'):.5f}" for w in W3)} |
| Ours, largest oxygen deviation, 14 days (mg/L) | {' / '.join(f"{v1(w,'Ours','DEVmax_SO5'):.3f}" for w in W3)} | {' / '.join(f"{v(w,'Ours','DEVmax_SO5'):.3f}" for w in W3)} |
| Ours, largest deviation within 15 min of midnight | {' / '.join(f"{mid(T1,'Ours',w,'midnight'):.3f}" for w in W3)} | {' / '.join(f"{mid(T,'Ours',w,'midnight'):.3f}" for w in W3)} |
| PID ×3 + ILC, oxygen IAE | {' / '.join(f"{v1(w,'PIDx3+ILC','IAE_SO5_test'):.5f}" for w in W3)} | {' / '.join(f"{v(w,'PIDx3+ILC','IAE_SO5_test'):.5f}" for w in W3)} |
| PID ×3 + ILC, nitrate IAE | {' / '.join(f"{v1(w,'PIDx3+ILC','IAE_SNO2_test'):.5f}" for w in W3)} | {' / '.join(f"{v(w,'PIDx3+ILC','IAE_SNO2_test'):.5f}" for w in W3)} |

  The midnight jump is gone (largest deviation near midnight in dry weather
  {mid(T1,'Ours','dry','midnight'):.3f} → {mid(T,'Ours','dry','midnight'):.3f} mg/L), and our largest dry-weather deviation
  is now {v('dry','Ours','DEVmax_SO5'):.3f} mg/L, below PID ×3's {v('dry','PIDx3','DEVmax_SO5'):.3f}. Nitrate in rain is back to the
  PID ×3 level; the price is dry weather, where the nitrate ILC had helped
  ({v1('dry','Ours','IAE_SNO2_test'):.5f} → {v('dry','Ours','IAE_SNO2_test'):.5f}). In the storm the largest deviation ({v('storm','Ours','DEVmax_SO5'):.2f}) is set by
  the storm itself, the same for PID ×3. A small dip (about 0.01 mg/L, in
  the half hour before midnight) remains: the load is not exactly the same
  from one day to the next, so the learned profile does not quite join up
  at midnight, and the smoothing spreads that step over the last half hour.

![midnight](figures/study/midnight.png)

* **Gate.** On: {v('dry','Ours','gated_share'):.1%} of the time in dry,
  {v('rain','Ours','gated_share'):.1%} in rain, {v('storm','Ours','gated_share'):.1%}
  in storm, {v('noise','Ours','gated_share'):.1%} with actuator noise. Without
  it, SAC-IL extrapolates in rain and storm (row "ours without the safety
  gate"): {n_bad} of {n_seeds} trained policies go badly wrong there, which the
  gate prevents. Training is also not always smooth: one of the {n_seeds} runs
  became unstable after its first SAC episode; since the policy kept is the
  best one on the training week, that run kept its first-episode policy.

![daily learning](figures/study/daily.png)

## Step 10. Trade-offs

{tradeoff_table()}
**Computing time per control step** (this machine, one core): paper's PID
{C['PID_ms'] * 1000:.0f} µs; ILC {C['ILC_ms_per_period_incl_daily_update'] * 1000:.0f} µs (including its
nightly update); SAC-IL actor {C['SAC_actor_ms'] * 1000:.0f} µs ({C['SAC_actor_parameters']:,} weights);
MAACC {C['MAACC_ms'] * 1000:.0f} µs (it learns at every step). All far below the 45 s step.

**Training.** SAC-IL needs recorded expert data (7 days of the plant) and
about 15 minutes of computing per run here; MAACC learns on-line but did not
learn. The ILC needs no training, only a few days of operation.

![trade-offs](figures/study/tradeoffs.png)

**In words.**

* **Better:** oxygen tracking much better in every weather; aeration and
  pumping energy about the same; effluent quality about the same.
* **Costs:** the actuators move more (PID ×3 roughly doubles the aeration
  movement); the stability margin is smaller (×3 is 60 % of the unstable
  gain); the ILC relies on the daily repetition of the load; the learned part
  needs training data and a safety gate to stay reliable outside its training
  range, and its gain is mostly in the weather it was trained on; nitrate
  gains come from the PID ×3 alone (no learning on that loop).

## Step 11. What we can claim

* On the official BSM1, with the paper's data, set points and metrics, our
  controller reduces the oxygen tracking error on the test week by
  **{rng(o['Ours'])}** against the paper's PID and against MAACC as printed
  (which gives {rng(o['MAACC'])}).
* Over the paper's 14-day protocol: {rng(of['Ours'])} on oxygen.
* Nitrate, test week: {pct(-n['Ours'][0])} / {pct(-n['Ours'][1])} / {pct(-n['Ours'][2])} (dry / rain / storm), all from
  the retuned PID.
* The largest oxygen deviation is no worse than PID ×3's in any weather
  (step 9).
* **The most robust part** is PID ×3 + the daily ILC on oxygen:
  {pct(-o['PIDx3+ILC'][0])} / {pct(-o['PIDx3+ILC'][1])} / {pct(-o['PIDx3+ILC'][2])} on oxygen in the test week, with no learned network.
* What the guide's SAC-IL part adds on top of PID ×3 + ILC (oxygen, test
  week): {pct(-ours_vs_ilc[0])} in dry weather (the kind it was trained on),
  {pct(-ours_vs_ilc[1])} in rain and {pct(-ours_vs_ilc[2])} in the storm. On its own (without
  the ILC) it gives {pct(-o['PIDx3+SAC-IL'][0])} / {pct(-o['PIDx3+SAC-IL'][1])} / {pct(-o['PIDx3+SAC-IL'][2])}
  against the paper's PID, close to what the ILC gives alone.
* **Not claimed:** the paper's −92 to −95 %, which no setting the paper allows
  reproduces; a real-plant result; a gain under heavy actuator noise beyond
  the retuned PID.

## Next step

Done: the stronger baselines and the held-out test, in
`docs/PAPER1_BASELINES_HELDOUT.md`, with the next steps they point to.

## Step 12. Questions to brainstorm

1. Should the ILC learn the weekly cycle (weekends) as well as the daily one?
2. Would SAC-IL gain more if it also saw the time of day or the ILC profile?
3. Can the gate be made smoother (blend instead of switch)?
4. How would the method behave on BSM2, or with sensor delay and noise?
5. Can the same scheme improve paper 2's energy saving while keeping the
   ammonium limit?
6. Which of the reasons in step 4 did the guide want to emphasise?

## Reproduce

```bash
python experiments/32_paper1_tuning.py            # step 6
python experiments/28_sac_il_paper1.py --seed 0 --episodes 5 --train-days 7 --noise-runs 0 --ilc --balanced --oxygen-only --tag sacil_v3   # seeds 0-4
python experiments/28_sac_il_paper1.py --seed 0 --episodes 5 --train-days 7 --noise-runs 0 --balanced --oxygen-only --tag sacil_noilc      # seeds 0-2
python experiments/33_paper1_study.py             # step 8: every controller, every case
python experiments/34_paper1_diagrams.py          # diagrams
python experiments/35_paper1_study_doc.py         # figures and tables
python experiments/36_paper1_study_md.py          # this document
```
"""

(ROOT / "docs" / "PAPER1_STUDY.md").write_text(md)
print("wrote docs/PAPER1_STUDY.md")
