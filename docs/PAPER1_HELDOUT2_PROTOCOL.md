# Paper 1: held-out test 2 protocol (written and pushed before any of its runs)

The first held-out test (`docs/PAPER1_HELDOUT_PROTOCOL.md`) showed two flaws
that every controller design shared, and the fixes below were made after
seeing it. That test's data is therefore no longer unseen for them. This
second test uses **new** data and noise. It is fixed here and pushed before
any of its runs, and every result will be reported.

## 1. The two fixes (made after held-out test 1)

1. **ILC anti-windup** (`src/wwtp/sacil/ilc.py`, `Paper1Env.step`). The PID's
   limits move with the ILC feedforward, so PID + feedforward can reach
   either actuator limit and the PID never winds up. Periods spent at a
   limit teach the ILC nothing, and a day mostly at a limit keeps
   yesterday's profile. There is no new tuning. Checked while building it on
   data already seen: BSM2 window 245 without a clarifier (tuned PID + ILC,
   oxygen IAE days 7–14: 0.266 without anti-windup, 0.112 with it, 0.112
   for the PID alone) and the official dry and rain test weeks (unchanged:
   0.0011, 0.00115).
2. **Sensor-noise robustness** (`experiments/42_robust_pid.py`). This is the
   tuned PID + ILC with anti-windup, plus a first-order filter on the oxygen
   measurement, with the PID gains of each loop scaled. Both were chosen on
   dry days 0–7 with the official sensor models and noise seeds 100 and 101,
   never used in any test, by the rule stated in that script. A first search
   filtered both loops and made the nitrate loop oscillate; the second
   search filtered the oxygen loop only. Both searches are kept
   (`results/paper1_robust_pid_v1.json`, `results/paper1_robust_pid.json`).
   Chosen: filter 0.75 min, gains ×0.5 on both loops. On dry days 0–7 its
   oxygen IAE is 0.30 with the sensors (the tuned PID + ILC: 0.41, one
   failure) and 0.0020 with ideal sensors (0.00115). On the study's official
   test week (ideal sensors, already seen), oxygen is −83 / −80 / −73 %
   against the paper's PID, with the paper's aeration movement (522 vs 501).

## 2. Frozen controllers

| # | Controller | Settings |
|---|---|---|
| 1 | Paper's PID | paper 1 |
| 2 | MAACC as printed | seed 0 |
| 3 | Tuned PID | `results/paper1_tuned_pid.json` |
| 4 | Tuned PID + ILC (no anti-windup) | `results/paper1_tuned_pid.json` |
| 5 | **Tuned PID + ILC with anti-windup** | the same, `antiwindup=True` |
| 6 | **Robust**: filtered, re-scaled tuned PID + ILC with anti-windup | `results/paper1_robust_pid.json` |
| 7 | Repaired MAACC on the paper's PID | `artifacts/maacc_rep_paper_actor_seed{0-4}.pt` |
| 8 | Ours, round 2: PID ×3 + ILC + SAC-IL + gate | `artifacts/sacil_v3_actor_seed{0-4}.pt` |

Nothing is re-tuned or retrained after this file is pushed.

## 3. Held-out conditions (all new)

New BSM2 windows: 14 days starting on days **273, 329, 385, 441, 497 and
553** (four weeks after each window of test 1, with no overlap). The flow is
scaled as in test 1.

* **H4: new influent after a primary clarifier.** The particulate
  components (X_I, X_S, X_B,H, X_B,A, X_P, X_ND, TSS) are halved, a fixed
  simplification of a primary clarifier (`PRIMARY_REMOVAL = 0.5`). Ideal
  sensors.
* **H5: new influent, raw** (no clarifier). The plant is overloaded, which
  tests behaviour at the actuator limits. Ideal sensors.
* **H6: official sensors, new noise.** The official dry, rain and storm
  files with the official BSM1 sensor models, with new noise seeds
  **10, 11, 12**.
* **H7: new influent after the clarifier, with the official sensors**
  (noise seed 10): new data and real sensors together.

As before: start from the same warm-up state, 14 days, **score on days
7–14** from the true concentrations. A run whose oxygen leaves
0.3–4 mg/L is a failure; failures are counted and excluded from the means.

## 4. Pre-declared questions

"Better" means a lower mean oxygen IAE **and** better in at least two
thirds of the windows or runs, as in test 1.

* (i) Anti-windup: is controller 5 better than controller 4 on H5, and no
  worse than 1 % on H4?
* (ii) Robustness: does controller 6 have fewer failures than the paper's
  PID on H6 and H7, and is it better than the paper's PID there?
* (iii) The price of robustness: how much worse is controller 6 than
  controller 5 on H4 (ideal sensors)? This is reported, not tested.
* (iv) Is controller 6 better than the paper's PID on H4 and H5?
* (v) Is the repaired MAACC better than the paper's PID on H4–H7?

Runner: `experiments/43_paper1_heldout2.py` → `results/summary_paper1_heldout2.json`.
