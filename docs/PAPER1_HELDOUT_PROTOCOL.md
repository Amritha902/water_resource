# Paper 1: held-out test protocol (written and pushed before any held-out run)

The study's test week (week 2 of the official dry, rain and storm files) was
used to find the two flaws fixed in round 2, so it is no longer fully unseen
for those choices. This protocol defines a new test on data and conditions
that no setting, training run or design choice has looked at. It is fixed
here, committed and pushed **before** the held-out runs are made; the git
history shows the order. Every result will be reported, whichever way it
goes.

## 1. What is frozen

All settings come from dry weather days 0–7 only, as in the study:

| # | Controller | Settings |
|---|---|---|
| 1 | Paper's PID | paper 1's gains |
| 2 | MAACC as printed | `src/wwtp/maacc/exact.py`, seed 0 |
| 3 | PID ×3 | `results/paper1_tuning.json` |
| 4 | PID ×3 + ILC | `results/paper1_tuning.json` (oxygen ILC, bumpless) |
| 5 | Tuned PID | `results/paper1_tuned_pid.json` (P and I tuned per loop) |
| 6 | Tuned PID + ILC | `results/paper1_tuned_pid.json` |
| 7 | Repaired MAACC on the paper's PID | `artifacts/maacc_rep_paper_actor_seed{0-4}.pt` |
| 8 | PID ×3 + ILC + repaired MAACC + gate | `artifacts/maacc_rep_stack_actor_seed{0-4}.pt`, same gate as ours |
| 9 | **Ours**: PID ×3 + ILC + SAC-IL + gate | `artifacts/sacil_v3_actor_seed{0-4}.pt` (round 2) |

No controller is re-tuned or retrained for the held-out test. The learned
controllers (7, 8, 9) are run with each of their 5 trained policies.

## 2. Held-out conditions

**H1. New influent: the official BSM2 influent.** `dyninfluent_bsm2.csv`
(shipped with bsm2-python; 609 days at 15 min; a different, longer,
phenomenologically generated influent with seasonal and weekly patterns).
Six 14-day windows, starting on days **245, 301, 357, 413, 469, 525** (every
8 weeks through the BSM2 evaluation year). Flow scaled by one constant so
that the whole file's mean flow equals the BSM1 dry-weather mean
(factor 0.878); concentrations unchanged; temperature 15 °C (the BSM1 plant
model has no temperature dependence). Sensors ideal, as in paper 1.

**H2. Real sensors: the official BSM1 sensor models.** The official dry,
rain and storm files, with the sensors of the official BSM1 closed-loop
model (bsm2-python `aerationcontrolinit_bsm1.py`): S_O,5 class A,
T90 = 1 min; S_NO,2 class B0, T90 = 10 min; white noise of standard
deviation 2.5 % of the range 0–10 g/m³ (0.25 g/m³) on both. Three noise
seeds (0, 1, 2) per weather.

**H3. Sensor delay only.** As H2, without the noise (sensor dynamics only).
One run per weather.

In every condition each run starts from the same warm-up state as the study
and lasts 14 days; the **score is days 7–14** (the first week lets the plant
settle on the new influent and the ILC learn), computed on the true
concentrations, not the measured ones.

## 3. Measures and comparisons

* Primary: oxygen IAE (S_O,5), days 7–14.
* Secondary: nitrate IAE (S_NO,2), largest oxygen deviation, aeration
  movement (KLa,5 per day), aeration energy, effluent ammonium and total
  nitrogen, time the gate is on.
* **Failure:** a run in which S_O,5 leaves 0.3–4 mg/L is stopped and counted
  as a failure; failures are reported per controller and condition, and
  excluded from the means (the number of runs in each mean is given).
* Comparisons reported for each condition, as the change of mean oxygen IAE
  and the number of windows (H1) or weathers × seeds (H2) in which ours is
  better:
  (a) ours vs PID ×3 + ILC (does SAC-IL add anything on new data?);
  (b) ours vs PID ×3 + ILC + repaired MAACC (is SAC-IL better than a fair,
  repaired version of the paper's learner?);
  (c) ours vs tuned PID + ILC (is the gain more than good PID tuning?);
  (d) everything vs the paper's PID.
* We will call ours better than another controller on a condition only if
  its mean oxygen IAE is lower **and** it is better in at least two thirds
  of the windows (H1) or of the weather × seed runs (H2).

## 4. What was seen before this protocol was written (disclosure)

* All results on the official dry, rain and storm files without sensors
  (the study, rounds 1 and 2).
* While writing the sensor code, two one-day smoke tests: PID ×3 with the
  official sensors on BSM2 window 245 left the 0.3–4 mg/L range on day 1;
  the paper's PID with the official sensors on dry day 1 did not (oxygen IAE
  0.28). No other run on BSM2 influent or with sensors was made.
* The tuned PID's tuning results on dry week 1 (that is its training data).

Runner: `experiments/40_paper1_heldout.py` → `results/summary_paper1_heldout.json`.

## Amendment 1 (before any held-out run)

On the official test week (not the held-out data), the tuned PID + ILC
(controller 6) turned out better than ours (controller 9) in every weather.
So one controller is added, trained exactly as ours but on that base:

| # | Controller | Settings |
|---|---|---|
| 10 | **Ours on the tuned PID**: tuned PID + ILC + SAC-IL + gate | `artifacts/sacil_tuned_actor_seed{0-4}.pt` (dry week 1, 5 episodes, as ours) |

Added comparison: (e) controller 10 vs tuned PID + ILC. Everything else in
this protocol is unchanged; no held-out run has been made.
