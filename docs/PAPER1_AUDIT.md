# Paper 1: audit of the numbers

`python experiments/46_paper1_audit.py` runs the audit. It writes
`results/paper1_audit.json` and gives **76 passed, 0 failed**.

## What was checked

1. **Settings against paper 1's text** (page 7, Section VI, and eq. 36 of the
   uploaded PDF).

   | Setting | Paper 1 | Code |
   |---|---|---|
   | PID gains | Kp = diag{100, 100 000}, Ki = diag{20, 30 000}, Kd = diag{10, 1000} | the same |
   | Control period | 45 s | the same |
   | Run length | 14 days | the same |
   | Input limits | 0–240 and 0–92 230 | the same |
   | Actuator noise | std 3 % of the upper bound | the same |
   | IAE, ISE, DEVmax | mean of \|e\|, mean of e², max \|e\| over the period | the same |

2. **Settings against the official BSM1 code** (bsm2-python):
   * tank volumes (1000 and 1333 m³) and KLa of tanks 3–4 (240 1/d);
   * oxygen saturation 8 g/m³, used in the aeration-energy formula;
   * Q_a limit = 5 × Q_in0;
   * influent files: 14 days at 15 min.
3. **Re-runs.** Eleven stored runs, drawn from every result file (study,
   held-out test 1, held-out test 2, with and without sensors, learned and
   PID controllers), were re-run in fresh processes with today's code. Every
   stored value matched to machine precision, including the failure times.
4. **Independent metric computation.** Three runs were repeated reading
   S_O,5 straight from the plant state, without the project's `evaluate()`.
   IAE, ISE, DEVmax and aeration energy all matched.
5. **Headline numbers.** Every percentage quoted in the README and the
   documents (test week, held-out questions, failure counts) was recomputed
   from the raw run files, and all agree after rounding.
6. **Data isolation.**
   * Every tuning and training script uses dry days 0–7 only.
   * The SAC-IL training evaluations contain exactly 7 days of steps.
   * The held-out noise seeds (0–2, 10–12) differ from each other and from
     the tuning seeds (100, 101).

## What the audit cannot remove (limits to state in any paper)

* **The comparison is against our reimplementation of the paper's PID.**
  Against the paper's Table II, our oxygen PID matches in dry (0.0114 vs
  0.0114) and storm (0.0118 vs 0.0117) weather but is 10 % higher in rain
  (0.0108 vs 0.0098). Our nitrate PID error is about half the paper's in
  every weather (dry 0.0064 vs 0.0145); the paper does not give enough
  detail to find out why. Every "%" is relative to our PID, not to the
  paper's printed numbers.
* **Set points.** The paper prints [1, 2]. We use [2, 1], the standard BSM1
  values, because the paper's figures and PID numbers only match those.
* **Choices that are ours, not the paper's:**
  * the failure rule (oxygen outside 0.3–4 mg/L);
  * the S_I correction of the rain and storm files;
  * the train-week / test-week split;
  * halving the particulates as a primary clarifier for the BSM2 influent;
  * sensor noise drawn every 45 s from a seeded generator (the official
    model reads a 1-minute noise file).
* **Statistics.** PID controllers are deterministic (one run per case).
  Learned controllers have 5 training runs, and there are 6 windows or 9
  noise runs per held-out condition. We report means, counts and
  "better in k of n", but no confidence intervals or significance tests.
* **Retraining.** Re-running the stored controllers is exact. Retraining
  them on another machine lands near the stored results, not exactly on
  them (floating-point order).
* **Run length.** A run is 26 879 steps (the last influent sample starts no
  new period), which is negligible.
* **Test-week reuse.** The study's test week was looked at before the
  round-2 fixes and before the tuned PID was added. Only the held-out tests
  were fixed before they were run.
