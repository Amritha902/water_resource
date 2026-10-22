# RL control of a wastewater treatment plant

We reimplemented two base papers on the official BSM1 wastewater benchmark
plant and compared our results with the papers' own tables. Every number
below comes from `results/`.

## The two base papers

| | **Paper 1: MAACC** (Wang et al., IEEE T-II 2026) | **Paper 2: floating DO** (Du et al., Sci China Tech Sci 2023) |
|---|---|---|
| **What it is** | Two RL agents (oxygen, nitrate) on top of a PID. The learned part starts at zero, so day one is plain PID. | One DDPG agent that lets oxygen float instead of holding 2 mg/L, to save aeration energy. |
| **Main claimed result** | Oxygen error 92–95% and nitrate error 71–76% below PID (Table II); still 86% / 69% under actuator noise (Table III); over 90% of 20 runs beat PID. | 5–7% less aeration energy than PID, effluent still within limits. |
| **Exact replication** | Official BSM1 plant (`bsm2-python`, a port of the IWA benchmark code) with the official dry and rain influent; algorithm exactly as printed with the Table I values; 20 runs per case. | Same official plant and influent; PID and fuzzy comparators; DDPG-A and DDPG-B, 3 seeds each, dry and rain. |
| **What we got** | Plant check: our PID is within ~30% of the paper's PID row in dry weather. **MAACC as printed equals PID** in every run (its learned term stays ~10⁻⁵); 0 of 180 runs beat PID. Larger initial weights only make it diverge. | PID and fuzzy: energy and effluent index within **1%** of the paper. DDPG-B saving: dry **10.6%**, rain **7.1%** (paper 7.24 / 5.52), with more ammonia than the paper. Seeds spread widely (dry 5.8–15.7%); one seed lands on the paper (dry: 3513 vs 3430 kWh/d, DO 1.01 vs 1.03). |
| **Verdict** | **Not reproducible from the paper as printed.** Its numbers need details the paper does not give. | **Comparators reproduced exactly; DDPG reproduced in direction, not in exact size** |

### Paper vs our reimplementation

![Paper 1: error reduction vs PID, paper vs reimplementation](docs/figures/readme_paper1_vs_ours.png)

![Paper 1 reimplemented: PID vs MAACC on the official plant](docs/figures/readme_paper1_tracking.png)

![Paper 2: energy saving vs PID, paper vs reimplementation](docs/figures/readme_paper2_vs_ours.png)

Orange = the paper's published numbers; green = our reimplementation on the
official plant. Paper 1 reimplemented exactly as printed: MAACC's learned
correction stays around 10⁻⁵, so its trajectory sits on top of the PID's.
Paper 2's DDPG uses three corrections (actuator setting in the state,
15-minute decisions, reward scaled by 1 − γ); run exactly as published (one
seed, dry) it ends at a similar 10.7% saving. Video of the runs:
`docs/video/results.mp4`. Redraw: `python experiments/19_readme_figures.py`.

What we got wrong earlier, and fixed: paper 1's Table I (all its
hyperparameters) is printed as an image, and we missed it at first. The
reimplementation uses Table I.

Where the papers do not run as written: paper 1's critic update (eq. 15)
grows without bound; paper 2's state is not Markov, and its 45 s decision step
is short for the ammonia response. Details: `docs/REPRODUCTION.md`.

## What we found that the papers do not report

The energy saving is bought by breaking the ammonia limit far more often
(official plant, evaluation week, mean of 3 seeds):

| weather | DDPG-B saving | time above 4 g N/m³, DDPG-B | PID |
|---|---|---|---|
| dry | 10.6% | 56.0% | 21.0% |
| rain | 7.1% | 56.9% | 33.5% |

## Our contribution: a better RL network

Four changes to paper 2's agent: **(1)** a 2-hour influent forecast as input,
**(2)** a risk-aware critic that optimises the worst 30% of outcomes,
**(3)** violation budgets via Lagrange multipliers instead of hand-tuned
weights, **(4)** a learned plant model for extra training data.

Same objective, arms differing only in the network; gap versus paper 2's agent
at equal violation rate (positive = cheaper):

| forecast only | risk-aware critic only | both |
|---|---|---|
| +19.5 kWh/d | +25.9 kWh/d | **+28.6 kWh/d** |

## Limits (read these)

- Everything is simulated. The plant and influent are rebuilt from the model
  definition, not the official files, so numbers match the papers in direction,
  not exactly.
- Each sweep point is a single seed, and our agent has more inputs than paper
  2's, so part of the gain could be capacity.
- Change (4) did not work: it saves training data but broke the ammonia limit
  (14% → 67% violations), and our attempted fix did not help.
- The forecaster's weather-class head is at chance on fresh scenarios; its
  95.8% validation accuracy was a leak from overlapping windows. The flow and
  ammonia forecasts themselves hold up on unseen data.
- Two of our own earlier claims were withdrawn after the full benchmark.

## Project website

`docs/` is a static site: an overview with a claim-by-claim verification
ledger, one interactive page per base paper, and one for the prediction
models. Open `docs/index.html` directly, or publish it via Settings → Pages →
branch `claude/water-resource-ml-prediction-680kno`, folder `/docs`. Data comes
from `docs/data/site-data.js`, built by `experiments/14_build_site_data.py`.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # 80 tests
python experiments/09_write_results.py     # regenerate docs/05_results.md
```

RL learners reproduce exactly at the benchmark's thread count (`--threads 2`);
with other counts they land elsewhere on the same energy/violation trade-off.

Docs: `docs/00_code_walkthrough.md` (modules), `01`–`02` paper 1, `03` our
contribution, `04` paper 2, `05` results, `07_viva_prep.md` (questions to
expect), `09_faculty_speech.md` (two-minute talk),
`10_paper_study_notes.md` (page-by-page notes on both papers), `STATUS.md` (what is left).
