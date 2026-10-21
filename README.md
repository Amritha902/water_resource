# RL control of a wastewater treatment plant

We rebuilt the standard BSM1 plant in code, reimplemented two base papers on it,
and improved the RL agent of the second. Every number below comes from
`results/`; `docs/05_results.md` is generated from those files.

## The two base papers

| | **Paper 1: MAACC** (Wang et al., IEEE T-II 2026) | **Paper 2: floating DO** (Du et al., Sci China Tech Sci 2023) |
|---|---|---|
| **What it is** | Two RL agents (oxygen, nitrate) on top of a PID. The learned part starts at zero, so day one is plain PID. | One DDPG agent that lets oxygen float instead of holding 2 mg/L, to save aeration energy. |
| **Main claimed result** | Tighter tracking than other methods, robust to actuator noise, over 90% training success. | 5–7% less aeration energy than PID, effluent still within limits. |
| **How we re-obtained it** | Reimplemented the agents, PID gains, setpoints and noise protocol. 54 runs: 3 conditions × 3 weathers × 2 seeds. | Reimplemented environment, rewards, DDPG, PID and fuzzy comparators. All 3 weathers. |
| **What we got** | Under the paper's noise protocol, nitrate tracking beats PID by **+27%** (dry), **+21%** (rain), **+10%** (storm); oxygen within a few %. Clean setting: no change from PID. Part of the gain is what an anti-windup-tuned PID also gives. | Saving reproduced: dry **6.8%** (paper 7.24), rain **4.9%** (5.52), storm **5.0%** (6.06). |
| **Verdict** | **Partly reproduced** | **Reproduced** |

Not checked for paper 1: its table against seven other methods (we compared
with PID only) and its stability proof.

Neither paper runs exactly as published. Paper 1's critic update (eq. 15)
diverges, and our first actor did not follow eq. (8) (it drifted into its clip
and acted as a constant offset; fixed and tested). Paper 2's state is not
Markov, its 45 s decision step is too short for the ammonia response, and its
critic gradient has the wrong sign early on. Details: `docs/02`, `docs/04`,
`docs/REPRODUCTION.md`.

## What we found that the papers do not report

The energy saving is bought by breaking the ammonia limit far more often:

| weather | saving | time above 4 g N/m³ (DDPG-B) | PID |
|---|---|---|---|
| dry | 6.8% | 38.1% | 12.2% |
| rain | 4.9% | 35.2% | 19.6% |
| storm | 5.0% | 37.0% | 23.2% |

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
python -m pytest tests -q                  # 73 tests
python experiments/09_write_results.py     # regenerate docs/05_results.md
```

RL learners reproduce exactly at the benchmark's thread count (`--threads 2`);
with other counts they land elsewhere on the same energy/violation trade-off.

Docs: `docs/00_code_walkthrough.md` (modules), `01`–`02` paper 1, `03` our
contribution, `04` paper 2, `05` results, `07_viva_prep.md` (questions to
expect), `STATUS.md` (what is left).
