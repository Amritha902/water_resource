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
| **What we got** | Works under noise: nitrate tracking **+23–29%** vs PID. Clean: about level with PID on oxygen. | Saving reproduced: dry **6.8%** (paper 7.24), rain **4.9%** (5.52), storm **5.0%** (6.06). |
| **Verdict** | **Partly reproduced** | **Reproduced** |

Not checked for paper 1: its table against seven other methods (we compared
with PID only) and its stability proof.

Neither paper runs exactly as published. We fixed: paper 1's critic update
(eq. 15) diverges; paper 2's state is not Markov, its 45 s decision step is too
short for the ammonia response, and its critic gradient has the wrong sign
early on. Each fix has a test (`docs/02`, `docs/04`).

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
- Two of our own earlier claims were withdrawn after the full benchmark.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # 72 tests
python experiments/09_write_results.py     # regenerate docs/05_results.md
```

Docs: `docs/00_code_walkthrough.md` (modules), `01`–`02` paper 1, `03` our
contribution, `04` paper 2, `05` results, `STATUS.md` (what is left).
