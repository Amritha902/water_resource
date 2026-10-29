# RL control of a wastewater treatment plant: three base papers on BSM1

Three published learning controllers for the Benchmark Simulation Model No. 1
(BSM1) are reimplemented exactly as printed, run on the official BSM1 plant
with the official dry, rain and storm influent, and compared with the papers'
own tables and figures. All 24 figures of the three papers are redrawn from
our runs.

| | Paper |
|---|---|
| 1 | D. Wang, X. Li, J. Ren, J. Qiao, "Multiagent Adaptive Critic Control With Expert Knowledge for Wastewater Treatment Plants", IEEE Trans. Ind. Informat., 2026, doi:10.1109/TII.2026.3659924 |
| 2 | S. L. Du, P. X. Chen, H. G. Han, J. F. Qiao, "Dissolved oxygen concentration control in wastewater treatment process based on reinforcement learning", Sci. China Tech. Sci. 66(9), 2549–2560, 2023, doi:10.1007/s11431-022-2403-8 |
| 3 | M. Zhao, D. Wang, J. Ren, J. Qiao, "Integrated Online Q-Learning Design for Wastewater Treatment Processes", IEEE Trans. Ind. Informat. 21(2), 1833–1842, 2025, doi:10.1109/TII.2024.3488790 |

## Results

| | Paper 1 (MAACC) | Paper 2 (DDPG, floating DO) | Paper 3 (IOQL) |
|---|---|---|---|
| Plant check: our PID vs the paper's | oxygen IAE 1.14·10⁻² vs 1.14·10⁻² (dry) | energy and effluent index within 1 % | IAE 1.13·10⁻² vs 1.16·10⁻² (dry) |
| Algorithm as printed | equals its PID prior (IAE within 1 %) in 80 of 80 runs | saves 9.4 / 6.5 / 8.0 % energy (dry / rain / storm; paper 7.2 / 5.5 / 6.1 %), with the ammonium limit broken 54–65 % of the week (PID 21–34 %) | unstable: stays at PID or the oxygen collapses |
| Verdict | not reproducible as printed | reproduced in direction | not reproducible as printed |
| Our improvement | PID prior retuned (gain ×3): oxygen error −63 to −66 %, nitrate −46 to −73 % | none: an agent held to the PID's compliance saves at most 2.5 % | PID prior retuned (gain ×4): oxygen error −64 to −74 % |

The retuned gains were chosen on dry weather only, with a margin below the
gain where the loop starts to oscillate (×5 for paper 1, ×6 for paper 3), and
applied unchanged to the other weathers. Under paper 1's actuator noise the
gain falls to −26 % (oxygen) and −42 % (nitrate).

Paper 1's text gives set points [1, 2] for [S_O,5, S_NO,2]; its figures and the
PID numbers we reproduce correspond to [2, 1], which we use.

## What is in the repository

```
README.md               this file
requirements.txt        Python packages
src/wwtp/               the library
  official/plant.py       official BSM1 plant (bsm2-python 0.0.16), influent loading and correction
  maacc/exact.py          paper 1, MAACC exactly as printed
  rl/                     paper 2: environment, DDPG, rewards, PID and fuzzy comparators,
                          constrained agent (panda_rl.py)
  ioql/ioql.py            paper 3, IOQL exactly as printed
  baselines/              incremental PID used as the prior
  bsm1/, forecast/, twin/, warning/, panda/   earlier work (see below)
experiments/            numbered scripts; 20, 23–27 produce the final results
tools/                  report builders (build_report.py writes docs/REPORT.md)
tests/                  82 tests
results/                every result file behind the report (JSON and CSV)
docs/
  REPORT.md               the full report: every table, figure notes, appendices
  DATASET.md, DATASET.pdf plant, influent data, the correction made to it,
                          simulation protocol and every setting
  figures/reproduced/     the 24 figures of the three papers, redrawn from our runs
  figures/ours/           figures of our improved controller
  comparison_notes.json   per-figure notes used by the report builder
  archive/                earlier notes, website, talk and slides (superseded)
artifacts/              trained models from the earlier work
.github/workflows/      GitHub build-provenance workflow
```

`results/published_paper{1,2,3}.json` hold the values typed in from the
papers' tables, so every comparison in the report is made in code.

## Install and test

Python 3.11.

```bash
pip install -r requirements.txt
python -m pytest tests -q        # 82 tests, about 1–2 minutes
```

The official influent files (dry, rain, storm) are downloaded on first use
into `data/official/` from the public repository davidfernand/BSM1_in_Python;
an internet connection is needed once.

## Reproduce every result

Each command writes `results/summary_<tag>.json` (and, where figures need
them, `results/traces_<tag>.json`). Run from the repository root.

**Paper 1 (MAACC)**

| Result file | Command |
|---|---|
| `summary_paper1_exact2.json`: as printed, 20 runs per case (Tables II, III; Figs. 4–9) | `python experiments/20_exact_paper1.py --seeds 20 --noise --critics paper --tag exact2` |
| `summary_paper1_exact_init_sweep.json`: 12 initial-weight scales | `python experiments/20_exact_paper1.py --init-sweep` |
| `summary_paper1_sweep_p1_dry.json`: gain sweep, dry | `python experiments/20_exact_paper1.py --controllers PIDx2 PIDx3 PIDx4 PIDx5 --seeds 1 --weathers dry --tag sweep_p1_dry` |
| `summary_paper1_ours_p1_clean.json`: retuned prior ×3 | `python experiments/20_exact_paper1.py --controllers PIDx3 --seeds 1 --weathers dry rain storm --tag ours_p1_clean` |
| `summary_paper1_ours_p1_noise.json`: retuned prior ×3, actuator noise | `python experiments/20_exact_paper1.py --controllers PIDx3 --seeds 20 --noise --weathers --tag ours_p1_noise` |

**Paper 2 (DDPG)**: `06_aeration_benchmark.py --official` (2 torch threads by default).

| Result file | Command |
|---|---|
| `summary_fig_p2_baselines.json`: PID and fuzzy | `python experiments/06_aeration_benchmark.py --official --weathers dry rain storm --methods PID Fuzzy --tag fig_p2_baselines` |
| `summary_fig_p2_<w>_A.json`, `_B.json`: DDPG-A/B with traces (Figs. 6–8) | `python experiments/06_aeration_benchmark.py --official --weathers <w> --methods DDPG-A --seeds 1 --tag fig_p2_<w>_A` (and `DDPG-B`, `_B`; storm with `--seeds 2`) |
| `summary_aeration_official_<w>_DDPG-A.json`, `_DDPG-B.json`: 3 more seeds, dry and rain | `python experiments/06_aeration_benchmark.py --official --weathers <w> --methods DDPG-A --seeds 3 --tag aeration_official_<w>_DDPG-A` |
| `summary_aeration_official_published.json`: exactly as published, one run | `python experiments/06_aeration_benchmark.py --official --published-env --weathers dry --methods DDPG-B --seeds 1 --tag aeration_official_published` |
| `summary_ours2_p2_<w>.json`: constrained agent at the PID's violation rates | `python experiments/06_aeration_benchmark.py --official --weathers <w> --methods PANDA-RL-pidbudget --seeds 2 --tag ours2_p2_<w>` |
| `summary_fig4_official.json`: effluent against DO set point (Fig. 4) | `python experiments/26_fig4_official.py` |

`<w>` is `dry`, `rain` or `storm`. The report's DDPG rows average all runs of
a weather (4 dry, 4 rain, 2 storm).

**Paper 3 (IOQL)**

| Result file | Command |
|---|---|
| `summary_ioql.json`: PID, IOQL, priors K1–K3, fixed and dynamic reference, 3 runs each (Tables I–IV; Figs. 4–7) | `python experiments/23_ioql.py --seeds 3` |
| `summary_sweep_p3_dry.json`: gain sweep, dry | `python experiments/23_ioql.py --methods PID PIDx2 PIDx3 PIDx4 PIDx5 PIDx6 --weathers dry --fixed-only --tag sweep_p3_dry` |
| `summary_ours_p3.json`: retuned prior ×4 | `python experiments/23_ioql.py --methods PID PIDx4 --tag ours_p3` |

**Figures and report**

```bash
python experiments/24_diagrams.py        # redrawn block diagrams
python experiments/25_paper_figures.py   # result figures (needs the traces_*.json above)
python experiments/27_ours_figures.py    # figures of our controller
python tools/comparison_notes.py         # docs/comparison_notes.json
python tools/build_report.py --md docs/REPORT.md
```

The per-run traces (`results/traces_*.json`) are large and are not kept in
git; the commands above write them again. The repository already contains
the redrawn figures, so the traces are only needed to redraw them.

## Earlier work

Before the official plant was used, the plant was rebuilt from the model
definition (`src/wwtp/bsm1/`), and an influent forecaster, a digital twin and
an ammonium early-warning model were trained on it (`experiments/01`–`19`,
`21`–`22`; trained models in `artifacts/`). Those notes, the old website and
the earlier talk and slides are in `docs/archive/`. Their numbers come from
the rebuilt plant and are superseded by the report above.

## Limits

- Everything is simulated, on one benchmark plant.
- The retuned prior is a fixed controller, not a learning method.
- Paper 1's nitrate-loop PID error is about half the paper's in every weather,
  and its noise experiment cannot be matched; the paper does not give enough
  detail to find out why.
- DDPG results vary widely between seeds (dry: 5.6–15.7 % saving), and RL
  training depends on the floating-point order, so a rerun on another machine
  or thread count lands near the stored values, not exactly on them. The PID runs
  (paper's prior and retuned) were rerun from a fresh copy of the repository
  and matched the stored values to every digit.

## Sources and licences

- BSM1 plant: bsm2-python 0.0.16 (FAU-EVT, BSD-3).
- Influent files: IWA BSM1 dry, rain and storm files, from
  davidfernand/BSM1_in_Python (commit 5dadb20, GPL-3), downloaded at run time.
- The papers themselves and their figures are not included.
