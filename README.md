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
gain where the loop starts to oscillate, and applied unchanged to the other
weathers. Under paper 1's actuator noise the gain falls to −26 % (oxygen) and
−42 % (nitrate).

Paper 1's text gives set points [1, 2] for [S_O,5, S_NO,2]; its figures and the
PID numbers we reproduce correspond to [2, 1], which we use.

## Documents

- `docs/REPORT.md`: the full report, paper by paper, with every table.
- `docs/DATASET.md` (and `.pdf`): plant, influent data, correction made to the
  influent files, simulation protocol and every setting.
- `docs/figures/reproduced/`: the 24 redrawn figures; `docs/figures/ours/`:
  figures of our improved controller.
- `results/`: every number in the report; `published_paper{1,2,3}.json` holds
  the values typed in from the papers.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q           # 82 tests

# paper 1: as printed (20 runs per case) and retuned prior
python experiments/20_exact_paper1.py --seeds 20 --noise --critics paper --tag exact2
python experiments/20_exact_paper1.py --controllers PIDx3 --seeds 1 --weathers dry rain storm --tag ours_p1_clean
# paper 2
python experiments/06_aeration_benchmark.py --official --weathers dry rain storm --methods PID Fuzzy DDPG-A DDPG-B
python experiments/26_fig4_official.py
# paper 3
python experiments/23_ioql.py --seeds 3
python experiments/23_ioql.py --methods PID PIDx4 --tag ours_p3
# figures and report
python experiments/24_diagrams.py && python experiments/25_paper_figures.py && python experiments/27_ours_figures.py
python tools/build_report.py --md docs/REPORT.md
```

The influent files are downloaded on first use into `data/official/`.

## Code

| | |
|---|---|
| `src/wwtp/official/plant.py` | official BSM1 plant (bsm2-python 0.0.16), influent loading and correction |
| `src/wwtp/maacc/exact.py` | paper 1, MAACC as printed |
| `src/wwtp/rl/` | paper 2: environment, DDPG, rewards, comparators; constrained agent (`panda_rl.py`) |
| `src/wwtp/ioql/ioql.py` | paper 3, IOQL as printed |
| `experiments/20`, `23`–`27` | the runs and figures behind the report |
| `tools/` | report builders |

## Earlier work

Before the official plant was used, the plant was rebuilt from the model
definition (`src/wwtp/bsm1/`), and an influent forecaster, a digital twin and
an ammonium early-warning model were trained on it (`experiments/01`–`19`,
`21`–`22`). Those notes, the old website and the earlier talk and slides are
in `docs/archive/`. Their numbers come from the rebuilt plant and are
superseded by the report above.

## Limits

- Everything is simulated, on one benchmark plant.
- The retuned prior is a fixed controller, not a learning method.
- Paper 1's nitrate-loop PID error is about half the paper's in every weather,
  and its noise experiment cannot be matched; the paper does not give enough
  detail to find out why.
