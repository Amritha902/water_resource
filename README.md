# PANDA-MAACC — prediction-driven multiagent adaptive critic control for wastewater treatment plants

Extension of

> D. Wang, X. Li, J. Ren, J. Qiao, **"Multiagent Adaptive Critic Control With
> Expert Knowledge for Wastewater Treatment Plants"**, *IEEE Transactions on
> Industrial Informatics*, 2026. DOI 10.1109/TII.2026.3659924

The base paper is a control-theory contribution: it decomposes a BSM1 plant
into interconnected subsystems, puts one adaptive-critic agent on each, and
anchors the learning to an expert PID prior so the loop is safe from step
zero. This repository reproduces that algorithm and then adds the part the
paper leaves open — **machine learning that predicts what the plant is about
to face, instead of only reacting to what it has already felt.**

```
u_{k,i}  =  b_{k,i}   +   g_{k,i} · a_{k,i}   +   f_{k,i}
            expert        gated adaptive          anticipatory
            prior         critic term             feed-forward
            (paper)       (paper + forecast)      (new)
```

Setting `g ≡ 1, f ≡ 0` recovers the paper's eq. (7) exactly, so the extension
is a strict generalisation, not a replacement.

---

## The four learned components

| # | Component | What it is | Where |
|---|---|---|---|
| 1 | **Probabilistic influent forecaster** | dilated causal TCN over 24 h of inlet history → 10/50/90 % quantiles of flow, ammonium and readily-biodegradable COD for the next 2 h, plus a dry/rain/storm regime head | `src/wwtp/forecast/` |
| 2 | **Neural digital twin** | GRU residual model of the closed loop on a 6-minute grid; its recurrent state stands in for the unknown interconnection term `G_i(s_k)` of eq. (4), and it predicts effluent ammonium too | `src/wwtp/twin/` |
| 3 | **Anticipatory feed-forward** | the move is found by *differentiating the twin* over a 48-minute horizon driven by the forecast — closed-loop aware and offset-free (see below) | `src/wwtp/panda/` |
| 4 | **Forecast-conditioned agents + uncertainty gate** | the action and critic networks are conditioned on a 6-D forecast context; the adaptive term is scaled by `1/(1+κ·(q90−q10)/q50)`, so low forecast confidence means falling back to the expert prior | `src/wwtp/panda/` |

Full rationale, including the two design corrections that were needed to make
the feed-forward work at all, is in [`docs/03_novelty.md`](docs/03_novelty.md).

## Layout

```
src/wwtp/
  bsm1/        ASM1 biokinetics, Takács clarifier, five-reactor plant, influent generator
  baselines/   incremental PID (the paper's expert prior)
  maacc/       faithful reproduction of the paper's algorithm
  forecast/    probabilistic influent forecaster + online predictor
  twin/        excitation data collection + neural digital twin
  panda/       the PANDA-MAACC controller
  envs.py      closed-loop harness;  metrics.py  IAE/ISE/DEVmax, BSM1 EQ and OCI
experiments/   01 train forecaster · 02 train twin · 03 run the control benchmark
docs/          paper summary · reproduction notes · the contribution
tests/         28 tests: plant physics, critic convergence, learned-model sanity
artifacts/     trained forecaster and twin checkpoints (≈1 MB, committed)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # ~6 min, mostly simulation

# the checkpoints in artifacts/ are already trained; to rebuild them:
python experiments/01_train_forecaster.py      # ~10 min CPU
python experiments/02_train_twin.py            # ~40 min CPU (data collection dominates)

python experiments/03_run_control.py --seeds 3 # closed-loop benchmark
```

## Status

**This is work in progress.** The simulator, the reproduction, both learned
models and the controller are complete and tested; the final benchmark sweep
and the figures are not. Read [`STATUS.md`](STATUS.md) before continuing — it
lists exactly what is done, what is not, what the numbers currently say, and
the next concrete step.
