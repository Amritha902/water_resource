# Simulation data and settings used for the three base papers

## 1. Purpose

This note records the plant model, the influent data and every setting used to
reimplement the three base papers. Anyone with the repository and an internet
connection can regenerate every number and figure from it.

- Paper 1: D. Wang, X. Li, J. Ren, J. Qiao, "Multiagent Adaptive Critic Control
  With Expert Knowledge for Wastewater Treatment Plants", IEEE Trans. Ind.
  Informat., 2026, doi:10.1109/TII.2026.3659924.
- Paper 2: S. L. Du, P. X. Chen, H. G. Han, J. F. Qiao, "Dissolved oxygen
  concentration control in wastewater treatment process based on
  reinforcement learning", Sci. China Tech. Sci. 66(9), 2549–2560, 2023,
  doi:10.1007/s11431-022-2403-8.
- Paper 3: M. Zhao, D. Wang, J. Ren, J. Qiao, "Integrated Online Q-Learning
  Design for Wastewater Treatment Processes", IEEE Trans. Ind. Informat.
  21(2), 1833–1842, 2025, doi:10.1109/TII.2024.3488790.

All three papers use Benchmark Simulation Model No. 1 (BSM1) of the IWA Task
Group on Benchmarking of Control Strategies. None of them publishes code, so
the plant and the data below are the public benchmark, not the authors' own
files.

## 2. Plant model

The plant is the official BSM1 model as ported to Python by the `bsm2-python`
package, version 0.0.16 (FAU-EVT, BSD-3 licence). That package reproduces the
IWA MATLAB/Simulink implementation. We call it through
`src/wwtp/official/plant.py`, which exposes the two inputs the papers
manipulate and the two outputs they control.

| Item | Value |
|---|---|
| Biological model | ASM1, 13 state variables, 8 processes, 15 °C |
| Reactors | 5 in series: units 1–2 anoxic (1000 m³ each), units 3–5 aerated (1333 m³ each) |
| Secondary settler | 10 layers, area 1500 m², height 4 m, feed to layer 5; Takács double-exponential settling (v₀′ = 250, v₀ = 474 m/d, r_h = 5.76·10⁻⁴, r_p = 2.86·10⁻³ m³/g, f_ns = 2.28·10⁻³, X_t = 3000 g/m³) |
| Return sludge Q_r | 18 446 m³/d (fixed) |
| Wastage Q_w | 385 m³/d (fixed) |
| Internal recycle Q_a | 55 338 m³/d by default; manipulated in paper 1, range 0–92 230 m³/d |
| Oxygen transfer | KLa₁ = KLa₂ = 0, KLa₃ = KLa₄ = 240 d⁻¹; KLa₅ manipulated, range 0–240 d⁻¹, default 84 d⁻¹ |
| Oxygen saturation | 8 g/m³ |
| Kinetic parameters (15 °C) | μ_H 4.0 d⁻¹, K_S 10, K_OH 0.2, K_NO 0.5, b_H 0.3 d⁻¹, μ_A 0.5 d⁻¹, K_NH 1.0, K_OA 0.4, b_A 0.05 d⁻¹, η_g 0.8, k_a 0.05, k_h 3.0, K_X 0.1, η_h 0.8 |
| Stoichiometry | Y_H 0.67, Y_A 0.24, f_P 0.08, i_XB 0.08, i_XP 0.06, TSS/COD 0.75 |
| Controlled outputs | S_O,5 (dissolved oxygen in unit 5) and S_NO,2 (nitrate in unit 2) |

Each reactor is integrated over one control period with `scipy.integrate.odeint`
inside `bsm2-python`; the settler is integrated the same way. The control
period is 45 s for all three papers.

## 3. Influent data

### 3.1 Source

The three standard BSM1 influent files (dry, rain and storm weather) are the
IWA files `Inf_dry_2006`, `Inf_rain_2006` and `Inf_strm_2006`. We take them
in spreadsheet form from the public repository `davidfernand/BSM1_in_Python`,
commit 5dadb20 (GPL-3). They are downloaded on first use into `data/official/`
and are not copied into this repository.

Two checks were made against `bsm2-python`, which ships the dry and rain files
in its own format:

- The dry file is identical in every column and every row.
- The rain file has the same flows (to 0.003 m³/d) and the same
  concentrations, except for one column (S_I, below).

### 3.2 Format

Each file covers 14 days at 15-minute intervals, giving 1345 rows from
t = 0 to t = 14 d. The columns are time, the 13 ASM1 concentrations (S_I,
S_S, X_I, X_S, X_B,H, X_B,A, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK) and the
flow rate Q. The plant needs four more columns, which we add:

- total suspended solids, TSS = 0.75 (X_I + X_S + X_B,H + X_B,A + X_P);
- temperature, 15 °C;
- five dummy states, set to zero.

Between samples the influent is held constant.

### 3.3 Correction made

In the spreadsheets the inert soluble COD S_I is 30 g/m³ in every row of every
file. In the official rain file distributed with `bsm2-python`, S_I is instead
diluted by the rain water. It follows S_I = 30·Q_dry/Q, and we confirmed that
rule against all 1344 rows of that file to within 0.005 g/m³. We apply the same
rule to the rain and storm data; for dry weather it changes nothing.

S_I is biologically inert, so this correction affects only effluent COD and
the effluent quality index. It does not affect the controlled variables.

### 3.4 The three weathers

| Weather | Week | Mean Q (m³/d) | Min–max Q (m³/d) | Flow-weighted COD (g/m³) | COD load (kg/d) | NH₄ load (kg N/d) | TSS (g/m³) |
|---|---|---|---|---|---|---|---|
| Dry | 1 | 18 446 | 10 000–32 180 | 381 | 7032 | 582 | 211 |
| Dry | 2 | 18 451 | 10 000–32 180 | 381 | 7034 | 582 | 211 |
| Rain | 2 | 24 189 | 10 000–52 126 | 291 | 7034 | 582 | 161 |
| Storm | 2 | 21 044 | 10 000–60 000 | 352 | 7402 | 582 | 198 |

Week 1 is the same dry-weather week in all three files. In the rain file a long
rain period (days 8.4–10.4) raises the flow and dilutes the influent without
adding load. In the storm file two short storms (around days 8.8 and 11) bring
the flow up to 60 000 m³/d with a first flush of particulate matter, which is
why its COD load is higher.

## 4. Simulation protocol

1. **Initial state.** Every run starts from the steady state supplied with
   `bsm2-python`.
2. **Warm-up.** The plant is then run for 14 days on dry weather under the PID
   prior of the paper being reproduced:
   - paper 1: KLa₅ and Q_a, set points 2 and 1 mg/L;
   - paper 2: KLa₅ at a dissolved oxygen of 2 mg/L, Q_a fixed;
   - paper 3: KLa₅ at 2 mg/L, Q_a fixed.

   The plant state at the end of the warm-up is the start of every evaluation
   run, so every controller begins from the same plant.
3. **Evaluation.** Each evaluation run covers the 14 days of the weather file
   in 45 s control periods (26 880 periods).
4. **Paper 2's split.** Paper 2 learns on week 1 and is evaluated on week 2,
   as in that paper.
5. **Sensors.** Sensors are ideal: no delay and no measurement noise.
6. **Paper 1's noise test.** In the noise experiment of paper 1, zero-mean
   white noise with a standard deviation of 3 % of each actuator's upper bound
   (7.2 d⁻¹ on KLa₅, 2766.9 m³/d on Q_a) is added to the applied inputs, which
   are then limited to their ranges.
7. **Repeated runs.** Repeated runs differ only in the random initial weights
   of the networks and, in the noise case, the noise sequence. They use seeds
   0, 1, 2 and so on.

## 5. Settings for each paper

### 5.1 Paper 1, MAACC

Taken from the paper, including its Table I:

| Setting | Value |
|---|---|
| Controlled variables and set points | S_O,5 = 2 mg/L via KLa₅; S_NO,2 = 1 mg/L via Q_a (see note below) |
| Prior strategy | incremental PID, K_p = diag(100, 100 000), K_i = diag(20, 30 000), K_d = diag(10, 1000) |
| Utility weights | W = (1, 2), Y = (0.5, 0.05), R = (10⁻⁴, 10⁻⁸), P₁ = 2I, P₂ = I; raw units |
| Discount, learning rates | λ = 0.95, l_c = 0.01, l_a = 0.005 |
| Hidden units | critic 12 and 14, action network 10 and 12; bipolar sigmoid; only outer weights learn; action-network outer weights start at zero |
| Critic update | eq. (15)–(18) as printed (TD(0) and residual-gradient readings were also checked, with the earlier [1, 2] set points; no difference) |
| Runs | 20 per case (dry, rain, storm, dry with noise) |

A note on the set points: the text of paper 1 writes r = [1, 2] for
[S_O,5, S_NO,2], but its Figs. 4–6 and 8 show S_O,5 held at 2 mg/L and S_NO,2
at 1 mg/L, the standard BSM1 values. With [2, 1] our PID reproduces the
paper's PID row (dry weather oxygen IAE 1.140·10⁻² against 1.14·10⁻²; storm
1.176·10⁻² against 1.17·10⁻²). With [1, 2] it does not, so [2, 1] is used.

Not given in the paper and chosen by us:

- initial inner weights uniform on [−1, 1];
- initial critic outer weights uniform on [−0.1, 0.1].

A sweep of twelve other initial scales is reported in `docs/REPRODUCTION.md`.
Code: `src/wwtp/maacc/exact.py`, `experiments/20_exact_paper1.py`.

### 5.2 Paper 2, DDPG

| Setting | Value |
|---|---|
| Control | KLa₅ only, no set point; Q_a at 55 338 m³/d |
| State, action | [S_S,5, S_O,5, S_NH,5]; increment of KLa₅ |
| Rewards | A: −(max(S_NH,e − 4, 0) + 0.38·max(N_tot,e − 18, 0)); B: −(AE + 0.42·max(S_NH,e − 4, 0)) |
| Networks | actor 3-256-256-1 (ReLU, tanh), critic 4-256-256-1 |
| Hyperparameters | γ 0.99, mini-batch 256, buffer 30 000, τ 0.001, learning rates 10⁻⁴ (actor) and 3·10⁻⁴ (critic) |
| Comparators | PID on S_O,5 = 2 mg/L (K_p 200, K_i 15, K_d 2) and a fuzzy controller |

Three corrections are applied in the main runs, because the agent did not
learn as published on our first plant:

1. the actuator setting KLa₅/240 is added to the state;
2. each action is held for 15 minutes;
3. the reward is scaled by 1 − γ.

AE enters the reward as KLa₅/240, because the paper does not give its units.
Training uses 30 passes over the learning week, with exploration noise
annealed from 0.35 to 0.05 and a replay buffer of 80 000 (the paper's is
30 000). One run exactly as published is reported separately.

Code: `src/wwtp/rl/`, `experiments/06_aeration_benchmark.py --official`.

### 5.3 Paper 3, IOQL

| Setting | Value |
|---|---|
| Control | KLa₅ only; Q_a at 55 338 m³/d |
| Reference | fixed, r = 2 mg/L; or dynamic, eq. (33): 2 / 2.2 / 1.8 mg/L in two-day blocks |
| Prior policy | PID K_p 100, K_i 20, K_d 1; for the prior study K1 (80, 15, 10), K2 (100, 30, 10), K3 (120, 25, 10) with ω = 0.99^k |
| Weighting | ω = 1 in the main runs |
| Utility | G = e² + 0.01·u², with e = S_O,5 − r and u = ωα + β; γ = 1 |
| Networks | 15 hidden units each, tanh; only the outer weights learn |
| Learning | l_c = 0.01, l_a = 0.005; 50 critic and 60 action-network training steps per control period |
| Runs | 3 per case |

Not given in the paper and chosen by us:

- initial weights uniform on [−1, 1];
- the input applied in a period is the one after that period's updates;
- u is limited to 0–240 d⁻¹.

Code: `src/wwtp/ioql/ioql.py`, `experiments/23_ioql.py`.

## 6. Performance measures

For a run of N control periods, with tracking error e = S − r:

- IAE = (1/N) Σ |e|
- ISE = (1/N) Σ e²
- DEV_max = max |e|

These follow eq. (36) of paper 1 and eq. (32) of paper 3. Paper 2 uses:

- AE = (S_O,sat / 1800) Σ V_i KLa_i (kWh/d);
- the BSM1 effluent quality index EQ (kg pollution units/d);
- effluent concentrations averaged over the evaluation week.

## 7. Files

| File | Contents |
|---|---|
| `results/published_paper{1,2,3}.json` | numbers typed in from each paper's tables |
| `results/summary_paper1_exact2.json`, `traces_paper1_exact2.json` | paper 1 runs and trajectories |
| `results/summary_fig_p2_*.json`, `traces_fig_p2_*.json` | paper 2 runs and trajectories |
| `results/summary_ioql.json`, `traces_ioql.json` | paper 3 runs and trajectories |
| `docs/figures/reproduced/` | every figure of the three papers, redrawn from these runs |
| `docs/REPRODUCTION.md` | the full comparison with the papers |
