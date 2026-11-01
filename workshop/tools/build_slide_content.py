"""Build workshop/slides/content.js: the notebook players and code tours of the deck.

* Notebook players: selected cells of the three workshop notebooks, with the
  outputs of a real run (the notebooks are executed here), figures saved to
  workshop/slides/img/.
* Code tours: functions of the project, cut from the source files by name
  (ast), so the slides always show the code as it is in the repository.

    python workshop/tools/build_slide_content.py          # from the repository root, about 1 minute
"""

from __future__ import annotations

import ast
import base64
import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "workshop"
NBDIR, SLIDES = WS / "notebooks", WS / "slides"
IMG = SLIDES / "img"

# ───────────────────────────── notebook players
# (notebook, first characters of the cell, concept title, concept text)
PLAYERS = {
    "collect": ("01_data_collect_clean_prepare", [
        ('REV = "5dadb', "Pin the source",
         "The URL contains a commit hash, so it always returns the same bytes. The file is downloaded once and then read from disk."),
        ('raw = pd.read_excel(paths["dry"], header=None)', "Look before you parse",
         "Row 0 holds names, not numbers. One text row is enough to turn every column into text (object). This is why the repository keeps only rows whose first cell is a number."),
        ("for w, d in df.items():", "The standard checks",
         "Time step, missing values, duplicates, impossible values. Run them on every dataset, every time. Here all files pass."),
        ('week1 = df["dry"].t < 7', "Check, don't assume",
         "Week 1 is identical in all three files. A careless split would test on training data."),
        ("def clean(x: pd.DataFrame", "Detect, then repair",
         "Each problem has its own rule: sort and de-duplicate, put rows on the full grid, impossible values to NaN, spikes against a rolling median, a stuck sensor by zero variation, then fill only short gaps."),
        ('err = (fixed[["Q", "S_NH"]]', "Measure the repair",
         "Here we know the truth, so we can measure how good the cleaning was. On real data, keep a flag column so you know which values were repaired."),
    ]),
    "prepare": ("01_data_collect_clean_prepare", [
        ("for w in FILES:", "Physics finds what statistics can't",
         "S_I stayed at 30 while rain doubled the flow. Rain dilutes it: S_I = 30·Q_dry/Q, the correction in plant.py:89. TSS is added because the plant model needs it."),
        ("def rows_of(weather, t_lo, t_hi):", "Scale with training data only",
         "Mean and std come from the training rows only. The last line shows what the numbers would be if the test data leaked in."),
        ("def scale(x):", "Windows and shapes",
         "Each example: 96 past steps × 3 channels → the next 8 flows. The assert checks that the input ends before the target starts."),
        ('X, Y, T = data["val"]', "One example, in physical units",
         "Blue is what the model sees, orange is what it must predict. Storm 1 starts right after this window."),
    ]),
    "model": ("02_choose_and_train_a_model", [
        ('d = np.load(DATA / "prepared.npz")', "Shapes first",
         "(N, 96, 3) → (N, 8): a sequence in, 8 numbers out. That makes this a regression with a sequence input."),
        ("def persistence(x):", "Baselines need no training",
         "Persistence and 'same time yesterday' cost nothing. Every model must beat them."),
        ("flat = lambda x: x.reshape(len(x), -1)", "A linear model",
         "288 inputs × 8 outputs + 8 biases = 2 312 parameters. A strong baseline on small data."),
        ("def mlp(hidden=(64, 64), dropout=0.0):", "Counting parameters",
         "Each Linear layer has in × out weights and out biases. 23 176 in total, for about 1 900 examples."),
        ('x1 = torch.from_numpy(X["val"][:1])', "Layer by layer",
         "The same forward pass as the animated slide, with real numbers: the shape after every layer, and how many neurons ReLU switched off."),
        ("class CausalBlock(nn.Module):", "A causal convolution",
         "Padding on the left only, so no output can see the future. The two checks prove it: change the newest sample and the output moves."),
        ('h = torch.from_numpy(X["val"][:1])', "Shapes inside the CNN",
         "Time stays 96 long through every block; the output is read at the last step, which is 'now'."),
    ]),
    "train": ("02_choose_and_train_a_model", [
        ("def loader(split, batch=128", "The training loop",
         "Batch → forward → loss → backward → update. Then validate without gradients and keep the best epoch. The same loop as forecast/train.py."),
        ("net = mlp()", "Train and watch both curves",
         "Training loss keeps falling; validation flattens. The dashed line is the epoch we keep."),
        ("big = mlp(hidden=(512, 512))", "Overfitting on purpose",
         "415 000 parameters, 200 examples: training loss goes to almost zero, validation goes up. The network memorised."),
        ("test_models = {", "The test set, once",
         "On the unseen storm the plain linear model wins at 2 h. With little data, simple wins."),
        ('torch.save({"state_dict": cnn.state_dict()', "Save everything the model needs",
         "Weights, scaler and settings in one file. Then reload it and forecast from raw sensor values, as the plant would."),
    ]),
    "rl": ("03_reinforcement_learning_in_brief", [
        ("class OxygenTank:", "An environment",
         "reset() and step(action) → next state, reward, done, the interface of rl/env.py. The reward: be at 2 mg/L, don't waste air."),
        ("def pid_policy(o, kp=80.0, ki=25.0):", "The expert",
         "An incremental PID: no data, no training. The baseline every learner must beat."),
        ("obs_list, act_list = [], []", "Imitation = supervised learning",
         "Record the PID's (state, action) pairs and fit a network: the first stage of the project's SAC-IL. The copy is as good as the teacher, and no better."),
        ("E_BINS = np.array(", "Q-learning from reward alone",
         "A table Q[state, action], updated with the Bellman rule. Early episodes are random and terrible: on a plant those would be bad days."),
        ("agents = [Qtab]", "Better than what?",
         "Q-learning beats the plain PID on three seeds and an unseen rain day. The PID with one line of physics does as well, with no training."),
        ("Q_cheap, _, _ = q_learning(energy_weight=5.0", "The reward is the specification",
         "Make air expensive and the agent switches it off. It does what the reward says, not what you meant."),
    ]),
}

# ───────────────────────────── code tours
# file, function (Class.method or name), [(first-line text, last-line text, title, explanation)]
TOURS = {
    "data": [
        ("src/wwtp/official/plant.py", "load_influent", "Official influent, cleaned", [
            ('if weather.startswith("bsm2:")', "return bsm2_window", "Held-out data",
             "A different influent (BSM2, 609 days) for the pre-registered held-out test. The same function serves both, so the test uses the same cleaning."),
            ("d = _official_table(weather)", "dry = _official_table", "Raw table, cached",
             "_official_table downloads the pinned spreadsheet once, keeps the numeric rows and caches them as .npy."),
            ("out = np.zeros((len(d), 22))", "out[:, 1:14] = d[:, 1:14]", "The layout the plant expects",
             "bsm2-python wants 22 columns: time, 13 concentrations, TSS, flow, temperature and five dummy states."),
            ("out[:, 1] = 30.0 * dry[:, 14] / d[:, 14]", "out[:, 1] = 30.0", "The physics fix",
             "Rain dilutes the inert COD: S_I = 30·Q_dry/Q. It was checked against the official rain file to 0.005 g/m³ and is documented in DATASET.md §3.3."),
            ("out[:, 14] = 0.75", "out[:, 16] = 15.0", "Derived and constant columns",
             "TSS from the particulate components; flow; 15 °C, the temperature of BSM1."),
        ]),
    ],
    "rlcode": [
        ("src/wwtp/rl/rewards.py", "reward_ae_eq", "Paper 2's reward", [
            ('"""Equation (23)', '"""Equation (23)', "The specification",
             "Everything the agent will ever want is in these two lines: use little air (aeration_fraction), and pay 0.42 per mg/L of ammonium above the limit of 4."),
            ("return -(b1", "+ b2", "What it leaves out",
             "Nothing stops long violations as long as they are small. Paper 2's agents broke the ammonium limit 54–65 % of the week (the PID: 21–34 %)."),
        ]),
        ("src/wwtp/rl/ddpg.py", "DDPGAgent.update", "DDPG: one update", [
            ("if self.buffer.size < max", "return {}", "Wait for data",
             "No learning until the replay buffer holds enough transitions."),
            ("o, a, r, o2, d = (t.to(self.device)", "self.buffer.sample", "A random batch from the past",
             "Replay: old experience is reused many times, in random order."),
            ("with torch.no_grad():", "torch.cat([o2, self.actor_target(o2)]", "The Bellman target",
             "y = r + γ·Q'(s', μ'(s')), computed with the slow target networks so that the target does not move every step."),
            ("q = self.critic(torch.cat([o, a]", "self.opt_critic.step()", "The critic is a regression",
             "Mean squared error between Q(s, a) and the target: exactly a supervised loss."),
            ("pre = self.actor.pre_activation(o)", "self.opt_actor.step()", "The actor climbs the critic",
             "Maximise Q(s, μ(s)). The small penalty keeps tanh from saturating at an actuator limit."),
            ("self._soft_update(self.actor", "self._soft_update(self.critic", "Soft target updates",
             "Targets move 0.1 % towards the online networks each step (τ = 0.001)."),
        ]),
        ("src/wwtp/sacil/sacil.py", "SACIL.update", "SAC-IL: one update", [
            ("o, a, r, o2, d = self._mixed_batch()", "self._critic_step", "Mixed batch, twin critics",
             "A quarter of each batch comes from the expert demonstrations. The critic step uses two critics and takes the smaller value."),
            ("alpha = self.log_alpha.exp().detach()", "loss_a = (alpha * logp", "Soft actor-critic",
             "Maximise Q plus an entropy bonus: keep exploring a little."),
            ("w = max(self.cfg.bc_floor", "loss_a = loss_a + w", "Imitation never fades out",
             "The behaviour-cloning term starts strong and never drops below a floor. Without it the policy drifted and the loops became unstable."),
            ("loss_al = -(self.log_alpha", "self.opt_al.step()", "Automatic temperature",
             "α is tuned so that the policy keeps a target amount of randomness."),
        ]),
    ],
    "papers": [
        ("src/wwtp/maacc/exact.py", "ExactAgent.step", "Paper 1: MAACC as printed", [
            ("zi = z[self.i]", "a = self.a + da", "Action network, incremental",
             "Δa = α·δ(η·z), then a = a + Δa (eqs. 8, 14). α starts at zero, so the controller starts as its PID."),
            ("x = np.array([zi, da, a])", "u_k = self.utility(z, da, a)", "Critic and utility",
             "Q = β·ϑ(θ·[z, Δa, a]); the utility in raw units with Table I's weights (R = 1e-4, 1e-8)."),
            ("if self.prev is not None:", "q = float(self.beta @ vt)", "Critic update, as printed",
             "e = U + λQ_k − Q_{k−1}; the gradient is taken through Q_k (eq. 18). On a test problem this literal update diverges while the TD(0) reading converges (tests/test_critic_update.py). On the plant, all three readings stayed at the PID."),
            ("dq_da = float(", "self.alpha = self.alpha - LA", "Why it never learns",
             "The actor's step is l_a·Q·∂Q/∂a·δ. With the printed weights the signal is about 10⁻⁴, so α hardly moves. In 80 of 80 runs, MAACC equals its PID."),
        ]),
        ("src/wwtp/ioql/ioql.py", "IOQL.__call__", "Paper 3: IOQL as printed", [
            ("e = x - r", "u = w * alpha + beta", "Prior plus learned term",
             "u = ω·PID + β(e): the PID is the prior policy; β is the action network."),
            ("for _ in range(TC):", "self.V = self.V - LC", "50 critic steps per period",
             "Eq. (9), repeated T_c = 50 times on the same sample."),
            ("for _ in range(TA):", "self.A = self.A - LA", "60 actor steps per period",
             "Eqs. (11)–(13). Repeating so many steps on one sample is what makes it unstable: it stays at the PID or the oxygen collapses."),
            ("self.k += 1", "return float(np.clip(u, 0.0, 240.0))", "Clip to the actuator",
             "KLa,5 is limited to 0–240 1/d, a choice the paper does not state."),
        ]),
    ],
    "ours": [
        ("src/wwtp/sacil/ilc.py", "DailyILC._learn", "Daily ILC: the nightly update", [
            ("new = self.f.copy()", "kern = np.ones", "Once a day",
             "Called at midnight with the PID's input recorded over the whole day."),
            ("dev = self._u[:, i] - self._u[:, i].mean()", "dev = self._u", "What the PID had to do",
             "The PID's daily deviation from its mean: the part of its work that repeats every day."),
            ("upd = self.f[:, i] + self.beta[i] * np.roll(dev, -self.lead[i])", "upd =", "Take over a share, earlier",
             "The feedforward takes over a share β of it, applied lead steps earlier, ahead of the load peak."),
            ("pad = np.concatenate", "self.f = new", "Smooth over the day",
             "A moving average (the day is periodic) and a forgetting factor keep it robust to odd days."),
        ]),
        ("src/wwtp/sacil/sacil.py", "Paper1Env.step", "One control period", [
            ("a = np.clip(np.asarray(a, float), -1.0, 1.0)", "a = np.clip", "Bounded learner",
             "The learned action is limited to [−1, 1] × RES_SCALE per period."),
            ("elif self.fallback_gain is not None:", "self.fallback_gain = None", "Safety fallback",
             "When the gate fires, this period is the demonstrator PID alone."),
            ("du = expert_increment(self.e, self.e1, self.e2, self.base_gain, self.terms) + self.mask", "du = expert_increment(self.e, self.e1, self.e2, self.base_gain, self.terms) +", "u = PID + learned",
             "Paper 1's structure kept: the PID increment plus the learned correction."),
            ("u_apply = self.u if self.ilc is None", "u_apply = np.clip(u_apply + self.rng", "Feedforward and noise",
             "The ILC's daily profile is added; paper 1's actuator-noise test adds noise here."),
            ("if self.ilc is not None:", "self.u = np.clip(self.u - (self.ilc.ff", "Bumpless at midnight",
             "When the new profile takes over, the same amount is taken out of the PID's state, so the total input does not jump."),
            ("r = -float(np.sum(np.abs(self.e) * self.w))", "return self._obs()", "Reward and failure",
             "Weighted tracking error plus a small action cost; leaving 0.3–4 mg/L ends the episode with a penalty."),
        ]),
        ("experiments/33_paper1_study.py", "sac_policy", "The safety gate", [
            ("lo, hi = training_range(ctrl)", "stats = {", "The range seen in training",
             "The observations the demonstrator produced on dry week 1."),
            ("if gated:", "need = abs(", "Two reasons to switch off",
             "An input more than 10 % outside the training range, or a correction larger than the learner may give."),
            ("if ood or need:", "return np.zeros(2)", "Fall back to the expert",
             "The learned part is switched off for this period; PID ×3 + ILC act alone."),
            ("o = np.clip(o, lo, hi)", "return net(", "Otherwise: clip and act",
             "Inside the range the network acts on clipped inputs. On the held-out test the gate switched the learned part off under new conditions."),
        ]),
        ("experiments/28_sac_il_paper1.py", "main", "Training script, start to end", [
            ("demo_cost = run_expert(train_env, agent, gain=a.demo_gain)", "agent.pretrain_critic(3000)", "1 · Expert, imitation, critic warm-up",
             "Record the demonstrator on dry week 1, clone it (4 000 steps), then warm up the critic."),
            ("cost = lambda ev:", "best = (cost(bc)", "Selection rule, fixed in advance",
             "Policies are compared on the training week with both loops scaled by the retuned prior's error."),
            ("for ep in range(1, a.episodes + 1):", "best = (score", "2 · Soft actor-critic",
             "Explore, store, update every step; evaluate after each 7-day episode and keep the best."),
            ("agent.actor.load_state_dict(best[1])", "torch.save(agent.actor.state_dict()", "Keep and save the best",
             "The kept policy is saved to artifacts/ and frozen."),
            ('for w in ("dry", "rain", "storm"):', "rows.append(dict(weather=w", "3 · Evaluate on every weather",
             "Rain and storm were never seen in training."),
        ]),
    ],
}

# experiment map (pipeline slide): written here so README numbers and commands stay in one place
PIPELINE = [
    {"id": "data", "col": 0, "title": "Official plant + data", "file": "src/wwtp/official/plant.py",
     "cmd": "(imported by every experiment)", "out": "data/official/*.npy",
     "what": "BSM1 from bsm2-python 0.0.16; IWA dry, rain, storm influent, pinned, with the S_I fix."},
    {"id": "e20", "col": 1, "title": "Paper 1 as printed", "file": "experiments/20_exact_paper1.py",
     "cmd": "python experiments/20_exact_paper1.py --seeds 20 --noise --critics paper --tag exact2",
     "out": "results/summary_paper1_exact2.json",
     "what": "MAACC exactly as printed, 20 runs per case: equals its PID in 80 of 80 runs."},
    {"id": "e06", "col": 1, "title": "Paper 2 (DDPG)", "file": "experiments/06_aeration_benchmark.py",
     "cmd": "python experiments/06_aeration_benchmark.py --official --weathers dry --methods DDPG-A --seeds 3",
     "out": "results/summary_aeration_official_*.json",
     "what": "Saves 9.4 / 6.5 / 8.0 % energy, but breaks the ammonium limit 54–65 % of the week."},
    {"id": "e23", "col": 1, "title": "Paper 3 (IOQL)", "file": "experiments/23_ioql.py",
     "cmd": "python experiments/23_ioql.py --seeds 3", "out": "results/summary_ioql.json",
     "what": "Unstable as printed: stays at the PID or the oxygen collapses."},
    {"id": "e32", "col": 2, "title": "Tuning (dry week 1)", "file": "experiments/32_paper1_tuning.py",
     "cmd": "python experiments/32_paper1_tuning.py", "out": "results/paper1_tuning.json",
     "what": "PID gain ×3 (unstable at ×5) and the daily ILC, chosen on dry week 1 only."},
    {"id": "e28", "col": 2, "title": "SAC-IL training", "file": "experiments/28_sac_il_paper1.py",
     "cmd": "python experiments/28_sac_il_paper1.py --seed 0 --ilc --balanced --oxygen-only --tag sacil_v3",
     "out": "artifacts/sacil_v3_actor_seed*.pt", "what": "Expert → imitation → SAC, 5 seeds, best policy kept."},
    {"id": "e38", "col": 2, "title": "Tuned PID baseline", "file": "experiments/38_tuned_pid.py",
     "cmd": "python experiments/38_tuned_pid.py", "out": "results/paper1_tuned_pid.json",
     "what": "P and I tuned per loop with a stability margin: oxygen P ×2, I ×8."},
    {"id": "e39", "col": 2, "title": "Repaired MAACC", "file": "experiments/39_maacc_repaired.py",
     "cmd": "python experiments/39_maacc_repaired.py", "out": "artifacts/maacc_rep_*_actor_seed*.pt",
     "what": "Paper 1's learner with replay, targets and twin critics: it does learn (−85 % dry)."},
    {"id": "e33", "col": 3, "title": "Study: test week", "file": "experiments/33_paper1_study.py",
     "cmd": "python experiments/33_paper1_study.py", "out": "results/summary_paper1_study.json",
     "what": "All controllers on week 2. Ours −87 / −82 / −74 %; tuned PID + ILC −90 / −89 / −80 %."},
    {"id": "e40", "col": 3, "title": "Held-out test", "file": "experiments/40_paper1_heldout.py",
     "cmd": "python experiments/40_paper1_heldout.py", "out": "results/summary_paper1_heldout.json",
     "what": "Protocol committed first. New influent, official sensors: high-gain controllers fail under sensor noise."},
    {"id": "docs", "col": 4, "title": "Report", "file": "docs/REPORT.md, docs/PAPER1_*.md",
     "cmd": "python tools/build_report.py --md docs/REPORT.md", "out": "docs/",
     "what": "Every table is computed from results/; published values typed into results/published_paper*.json."},
]


# ───────────────────────────── code maps: the same steps, highlighted in the same colour, in every place
# panel source: ("file", path, first-line text, last-line text) | ("func", path, qualname, first, last)
#               | ("cells", notebook, [first characters of each cell])
# marks: step index -> (first-line text, last-line text)
CODEMAPS = {
    "lesson1": {
        "steps": ["the knobs", "1 · guess", "2 · how wrong", "3 · which way", "4 · turn the knobs"],
        "panels": [
            ("Hands-on script", ("file", "workshop/examples/lesson1_knobs.py", "w, b = 0.0, 0.0", "b = b - lr * grad_b"),
             {0: ("w, b = 0.0, 0.0", "lr = 0.5"), 1: ("pred = w * x + b", "pred = w * x + b"), 2: ("loss = np.mean", "loss = np.mean"),
              3: ("grad_w = np.mean", "grad_b = np.mean"), 4: ("w = w - lr * grad_w", "b = b - lr * grad_b")}),
            ("Notebook 2 · fit()", ("cells", "02_choose_and_train_a_model", ["def loader(split"]),
             {0: ("opt = torch.optim.AdamW(model.parameters()", "opt = torch.optim.AdamW"), 1: ("pred = model(xb)", "pred = model(xb)"),
              2: ("loss = loss_fn(pred, yb)", "loss = loss_fn(pred, yb)"), 3: ("loss.backward()", "loss.backward()"),
              4: ("opt.step(); sched.step()", "opt.step(); sched.step()")}),
            ("Project · influent forecaster", ("func", "src/wwtp/forecast/train.py", "train_forecaster", "model = InfluentForecaster()", "sched.step()"),
             {0: ("model = InfluentForecaster()", "opt = torch.optim.AdamW"), 1: ("q, logits, _ = model(xb)", "q, logits, _ = model(xb)"),
              2: ("loss = (quantile_loss", "regime_weight *"), 3: ("loss.backward()", "loss.backward()"), 4: ("opt.step()", "opt.step()")}),
            ("Project · RL critic (DDPG)", ("func", "src/wwtp/rl/ddpg.py", "DDPGAgent.update", "with torch.no_grad():", "self.opt_critic.step()"),
             {1: ("q = self.critic(torch.cat([o, a]", "q = self.critic"), 2: ("loss_c = ((q - y) ** 2).mean()", "loss_c ="),
              3: ("loss_c.backward()", "loss_c.backward()"), 4: ("self.opt_critic.step()", "self.opt_critic.step()")}),
        ]},
    "lesson2": {
        "steps": ["1 · download", "2 · read", "3 · check", "4 · fix"],
        "panels": [
            ("Hands-on script", ("file", "workshop/examples/lesson2_load_and_clean.py", "# ── 1 DOWNLOAD", 'print("         S_I after the dilution fix'),
             {0: ("path = DATA /", "urllib.request.urlretrieve(URL + dry_path.name"), 1: ("raw = pd.read_excel", "d = d.astype(float)"),
              2: ("step_min = np.unique", "S_I values in the whole file"), 3: ("x = d.copy()", 'd["S_I"] = 30.0')}),
            ("Notebook 1 · parts A and B", ("cells", "01_data_collect_clean_prepare",
                                            ['REV = "5dadb', 'raw = pd.read_excel(paths["dry"], header=None)', "COLS = [", "for w, d in df.items():", "def clean(x: pd.DataFrame"]),
             {0: ("def download(weather", "urllib.request.urlretrieve"), 1: ("raw = pd.read_excel", "return d.astype(float)"),
              2: ("for w, d in df.items():", "negative values"), 3: ("def clean(x: pd.DataFrame", "return x, pd.concat(flags")}),
            ("Project · download and read", ("func", "src/wwtp/official/plant.py", "_official_table", "cache = os.path.abspath", "return arr"),
             {0: ("if not os.path.exists(xlsx):", "urllib.request.urlretrieve"), 1: ("import openpyxl", "arr = np.array(rows")}),
            ("Project · checks (tests) and the physics fix", ("multi", [
                ("func", "tests/test_official.py", "test_rain_file_is_repaired", None, None),
                ("func", "src/wwtp/official/plant.py", "load_influent", "d = _official_table(weather)", "return out")]),
             {2: ("def test_rain_file_is_repaired", "assert np.allclose(r[:672"), 3: ("out[:, 1] = 30.0", "out[:, 14] = 0.75")}),
        ]},
}


def execute(name: str):
    nb = nbformat.read(NBDIR / f"{name}.ipynb", 4)
    print("running", name, flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NBDIR)}}).execute()
    return nb


def outputs(cell, tag: str) -> dict:
    text, imgs = [], []
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            text.append(o.text)
        elif o.output_type in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                f = IMG / f"{tag}_{len(imgs)}.png"
                f.write_bytes(base64.b64decode(data["image/png"]))
                imgs.append(f"img/{f.name}")
            elif "text/plain" in data and not data["text/plain"].startswith(("<matplotlib", "Text(", "<Axes")):
                text.append(data["text/plain"] + "\n")
        elif o.output_type == "error":
            raise RuntimeError(f"{tag}: {o.ename}: {o.evalue}")
    t = re.sub(r"\n{2,}", "\n", "".join(text)).strip("\n")
    lines = t.splitlines()
    if len(lines) > 34:
        t = "\n".join(lines[:16] + [f"… ({len(lines) - 30} more lines) …"] + lines[-14:])
    return {"text": t, "img": imgs}


def function_source(path: Path, qual: str) -> tuple[list[str], int]:
    src = path.read_text(encoding="utf8")
    tree = ast.parse(src)
    parts = qual.split(".")
    nodes = tree.body
    node = None
    for p in parts:
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p)
        nodes = node.body
    lines = src.splitlines()[node.lineno - 1: node.end_lineno]
    return lines, node.lineno


def marks(lines, specs, where):
    out, pos = [], 0
    for first, last, title, text in specs:
        a = next((i for i in range(pos, len(lines)) if first in lines[i]), None)
        if a is None:
            raise ValueError(f"{where}: no line with {first!r}")
        b = next((i for i in range(a, len(lines)) if last in lines[i]), None)
        if b is None:
            raise ValueError(f"{where}: no line with {last!r} after {first!r}")
        out.append({"from": a, "to": b, "title": title, "text": text})
        pos = a + 1
    return out


def lines_between(lines, first, last):
    a = 0 if first is None else next(i for i, l in enumerate(lines) if first in l)
    b = len(lines) - 1 if last is None else next(i for i in range(a, len(lines)) if last in lines[i])
    return lines[a:b + 1]


def panel_lines(src, executed):
    kind = src[0]
    if kind == "file":
        return lines_between((ROOT / src[1]).read_text(encoding="utf8").splitlines(), src[2], src[3]), src[1]
    if kind == "func":
        lines, first = function_source(ROOT / src[1], src[2])
        return lines_between(lines, src[3], src[4]), f"{src[1]} · {src[2].split('.')[-1]}()"
    if kind == "cells":
        nb = executed.get(src[1]) or nbformat.read(NBDIR / f"{src[1]}.ipynb", 4)
        cells = [c for c in nb.cells if c.cell_type == "code"]
        out = []
        for start in src[2]:
            k, c = next((k, c) for k, c in enumerate(cells, 1) if c.source.lstrip().startswith(start))
            out += [f"# ── cell {k}"] + c.source.splitlines() + [""]
        return out[:-1], f"workshop/notebooks/{src[1]}.ipynb"
    if kind == "multi":
        out, names = [], []
        for sub in src[1]:
            lines, name = panel_lines(sub, executed)
            out += [f"# ── {name}"] + lines + [""]
            names.append(name.split(" · ")[0])
        return out[:-1], " + ".join(names)
    raise ValueError(kind)


def build_codemaps(executed):
    maps = {}
    for key, spec in CODEMAPS.items():
        panels = []
        for title, src, marks_ in spec["panels"]:
            lines, where = panel_lines(src, executed)
            mk = {}
            for step, (first, last) in marks_.items():
                a = next((i for i, l in enumerate(lines) if first in l), None)
                if a is None:
                    raise ValueError(f"{key}/{title}: no line with {first!r}")
                b = next(i for i in range(a, len(lines)) if last in lines[i])
                mk[step] = [a, b]
            panels.append({"title": title, "where": where, "code": lines, "marks": mk})
        maps[key] = {"steps": spec["steps"], "panels": panels}
    return maps


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for f in IMG.glob("*.png"):
        f.unlink()
    executed = {}
    players = {}
    for key, (nbname, specs) in PLAYERS.items():
        if nbname not in executed:
            executed[nbname] = execute(nbname)
        cells = [c for c in executed[nbname].cells if c.cell_type == "code"]
        items = []
        for k, (start, title, note) in enumerate(specs):
            cell = next((c for c in cells if c.source.lstrip().startswith(start)), None)
            if cell is None:
                raise ValueError(f"{nbname}: no cell starting with {start!r}")
            items.append({"code": cell.source, "out": outputs(cell, f"{key}{k}"), "title": title, "note": note,
                          "n": cells.index(cell) + 1})
        players[key] = {"notebook": f"workshop/notebooks/{nbname}.ipynb", "cells": items}

    tours = {}
    for key, entries in TOURS.items():
        tours[key] = []
        for file, qual, title, specs in entries:
            lines, first = function_source(ROOT / file, qual)
            tours[key].append({"file": file, "func": qual, "line": first, "title": title,
                               "code": lines, "marks": marks(lines, specs, f"{file}:{qual}")})

    js = ("// Generated by workshop/tools/build_slide_content.py from the executed notebooks and the\n"
          "// repository source. Do not edit by hand: change the script and run it again.\n"
          f"const PLAYERS = {json.dumps(players, ensure_ascii=False, indent=0)};\n"
          f"const TOURS = {json.dumps(tours, ensure_ascii=False, indent=0)};\n"
          f"const PIPELINE = {json.dumps(PIPELINE, ensure_ascii=False, indent=0)};\n"
          f"const CODEMAPS = {json.dumps(build_codemaps(executed), ensure_ascii=False, indent=0)};\n")
    (SLIDES / "content.js").write_text(js, encoding="utf8")
    print("wrote", SLIDES / "content.js", f"({len(js) / 1024:.0f} kB),", len(list(IMG.glob('*.png'))), "figures")


if __name__ == "__main__":
    sys.exit(main())
