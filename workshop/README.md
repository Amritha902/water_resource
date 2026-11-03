# Hands-on workshop: from plant data to a trained model

About 90 minutes, for research scholars. It teaches machine learning and reinforcement learning with this repository's project: learning to control the BSM1 wastewater treatment plant. Short animated slides alternate with notebooks the participants run themselves. Every step points to the file in `src/wwtp/` that does the same thing for real.

```
workshop/
  slides/index.html      animated, interactive slide deck (open in a browser; no internet needed)
  slides/content.js      notebook cells with real outputs and code tours (generated)
  tools/build_slide_content.py  regenerates content.js and slides/img/ from the notebooks and src/
  tools/make_bundle.py   builds dist/water_resource_workshop_bundle_<commit>.zip: everything, offline
  notebooks/
    01_data_collect_clean_prepare.ipynb   collect, clean, prepare (≈ 5 s to run)
    02_choose_and_train_a_model.ipynb     baselines, linear, MLP, causal CNN, overfitting, save/reload (≈ 30 s)
    03_reinforcement_learning_in_brief.ipynb  environment, PID, imitation, Q-learning, reward design (≈ 30 s)
  slides/stills/         a picture of every slide (JPEG, 1920 × 1080), for other presentation tools
  SPEECHIFY_OUTLINE.md   slide-by-slide text and narration, paste-ready for an AI presentation builder
  examples/              lesson1 … lesson6 scripts: each step of the lesson in one terminal command
  requirements.txt
  data/                  created by notebook 1 (downloads, prepared.npz, saved model); not in git
```

## Published pages

- Teaching guide, the whole lesson slide by slide with every script explained: https://claude.ai/artifact/HqpLrBanoEpinRiffb7Wma (source `workshop/guide/TEACHING_GUIDE.src.md`, built by `python workshop/tools/build_guide.py` into `workshop/TEACHING_GUIDE.md` and `workshop/teaching_guide.html`)

- Slide deck: https://claude.ai/artifact/66EGWBVkQQRAKsHadr5ob5
- Part 1, collect, clean and prepare: https://claude.ai/artifact/MhofB2ez4udSC5zVvviBkq
- Part 2, choose a model and train it: https://claude.ai/artifact/XJEAVKk3wnmgh8RBvUHtwm
- Part 3, reinforcement learning in brief: https://claude.ai/artifact/TsXUrKphd3CRGddACxi2m5

The notebook pages are made by `python workshop/tools/notebooks_to_html.py --links workshop/notebooks/html/links.json` (they show a real run; "Explanations only" hides the code). Each notebook has a "New to notebooks?" box, a glossary and an "In plain words" box for every step.

## Setup (do this the day before)

Python 3.10 or newer. From the repository root:

```bash
pip install -r workshop/requirements.txt
```

```bash
jupyter lab workshop/notebooks
```

The workshop needs neither `bsm2-python` nor a GPU. Notebook 1 downloads the three official influent files (≈ 150 kB each) from a pinned commit of `davidfernand/BSM1_in_Python` into `workshop/data/`. If the room has no internet, run notebook 1 once beforehand and hand out the `workshop/data/` folder.

**Slides:** open `workshop/slides/index.html` in Chrome, Edge or Firefox.

| Key | Action |
|---|---|
| → or space | next point, then next slide |
| ← | back |
| R | replay the slide's animation |
| F | full screen |
| `#12` at the end of the address | jump to slide 12 |

The buttons and sliders on the slides are live: press them during the talk.

## Timetable: one lesson in seven steps

The deck is one continuous lesson (62 slides). Every step has the same rhythm: a divider with the goal → the idea (animated) → **where is the code?** (the same steps, colour by colour, in the hands-on script, the notebook and the project) → the code running (notebook player or code tour) → **check yourself** (three questions, click to reveal) → hands-on.

| Time | Slides | Step | Terminal | Notebook |
|---|---|---|---|---|
| 0:00 | 1–6 | Welcome, the plan, glossary of every word, the problem, the plant and its two loops, the project's pipeline | | |
| 0:04 | 7–10 | **1 · What is a model?** Line fit, four training steps everywhere, quiz | `lesson1_knobs.py` | |
| 0:12 | 11–20 | **2 · Collect and clean** Data, loading, cleaning, time grid, code map, our loader, notebook running, quiz, hands-on | `lesson2_load_and_clean.py` | NB1 A, B |
| 0:32 | 21–28 | **3 · Prepare** Scaling, windows, split by time, code map, notebook running, quiz, hands-on | `lesson3_prepare.py` | NB1 C |
| 0:44 | 29–37 | **4 · Choose the model** Shapes decide, layer by layer, parameters, CNN, GRU, code map, notebook running, quiz | `lesson4_models.py` | NB2 1–6 |
| 0:54 | 38–46 | **5 · Train and evaluate** Loss, gradient descent, training loop, code map, notebook running, baselines, quiz, hands-on | `lesson5_train_and_evaluate.py` | NB2 5–9 |
| 1:14 | 47–55 | **6 · Reinforcement learning** RL loop, actor-critic, code map, our RL code, notebook running, SAC-IL, quiz, hands-on | `lesson6_rl.py` | NB3 |
| 1:27 | 56–62 | **7 · Our project's code** Papers as printed, our controller, held-out test 2 (the latest results), quiz, questions learners ask, take-home lessons | | |

Keys: → advances; on interactive slides it first steps inside the slide (next cell, next code block, next step, next answer). ← goes back. R replays an animation; F is full screen.

Every terminal script is in `workshop/examples/` and runs in seconds (lesson 5: about 20 s). Lessons 4 and 5 read `workshop/data/prepared.npz`, written by lesson 3 or notebook 1.

If time runs short: run the hands-on on the projector with the notebook-player slides instead of having everyone run them.

**After changing a notebook or the project code**, rebuild the players, tours and code maps (runs the notebooks, about 1 minute):

```bash
python workshop/tools/build_slide_content.py
```

## What each notebook shows (results from our run; small differences between machines are normal)

**Notebook 1.**
- The raw file has a header row, which turns every column into text: look before you parse.
- Week 1 is identical in the dry, rain and storm files, so splitting must be by time.
- On a deliberately damaged copy the cleaning finds missing rows, duplicates, a negative flow, three spikes and a stuck sensor. It also shows a pandas pitfall: `interpolate(limit=4)` fills the start of long gaps too.
- The S_I dilution fix (`plant.py:89`).
- Split: train = dry + rain week 2; validation = storm 1; test = storm 2. Scaling uses training statistics only. The windows give X `(N, 96, 3)` → y `(N, 8)`, with N = 1906 / 305 / 353.

**Notebook 2.** Validation error, 2 h ahead, in m³/d:

| Model | Parameters | Validation error, 2 h ahead [m³/d] |
|---|---|---|
| persistence | none | 5 749 |
| same time yesterday | none | 4 128 |
| ridge | 2 312 | 2 929 |
| MLP | 23 176 | 3 120 |
| causal CNN | 10 808 | 2 576 |

- On the test storm the **linear model is best at 2 h**. This makes the lesson concrete: with ≈ 1 900 examples a simple model wins, just as a tuned PID matched the project's learned controller.
- Overfitting demo: 415 k parameters on 200 examples. The gap between validation and training loss is 51× without regularisation and 7× with it. No model gets the validation loss much below ≈ 0.25, because the training data contains no storm: the limit is the data.

**Notebook 3.**
- The PID error is 0.30 mg/L. Imitating it with a small network gives 0.30: a copy cannot beat its teacher.
- Q-learning gives 0.24–0.25 on three seeds and on an unseen rain day.
- The **PID with a one-line physics feedforward gives 0.23**, with no training. Always ask: better than what?
- With air made 50× more expensive (`energy_weight = 5`) the agent switches the air almost off: oxygen falls to 0.5 mg/L, below 1.5 mg/L 77 % of the time. The reward is the specification (paper 2's DDPG did the same with ammonium).

## Mapping to the repository

| Workshop step | Repository file |
|---|---|
| pinned download, keep numeric rows, cache | `src/wwtp/official/plant.py:30` `_official_table` |
| S_I dilution fix, TSS column | `src/wwtp/official/plant.py:89–90`, `docs/DATASET.md` §3.3 |
| 15 min → 45 s hold | `src/wwtp/official/plant.py:144` `official_series` |
| log + standardise on train only | `src/wwtp/warning/dataset.py:224` `Normaliser` |
| causal windows | `src/wwtp/forecast/dataset.py:67` `_windows` |
| split by seeds / weeks / pre-registered held-out test | `experiments/04_collect_warning_data.py`, `src/wwtp/rl/train.py`, `docs/PAPER1_HELDOUT_PROTOCOL.md` |
| causal TCN, quantile (pinball) loss | `src/wwtp/forecast/models.py` |
| training loop, keep the best epoch | `src/wwtp/forecast/train.py:51`, `src/wwtp/warning/train.py:80` |
| baselines (persistence, trend, gradient boosting) | `src/wwtp/warning/baselines.py` |
| environment, reward | `src/wwtp/rl/env.py`, `src/wwtp/rl/rewards.py` |
| DDPG update | `src/wwtp/rl/ddpg.py:214` |
| imitation then SAC, safety gate | `src/wwtp/sacil/sacil.py:279`, `experiments/28_sac_il_paper1.py`, `experiments/33_paper1_study.py` |

## Notes

- The slide animations of the influent use the repository's own generator (`src/wwtp/bsm1/influent.py`), not the official files, which are not redistributed. The notebooks use the official files.
- The numbers on slide 22 (oxygen error against paper 1's PID) are copied from the repository README.
