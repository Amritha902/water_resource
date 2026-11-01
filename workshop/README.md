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
  requirements.txt
  data/                  created by notebook 1 (downloads, prepared.npz, saved model); not in git
```

## Published pages

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

## Timetable

The deck has three kinds of interactive slides besides the animations:

- **★ Notebook players:** real notebook cells with ▶ Run. Each run shows the output of a real run of the notebook, and a panel explains the idea. Press → to run the next cell; ← un-runs it.
- **◆ Code tours:** the project's own functions, cut from the source files. Each block is highlighted with its explanation; press → for the next block, or click a block or a tab.
- **Where is the code?** (slides 7 and 14): the same steps, the same colour, in the hands-on script (`workshop/examples/`), the notebook and the project code.
- **The pipeline map** (slide 5): every experiment script of the project in the order it was built. Click a box, or press →.

| Time | Slides | What happens | Notebook |
|---|---|---|---|
| 0:00 | 1–5 | The plant (move the air slider on slide 1), **glossary of every word in plain language** (slide 3), the four ML tasks, **pipeline map** of our code | |
| 0:06 | 6–7 | **What is a model**: press ▶ train and let the line fit itself; then **where is the code?** (slide 7): the same four training steps, colour by colour, in the script, the notebook and the project | |
| 0:09 | 8–14 | Collect, cleaning, time grid; **where is the code?** (slide 14): download, read, check, fix, in all three places; ◆ how our code loads and corrects the data; ★ collect and clean, cell by cell | |
| 0:17 | 15 | **Hands-on** | NB1 parts A, B (10 min) |
| 0:27 | 16–19 | Scaling, windows, split by time; ★ prepare, cell by cell | |
| 0:31 | 20 | **Hands-on** | NB1 part C (8 min) |
| 0:39 | 21–25 | Inputs and outputs decide the model; **inside a network, layer by layer** (press "next step" by hand); parameters; CNN; GRU; ★ shapes, baselines and models in code | |
| 0:48 | 26–30 | Loss, gradient descent, the training loop; ★ training, overfitting and the test in code | |
| 0:55 | 31–32 | Baselines; **hands-on** | NB2 (15 min) |
| 1:10 | 33–36 | RL loop, actor-critic parts, ◆ reward, DDPG and SAC-IL updates in our code, ★ RL cell by cell | |
| 1:16 | 37–39 | SAC-IL pipeline; ◆ papers 1 and 3 as printed; ◆ our controller (ILC, control step, safety gate, training script) | |
| 1:20 | 40 | **Hands-on** | NB3 (7 min) |
| 1:27 | 41 | Lessons, questions | |

If time runs short: the ★ players can replace the matching hands-on time (run them on the projector instead), and the ◆ tours can be shortened to their first tab.

**After changing a notebook or the project code**, rebuild the players and tours (runs the notebooks, about 1 minute):

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
