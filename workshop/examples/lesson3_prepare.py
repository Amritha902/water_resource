"""Lesson 3, hands-on: prepare the data so a model can learn from it.

Run it:   python workshop/examples/lesson3_prepare.py

Four steps, as in src/wwtp/forecast/dataset.py and src/wwtp/warning/dataset.py:
  1 CHOOSE   inputs (what the plant can measure) and target (what we want to know)
  2 SPLIT    by time: learn on the past, test on the future
  3 SCALE    log, then standardise with numbers from the TRAINING data only
  4 WINDOWS  cut the series into (24 h in → 2 h out) examples; save them
Writes workshop/data/prepared.npz, the same file notebook 1 writes (notebook 2 and lesson 4 read it).
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
REV = "5dadb209a5a34ecb0e2f83a2e79c060bf3470582"
URL = f"https://raw.githubusercontent.com/davidfernand/BSM1_in_Python/{REV}/code/Influent%20files/"
FILES = {"dry": "Inf_dry_2006.xlsx", "rain": "Inf_rain_2006.xlsx", "storm": "Inf_strm_2006.xlsx"}
COLS = ["t", "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P", "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK", "Q"]
df = {}
for w, name in FILES.items():
    if not (DATA / name).exists():
        urllib.request.urlretrieve(URL + name, DATA / name)
    d = pd.read_excel(DATA / name, header=0)
    d.columns = COLS
    df[w] = d.astype(float).iloc[:-1]                       # the last row repeats the first

# ── 1 CHOOSE: inputs a real plant measures at the inlet; target = the flow in the next 2 hours
CHANNELS = ["Q", "S_NH", "S_S"]                             # as forecast/dataset.py: CHANNELS
LOOKBACK, HORIZON = 96, 8                                   # 24 h in, 2 h out (15-min samples)
print("1 CHOOSE  inputs:", CHANNELS, f"  one example: ({LOOKBACK}, {len(CHANNELS)}) → ({HORIZON},)")

# ── 2 SPLIT by the time of the TARGET: train = past and calm, val = storm 1, test = storm 2
SPLITS = {"train": [("dry", 1.0, 14.0), ("rain", 7.0, 14.0)],
          "val": [("storm", 7.0, 10.25)],
          "test": [("storm", 10.25, 14.0)]}
print("2 SPLIT   ", {k: v for k, v in SPLITS.items()})

# ── 3 SCALE: log, then (x − mean) / std, with mean and std from the training rows only
train_rows = np.concatenate([df[w][(df[w].t >= lo - 1) & (df[w].t < hi)][CHANNELS].values for w, lo, hi in SPLITS["train"]])
MEAN, STD = np.log(train_rows).mean(0), np.log(train_rows).std(0)
leaky = np.log(np.concatenate([df[w][CHANNELS].values for w in FILES])).mean(0)
print("3 SCALE   mean of log, training rows only:", MEAN.round(3), "  with the test leaking in:", leaky.round(3))


def scale(x):
    return (np.log(x) - MEAN) / STD


# ── 4 WINDOWS: past 96 steps → next 8 flows; the input always ends before the target starts
def windows(weather, t_lo, t_hi):
    z, t = scale(df[weather][CHANNELS].values).astype(np.float32), df[weather].t.values
    X, Y, T = [], [], []
    for s in range(len(z) - LOOKBACK - HORIZON + 1):
        first = s + LOOKBACK                                # first target row
        if t[first] < t_lo - 1e-9 or t[first + HORIZON - 1] >= t_hi - 1e-9:
            continue
        X.append(z[s:first]); Y.append(z[first:first + HORIZON, 0]); T.append(t[first])
    return np.array(X), np.array(Y), np.array(T)


data = {}
for split, parts in SPLITS.items():
    xs, ys, ts = zip(*[windows(*p) for p in parts])
    data[split] = tuple(np.concatenate(a) for a in (xs, ys, ts))
    print(f"4 WINDOWS {split:5s} X {data[split][0].shape}  y {data[split][1].shape}")

# why not split at random? neighbouring windows share almost all their samples
rng = np.random.default_rng(0)
X = data["train"][0]
is_test = rng.random(len(X)) < 0.2
near_copies = np.mean([(is_test[i] and (not is_test[i - 1] or not is_test[i + 1])) for i in range(1, len(X) - 1)]) / is_test.mean()
print(f"          a random split would put {near_copies:.0%} of test windows right next to a training window "
      f"(sharing {LOOKBACK - 1} of {LOOKBACK} samples)")

np.savez_compressed(DATA / "prepared.npz", **{f"X_{s}": data[s][0] for s in data}, **{f"y_{s}": data[s][1] for s in data},
                    **{f"t_{s}": data[s][2] for s in data}, mean=MEAN, std=STD, channels=np.array(CHANNELS),
                    lookback=LOOKBACK, horizon=HORIZON, source_rev=REV)
print("          saved", DATA / "prepared.npz", "(windows + the scaler, together)")

# Try this: change LOOKBACK to 24 (6 hours). How many windows now? Does it matter for a daily cycle?
