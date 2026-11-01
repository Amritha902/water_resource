"""Lesson 2, hands-on: collect and clean the plant data.

Run it:   python workshop/examples/lesson2_load_and_clean.py

Four steps, the same ones the project uses in src/wwtp/official/plant.py:
  1 DOWNLOAD   from a pinned source, once
  2 READ       look at the raw file before trusting it
  3 CHECK      time step, missing values, duplicates, impossible values
  4 FIX        repair what is wrong, and the physics fix the project makes
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
REV = "5dadb209a5a34ecb0e2f83a2e79c060bf3470582"
URL = f"https://raw.githubusercontent.com/davidfernand/BSM1_in_Python/{REV}/code/Influent%20files/"
COLS = ["t", "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P", "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK", "Q"]

# ── 1 DOWNLOAD: pinned to one commit, so it is the same file for everyone, forever
path = DATA / "Inf_rain_2006.xlsx"
if not path.exists():
    urllib.request.urlretrieve(URL + path.name, path)
dry_path = DATA / "Inf_dry_2006.xlsx"
if not dry_path.exists():
    urllib.request.urlretrieve(URL + dry_path.name, dry_path)

# ── 2 READ: first, carelessly
raw = pd.read_excel(path, header=None)
print("2 READ   raw shape", raw.shape, "  first row:", list(raw.iloc[0, :4]), "...")
print("         column types:", {str(k): v for k, v in raw.dtypes.value_counts().items()}, " ← all text, because row 0 is names")
#    then properly: row 0 is the header; give the columns our names; make them numbers
d = pd.read_excel(path, header=0)
d.columns = COLS
d = d.astype(float)
print("         after reading properly:", d.shape, " types:", {str(k): v for k, v in d.dtypes.value_counts().items()})

# ── 3 CHECK: the standard questions, every dataset, every time
step_min = np.unique(np.round(np.diff(d.t) * 1440, 3))
print("3 CHECK  time step [min]:", step_min,
      "  missing values:", int(d.isna().sum().sum()),
      "  duplicate times:", int(d.t.duplicated().sum()),
      "  negative values:", int((d.drop(columns="t") < 0).sum().sum()))
print("         S_I values in the whole file:", d.S_I.unique(), " ← constant, even when rain doubles the flow?")

# ── 4 FIX (a) a broken sensor: damage a copy, then repair it
x = d.copy()
x.loc[100:102, "S_NH"] = np.nan                          # a 45-min gap
x.loc[300, "S_NH"] *= 10                                 # a spike
med = x.S_NH.rolling(9, center=True, min_periods=3).median()
spike = (x.S_NH - med).abs() > 0.5 * med                 # far from its neighbours → wrong
x.loc[spike, "S_NH"] = np.nan
x["S_NH"] = x.S_NH.interpolate(limit_area="inside")      # fill short gaps with a straight line
print("4 FIX    spikes found:", int(spike.sum()), "  largest error after repair:",
      round(float((x.S_NH - d.S_NH).abs().max()), 3), "g/m3")

# ── 4 FIX (b) the physics fix of the project: rain dilutes the inert COD   (plant.py, line 89)
dry = pd.read_excel(dry_path, header=0)
dry.columns = COLS
d["S_I"] = 30.0 * dry.Q.astype(float).values / d.Q.values
print("         S_I after the dilution fix: min", round(d.S_I.min(), 1), " max", round(d.S_I.max(), 1), "g/m3")

# ✏️ Try: change 0.5 * med to 0.05 * med. How many "spikes" now? Are they real?
