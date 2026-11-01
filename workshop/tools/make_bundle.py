"""Build the complete, offline workshop bundle as one zip.

Contents:
  * the whole repository at the current commit (git archive): library, experiments,
    tests, results, trained models, docs, demo, workshop;
  * the three notebooks executed, with their outputs (workshop/notebooks/executed/);
  * the data, so nothing needs the internet: the official IWA influent files with their
    GPL-3 licence (data/official/ and workshop/data/), the prepared dataset and the
    trained forecaster of notebook 2;
  * START_HERE.md.

    python workshop/tools/make_bundle.py            # from the repository root; writes dist/*.zip

The official influent files are GPL-3 (davidfernand/BSM1_in_Python); they are not
committed to git, only put into the bundle, with the licence next to them.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import nbformat
import numpy as np
import openpyxl
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
REV = "5dadb209a5a34ecb0e2f83a2e79c060bf3470582"
BASE = f"https://raw.githubusercontent.com/davidfernand/BSM1_in_Python/{REV}"
FILES = {"dry": "Inf_dry_2006.xlsx", "rain": "Inf_rain_2006.xlsx", "storm": "Inf_strm_2006.xlsx"}
NOTEBOOKS = ("01_data_collect_clean_prepare", "02_choose_and_train_a_model", "03_reinforcement_learning_in_brief")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def fetch(url: str, path: Path) -> None:
    if not path.exists():
        print("downloading", url.rsplit("/", 1)[-1], flush=True)
        urllib.request.urlretrieve(url, path)


def official_npy(xlsx: Path) -> np.ndarray:
    """The same parsing as src/wwtp/official/plant.py:_official_table."""
    ws = openpyxl.load_workbook(xlsx, read_only=True, data_only=True).worksheets[0]
    rows = [r[:15] for r in ws.iter_rows(values_only=True) if r and isinstance(r[0], (int, float))]
    return np.array(rows, dtype=float)


START_HERE = """# Start here

Bundle of the repository `amritha902/water_resource` at commit `{sha}` ({day}), with the
hands-on ML/RL workshop, its data and the notebooks already run. Nothing in it needs the internet.

## 1. The workshop (90 minutes, any laptop, Python 3.10+)

```bash
pip install -r workshop/requirements.txt
```

1. Slides: open `workshop/slides/index.html` in Chrome, Edge or Firefox. Use → and ←; on the
   "In code" slides → runs the next notebook cell; on the "Our code" slides → steps through the code.
2. Notebooks: `jupyter lab workshop/notebooks`. They run as they are (the data is already in
   `workshop/data/`). To see every output without running anything, open
   `workshop/notebooks/executed/`.
3. Presenter guide, timetable and expected results: `workshop/README.md`.
4. Pictures of every slide: `workshop/slides/stills/`; narration per slide: `workshop/SPEECHIFY_OUTLINE.md`.

## 2. The full project (Python 3.11)

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

```bash
python demo/run_demo.py
```

* `README.md`: results and the command behind every result file.
* `docs/REPORT.md`: the full report; `docs/PAPER1_STUDY.md` and `docs/PAPER1_BASELINES_HELDOUT.md`: our controller.
* `docs/DATASET.md`: plant, data, the correction made to it, and every setting.
* Code: `src/wwtp/` (library), `experiments/` (numbered scripts), `results/` (every number in the report),
  `artifacts/` (trained models).

The official influent is already in `data/official/` (the project would otherwise download it on first use).

## What is where

| Folder | Contents |
|---|---|
| `workshop/` | slides, notebooks (clean and executed), data, presenter guide |
| `src/wwtp/` | official plant wrapper, the three papers as printed, our controller (ILC, SAC-IL, repaired MAACC), forecaster, twin, warning model, RL agents |
| `experiments/` | one script per step of the project (map: slide 4 of the deck) |
| `tests/` | {n_tests} tests |
| `results/`, `artifacts/` | result files and trained models |
| `docs/` | report, study, held-out protocol and results, dataset description, figures |
| `demo/` | live demo with cached start states |
| `data/official/` | the IWA BSM1 influent files (GPL-3, see `data/official/NOTICE.md`) |

## Licences

* BSM1 plant: bsm2-python 0.0.16 (BSD-3), installed by pip.
* Influent files `Inf_dry/rain/strm_2006.xlsx`: IWA BSM1 files from davidfernand/BSM1_in_Python,
  commit `{rev}`, GPL-3. Licence text: `data/official/LICENSE-GPL-3.txt`.
"""

NOTICE = """# Official influent files

`Inf_dry_2006.xlsx`, `Inf_rain_2006.xlsx`, `Inf_strm_2006.xlsx` are the IWA BSM1 dry, rain and storm
influent files as distributed in https://github.com/davidfernand/BSM1_in_Python (commit `{rev}`,
folder `code/Influent files/`), under the GNU General Public License v3 (`LICENSE-GPL-3.txt`, copied from
the same commit). They are unmodified. The `.npy` files hold the same numbers (the numeric rows, first 15
columns), as `src/wwtp/official/plant.py` caches them. The correction the project applies to S_I is made in
code at load time (`load_influent`), not in these files.
"""


def main() -> None:
    sha = git("rev-parse", "--short", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("note: uncommitted changes are NOT in the bundle (it is built from the commit)", flush=True)
    work = DIST / "build"
    if work.exists():
        shutil.rmtree(work)
    top = work / f"water_resource_{sha}"
    top.mkdir(parents=True)

    # 1. the repository at this commit
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(top, filter="data")

    # 2. data: official files + licence, for the project and for the workshop
    cache = DIST / "cache"; cache.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        fetch(f"{BASE}/code/Influent%20files/{name}", cache / name)
    fetch(f"{BASE}/LICENSE", cache / "LICENSE-GPL-3.txt")
    off, wdata = top / "data" / "official", top / "workshop" / "data"
    off.mkdir(parents=True); wdata.mkdir(parents=True)
    for weather, name in FILES.items():
        shutil.copy(cache / name, off / name)
        shutil.copy(cache / name, wdata / name)
        np.save(off / f"{weather}.npy", official_npy(cache / name))
    for d in (off, wdata):
        shutil.copy(cache / "LICENSE-GPL-3.txt", d / "LICENSE-GPL-3.txt")
        (d / "NOTICE.md").write_text(NOTICE.format(rev=REV), encoding="utf8")

    # 3. run the notebooks inside the bundle: fills workshop/data/ and keeps executed copies
    nbdir = top / "workshop" / "notebooks"
    (nbdir / "executed").mkdir()
    for name in NOTEBOOKS:
        print("running", name, flush=True)
        nb = nbformat.read(nbdir / f"{name}.ipynb", 4)
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(nbdir)}}).execute()
        nbformat.write(nb, nbdir / "executed" / f"{name}.ipynb")

    # 4. guide
    n_tests = sum(1 for f in (top / "tests").glob("test_*.py") for line in f.read_text().splitlines() if line.startswith("def test_"))
    (top / "START_HERE.md").write_text(START_HERE.format(sha=sha, day=date.today().isoformat(), rev=REV, n_tests=n_tests),
                                       encoding="utf8")

    # 5. zip
    out = DIST / f"water_resource_workshop_bundle_{sha}.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(top.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and ".ipynb_checkpoints" not in f.parts:
                z.write(f, f.relative_to(work))
    shutil.rmtree(work)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
