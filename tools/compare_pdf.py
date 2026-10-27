"""Build the figure-by-figure comparison document (HTML, printed to PDF).

    python tools/compare_pdf.py THEIRS_DIR OUT.html

THEIRS_DIR holds the figures cropped from the papers (p1_fig1.png ...); they
are copyrighted and are not part of this repository, so the PDF is for
private use.  Our figures come from docs/figures/reproduced/, the notes and
tables from docs/comparison_notes.json and results/.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "docs" / "figures" / "reproduced"
FIGS = {
    1: ("Paper 1 — Wang, Li, Ren, Qiao: Multiagent Adaptive Critic Control With Expert Knowledge "
        "for Wastewater Treatment Plants (IEEE T-II, 2026)",
        {1: "Structural diagram of the WWTP based on the activated sludge process.",
         2: "Frame diagram of the MAACC algorithm.",
         3: "Influent flow rates of the WWTP under dry, rainy and stormy weather conditions.",
         4: "Trajectories of S_O,5 and S_NO,2 under the dry weather.",
         5: "Trajectories of S_O,5 and S_NO,2 under the rainy weather.",
         6: "Trajectories of S_O,5 and S_NO,2 under the stormy weather.",
         7: "Trajectories of KLa,5 and Q_a under different weather conditions.",
         8: "Trajectories of S_O,5 and S_NO,2 under the dry weather and the control input noise.",
         9: "Trajectories of KLa,5 and Q_a under dry weather."}),
    2: ("Paper 2 — Du, Chen, Han, Qiao: Dissolved oxygen concentration control in wastewater "
        "treatment process based on reinforcement learning (Sci. China Tech. Sci., 2023)",
        {1: "The overall layout of BSM1.",
         2: "RL-based control framework.",
         3: "Structure of the DDPG-based control strategy.",
         4: "Changes of effluent parameters when the S_O,5 set point value changes.",
         5: "Structure of neural network.",
         6: "The control results of S_O,5 under four dissimilar control strategies in rain weather.",
         7: "Comparisons of several important effluent components under four dissimilar control "
            "strategies in the rain weather.",
         8: "(a) The control results of S_O,5 in the dry weather; (b) in the storm weather."}),
    3: ("Paper 3 — Zhao, Wang, Ren, Qiao: Integrated Online Q-Learning Design for Wastewater "
        "Treatment Processes (IEEE T-II, 2025)",
        {1: "Activated sludge method in the BSM1 platform.",
         2: "Diagram of the IOQL framework.",
         3: "Inflow data in three weather conditions.",
         4: "Trajectories of S_O,5 under IOQL and PID methods and the fixed r_k. (a) Dry. (b) Rain. (c) Rainstorm.",
         5: "Trajectories of S_O,5 under IOQL with different prior policies and the fixed r_k.",
         6: "Trajectories of S_O,5 under IOQL and PID methods and the dynamic r_k.",
         7: "Trajectories of S_O,5 under IOQL with different prior policies and the dynamic r_k."}),
}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.5pt; color: #1f1f1e; line-height: 1.45; }
h1 { font-family: 'DejaVu Sans', sans-serif; font-size: 19pt; margin: 0 0 3mm; }
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 13pt; margin: 0 0 3mm; border-bottom: 1.5px solid #1f1f1e; padding-bottom: 1.5mm; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.5pt; margin: 5mm 0 2mm; }
.pb { page-break-before: always; }
.fig { page-break-inside: avoid; margin-bottom: 7mm; }
.pair { display: flex; gap: 4mm; align-items: flex-start; }
.pair > div { flex: 1; text-align: center; }
.pair img { max-width: 100%; max-height: 105mm; }
.lab { font-family: 'DejaVu Sans', sans-serif; font-size: 8pt; color: #6b6a64; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1mm; }
.note { font-size: 9.5pt; margin-top: 2mm; }
table { border-collapse: collapse; width: 100%; font-family: 'DejaVu Sans', sans-serif; font-size: 8.5pt; margin: 2mm 0 5mm; }
th, td { border-bottom: 1px solid #d8d7d0; padding: 1.2mm 1.6mm; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { color: #4a4a45; }
.small { font-size: 9pt; color: #4a4a45; }
"""


def img64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main(theirs: Path, out: Path) -> None:
    notes = json.loads((ROOT / "docs" / "comparison_notes.json").read_text())
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Figure comparison</title>"
             f"<style>{CSS}</style></head><body>"]
    parts.append(notes["cover"])
    for paper, (title, caps) in FIGS.items():
        parts.append(f"<div class='pb'></div><h2>{html.escape(title)}</h2>")
        parts.append(notes.get(f"p{paper}_intro", ""))
        for n, cap in caps.items():
            t, o = theirs / f"p{paper}_fig{n}.png", OURS / f"p{paper}_fig{n}.png"
            left = f"<img src='{img64(t)}'>" if t.exists() else "<p class='small'>(not available)</p>"
            right = f"<img src='{img64(o)}'>" if o.exists() else "<p class='small'>(not reproduced)</p>"
            parts.append(f"<div class='fig'><h3>Fig. {n}. {html.escape(cap)}</h3><div class='pair'>"
                         f"<div><div class='lab'>Published</div>{left}</div>"
                         f"<div><div class='lab'>Our reimplementation</div>{right}</div></div>"
                         f"<div class='note'>{notes.get(f'p{paper}_fig{n}', '')}</div></div>")
        parts.append(notes.get(f"p{paper}_tables", ""))
    parts.append(notes.get("closing", ""))
    parts.append("</body></html>")
    out.write_text("".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
