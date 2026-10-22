import re, sys, html
src, out = sys.argv[1], sys.argv[2]
text = open(src).read().split("\n", 1)[1].strip()
paras = [" ".join(p.split()) for p in text.split("\n\n")]
def fmt(p):
    p = html.escape(p)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
cues = {
  1: ("Set-up", None),
  2: ("Paper 1", "Claim: −95% oxygen, −74% nitrate vs PID"),
  3: (None, "As printed: = PID in 180/180 runs"),
  4: (None, "Our variant: −81% oxygen (dry), 18/20 runs"),
  5: ("Paper 2", "PID ±1% · DDPG-B 10.6% / 7.1% (paper 7.2 / 5.5) · NH₄ over limit 56% vs 21%"),
  6: ("Contribution", "−29 kWh/d at equal violations (rebuilt plant)"),
  7: ("Close", None),
}
body = []
for i, p in enumerate(paras):
    label, cue = cues.get(i, (None, None))
    if label:
        body.append(f'<div class="label">{label}</div>')
    if cue:
        body.append(f'<div class="cue">{html.escape(cue)}</div>')
    body.append(f"<p>{fmt(p)}</p>")
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Faculty talk</title>
<style>
@page {{ size: A4; margin: 22mm 24mm 20mm 24mm; }}
body {{ font-family: 'DejaVu Serif', Georgia, serif; font-size: 13.5pt; line-height: 1.6; color: #1f1f1e; }}
h1 {{ font-family: 'DejaVu Sans', sans-serif; font-size: 18pt; margin: 0 0 2mm 0; }}
.sub {{ font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; color: #6b6a64; margin-bottom: 8mm; }}
.label {{ font-family: 'DejaVu Sans', sans-serif; font-size: 9pt; letter-spacing: .08em; text-transform: uppercase;
          color: #2a78d6; margin: 6mm 0 1mm 0; font-weight: bold; }}
.cue {{ font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; color: #4a4a45; background: #f1f0ec;
        border-left: 3px solid #2a78d6; padding: 1.5mm 3mm; margin: 0 0 2mm 0; }}
p {{ margin: 0 0 3.5mm 0; }}
strong {{ font-weight: bold; }}
.foot {{ font-family: 'DejaVu Sans', sans-serif; font-size: 8.5pt; color: #6b6a64; margin-top: 8mm;
         border-top: 1px solid #dddcd6; padding-top: 2mm; }}
</style></head><body>
<h1>Two-minute talk for faculty</h1>
<div class="sub">Reproducing two RL papers on the official BSM1 benchmark · about 2 minutes read aloud</div>
{''.join(body)}
<div class="foot">Grey boxes are glance cues, not to be read aloud. Sources: github.com/Amritha902/water_resource —
docs/REPRODUCTION.md (§2.0 paper 1 exact, §3.0 paper 2 official plant).</div>
</body></html>"""
open(out, "w").write(page)
