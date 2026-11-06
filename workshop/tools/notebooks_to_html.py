"""Turn the workshop notebooks into self-contained web pages, with the outputs of a real run.

Each page: a contents bar, the explanations, the code (with Copy buttons), every output
(text, tables, figures inlined), and a switch to hide the code for readers who only want
the ideas. No library and no network needed to view it.

    python workshop/tools/notebooks_to_html.py                    # runs the notebooks (about 1 min)
    python workshop/tools/notebooks_to_html.py --links links.json # add links to the deck and the other parts

links.json: {"deck": url, "01_...": url, "02_...": url, "03_...": url}
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import markdown
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NBDIR = ROOT / "workshop" / "notebooks"
OUT = NBDIR / "html"
NAMES = {"00_ml_skeleton": "Notebook 0 · The skeleton of every ML notebook",
         "00b_three_ways_to_load_data": "Notebook 0b · Three ways to bring data in, and feed it to a model",
         "01_data_collect_clean_prepare": "Notebook 1 · Collect, clean and prepare the plant data",
         "02_choose_and_train_a_model": "Notebook 2 · Choose a model and train it",
         "03_reinforcement_learning_in_brief": "Notebook 3 · Reinforcement learning in brief"}

KW = set("def class return for in if elif else while with as import from not and or lambda None True False try "
         "except finally yield is break continue pass assert raise global nonlocal del".split())
BUILTIN = set("print len range zip enumerate float int str list dict tuple set min max sum abs isinstance super "
              "np torch nn F pd plt self".split())
TOKEN = re.compile(r'(#[^\n]*)|("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|[rbuf]?"(?:\\.|[^"\\\n])*"|[rbuf]?\'(?:\\.|[^\'\\\n])*\')'
                   r'|(\b\d+(?:\.\d*)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)|(@\w+)')


def highlight(code: str) -> str:
    out, last = [], 0
    for m in TOKEN.finditer(code):
        out.append(html.escape(code[last:m.start()]))
        t = m.group(0)
        if m.group(1): cls = "c"
        elif m.group(2): cls = "s"
        elif m.group(3): cls = "n"
        elif m.group(4): cls = "k" if t in KW else "b" if t in BUILTIN else ("f" if code[m.end():m.end() + 1] == "(" else "")
        else: cls = "k"
        out.append(f'<span class="{cls}">{html.escape(t)}</span>' if cls else html.escape(t))
        last = m.end()
    out.append(html.escape(code[last:]))
    return "".join(out)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "section"


CSS = """
:root{--bg:#fbfaf7;--ink:#1d2433;--ink2:#4b5563;--muted:#8a919c;--line:#e3e1db;--blue:#2563eb;--orange:#ea580c;--soft:#f1efe9}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 "Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(251,250,247,.96);backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
.bar{max-width:1240px;margin:0 auto;padding:10px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.bar .t{font-weight:700}.bar .k{font-size:13px;color:var(--blue);font-weight:600;letter-spacing:1px;text-transform:uppercase}
.bar .sp{flex:1}
button,.navlink{font:inherit;font-size:14px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 11px;cursor:pointer;color:var(--ink);text-decoration:none}
button:hover,.navlink:hover{border-color:var(--blue);color:var(--blue)}button.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.wrap{max-width:1240px;margin:0 auto;display:flex;gap:28px;padding:0 20px}
nav.toc{position:sticky;top:64px;align-self:flex-start;width:250px;flex:none;max-height:calc(100vh - 80px);overflow:auto;padding:18px 0;font-size:14px}
nav.toc a{display:block;color:var(--ink2);text-decoration:none;padding:3px 8px;border-left:2px solid transparent}
nav.toc a.h3{padding-left:20px;font-size:13px;color:var(--muted)}
nav.toc a.cur{border-color:var(--blue);color:var(--blue);font-weight:600}
main{flex:1;min-width:0;max-width:900px;padding:10px 0 80px}
h1{font-size:36px;line-height:1.2;margin:24px 0 10px}h2{font-size:27px;margin:44px 0 8px;padding-top:6px;border-top:1px solid var(--line)}h3{font-size:20px;margin:28px 0 6px}
h1,h2,h3{scroll-margin-top:70px}
p,li{max-width:76ch}
code{font-family:Consolas,Menlo,monospace;background:var(--soft);padding:1px 5px;border-radius:4px;font-size:.88em}
table{border-collapse:collapse;margin:10px 0;font-size:15px}td,th{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left;vertical-align:top}th{color:var(--ink2)}
blockquote{margin:14px 0;padding:12px 16px;background:#fff8e6;border:1px solid #fde68a;border-left:5px solid #f59e0b;border-radius:0 10px 10px 0}
blockquote p{margin:0}
hr{border:none;border-top:1px dashed var(--line);margin:18px 0}
.md ul,.md ol{padding-left:24px}
.cell{margin:14px 0}
.in{display:flex;align-items:center;gap:10px;font:13px Consolas,Menlo,monospace;color:var(--blue);margin-bottom:3px}
.in button{font-size:12px;padding:2px 8px}
pre{margin:0;overflow:auto;font:13.5px/1.5 Consolas,Menlo,monospace}
pre.code{background:#0f172a;color:#e2e8f0;padding:12px 14px;border-radius:10px}
pre.code .k{color:#c4b5fd}pre.code .s{color:#fcd34d}pre.code .c{color:#7d8ba1;font-style:italic}pre.code .n{color:#5eead4}pre.code .b{color:#93c5fd}pre.code .f{color:#bfdbfe}
.out{border-left:3px solid #fde68a;padding:4px 0 4px 12px;margin-top:6px}
.out pre{background:#fff;color:#111827;font-size:13px;max-height:420px}
.out img{max-width:100%;display:block;margin:6px 0;background:#fff}
.out .html{overflow:auto;max-height:440px;font-size:13px}.out .html table{font-size:13px}
.nocode .cell .in,.nocode .cell pre.code{display:none}.nocode .cell .out{border-left-color:transparent;padding-left:0}
.nocode .cell.no-output{display:none}
.endnav{display:flex;gap:10px;flex-wrap:wrap;margin-top:40px;padding-top:18px;border-top:1px solid var(--line)}
.note{font-size:13px;color:var(--muted)}
@media (max-width:900px){nav.toc{display:none}h1{font-size:28px}}
"""

JS = """
const root=document.documentElement;
function setMode(code){document.body.classList.toggle('nocode',!code);
  document.getElementById('m-code').classList.toggle('on',code);document.getElementById('m-ideas').classList.toggle('on',!code);
  try{localStorage.setItem('ws-nocode',code?'0':'1')}catch(e){}}
document.getElementById('m-code').onclick=()=>setMode(true);
document.getElementById('m-ideas').onclick=()=>setMode(false);
try{if(localStorage.getItem('ws-nocode')==='1')setMode(false)}catch(e){}
document.querySelectorAll('button.copy').forEach(b=>b.onclick=()=>{
  const code=b.closest('.cell').querySelector('pre.code').innerText;
  const done=t=>{b.textContent=t;setTimeout(()=>b.textContent='Copy',1400)};
  try{navigator.clipboard.writeText(code).then(()=>done('Copied ✓'),()=>done('Select and copy by hand'))}catch(e){done('Select and copy by hand')}});
const heads=[...document.querySelectorAll('main h2, main h3')],links=[...document.querySelectorAll('nav.toc a')];
const spy=()=>{let cur=heads[0];for(const h of heads){if(h.getBoundingClientRect().top<120)cur=h}
  links.forEach(a=>a.classList.toggle('cur',cur&&a.getAttribute('href')==='#'+cur.id))};
document.addEventListener('scroll',spy,{passive:true});spy();
"""


def render_md(src: str, toc: list) -> str:
    h = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])

    def head(m):
        level, inner = m.group(1), m.group(2)
        text = re.sub("<[^>]+>", "", inner)
        anchor = slug(text)
        base, k = anchor, 2
        while any(a == anchor for _, _, a in toc):
            anchor, k = f"{base}-{k}", k + 1
        if level in ("2", "3"):
            toc.append((level, html.unescape(text), anchor))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'
    return re.sub(r"<h([1-3])>(.*?)</h\1>", head, h)


def render_outputs(cell) -> str:
    parts = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            parts.append(f"<pre>{html.escape(o.text)}</pre>")
        elif o.output_type in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                parts.append(f'<img alt="figure" src="data:image/png;base64,{d["image/png"].strip()}">')
            elif "text/html" in d:
                parts.append(f'<div class="html">{d["text/html"]}</div>')
            elif "text/plain" in d and not d["text/plain"].startswith(("<matplotlib", "Text(", "<Axes", "[<matplotlib")):
                parts.append(f"<pre>{html.escape(d['text/plain'])}</pre>")
        elif o.output_type == "error":
            raise RuntimeError(f"{o.ename}: {o.evalue}")
    # merge consecutive stream outputs (print in a loop)
    joined = "".join(parts).replace("</pre><pre>", "")
    return f'<div class="out">{joined}</div>' if joined else ""


def page(name: str, nb, links: dict) -> str:
    toc, body = [], []
    n = 0
    for c in nb.cells:
        if c.cell_type == "markdown":
            body.append(f'<div class="md">{render_md(c.source, toc)}</div>')
        elif c.cell_type == "code" and c.source.strip():
            n += 1
            outs = render_outputs(c)
            body.append(f'<div class="cell{"" if outs else " no-output"}"><div class="in">In [{n}]:<button class="copy">Copy</button></div>'
                        f'<pre class="code">{highlight(c.source)}</pre>{outs}</div>')
    title = NAMES[name]
    nav = "".join(f'<a class="h{lv}" href="#{a}">{html.escape(t)}</a>' for lv, t, a in toc)
    keys = list(NAMES)
    i = keys.index(name)
    endnav = []
    if links.get("deck"):
        endnav.append(f'<a class="navlink" href="{links["deck"]}">The slide deck</a>')
    for j, k in enumerate(keys):
        if j != i and links.get(k):
            endnav.append(f'<a class="navlink" href="{links[k]}">{"← " if j < i else ""}{html.escape(NAMES[k])}{" →" if j > i else ""}</a>')
    top_links = "".join(endnav[:1])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body>
<header><div class="bar"><span class="k">Workshop notebook</span><span class="t">{html.escape(title)}</span><span class="sp"></span>
{top_links}<button id="m-code" class="on">Show code</button><button id="m-ideas">Explanations only</button></div></header>
<div class="wrap"><nav class="toc">{nav}</nav>
<main>
<p class="note">This page shows the notebook <code>workshop/notebooks/{name}.ipynb</code> with the results of a real run.
To run it yourself: <code>pip install -r workshop/requirements.txt</code>, then <code>jupyter lab workshop/notebooks</code>.
"Explanations only" hides the code and keeps the text and the results.</p>
{''.join(body)}
<div class="endnav">{''.join(endnav)}</div>
</main></div>
<script>{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", help="JSON file with the URLs of the deck and the notebooks")
    a = ap.parse_args()
    links = json.loads(Path(a.links).read_text()) if a.links else {}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        nb = nbformat.read(NBDIR / f"{name}.ipynb", 4)
        print("running", name, flush=True)
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NBDIR)}}).execute()
        f = OUT / f"{name}.html"
        f.write_text(page(name, nb, links), encoding="utf8")
        print("wrote", f, f"{f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
