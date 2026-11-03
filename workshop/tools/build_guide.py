"""Build the teaching guide: workshop/TEACHING_GUIDE.md and workshop/teaching_guide.html.

The source, workshop/guide/TEACHING_GUIDE.src.md, is the guide as written, with two kinds of marker
that this script fills in, so the guide always shows the real code and the real results:

  {{code:PATH|FIRST|LAST}}   the lines of PATH from the first line containing FIRST
                             to the next line containing LAST, with their line numbers
  {{run:SCRIPT}}             the output of python workshop/examples/SCRIPT.py, run now

The lesson scripts are run once each, in order (lessons 4 and 5 read the file lesson 3 writes).

    python workshop/tools/build_guide.py          # about 30 s
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "workshop"
SRC = WS / "guide" / "TEACHING_GUIDE.src.md"
SCRIPTS = ["lesson1_knobs", "lesson2_load_and_clean", "lesson3_prepare", "lesson4_models",
           "lesson5_train_and_evaluate", "lesson6_rl"]

sys.path.insert(0, str(Path(__file__).parent))
from notebooks_to_html import highlight  # noqa: E402


def code_block(path: str, first: str, last: str) -> str:
    lines = (ROOT / path).read_text(encoding="utf8").splitlines()
    a = next((i for i, l in enumerate(lines) if first in l), None)
    if a is None:
        raise ValueError(f"{path}: no line containing {first!r}")
    b = next((i for i in range(a, len(lines)) if last in lines[i]), None)
    if b is None:
        raise ValueError(f"{path}: no line containing {last!r} after {first!r}")
    body = "\n".join(lines[a:b + 1])
    return f"<!--code {path} {a + 1}-{b + 1}-->\n```python\n{body}\n```\n<p class=\"src\">{path}, lines {a + 1}–{b + 1}</p>\n"


def run_outputs() -> dict:
    out = {}
    (WS / "data" / "prepared.npz").unlink(missing_ok=True)       # lesson 3 rebuilds it: a full run, from scratch
    for s in SCRIPTS:
        print("running", s, flush=True)
        r = subprocess.run([sys.executable, str(WS / "examples" / f"{s}.py")], cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise RuntimeError(f"{s} failed:\n{r.stderr[-2000:]}")
        out[s] = r.stdout.rstrip().replace(str(ROOT) + "/", "")
    return out


def fill(src: str, outputs: dict) -> str:
    src = re.sub(r"\{\{code:([^|}]+)\|([^|}]+)\|([^}]+)\}\}", lambda m: code_block(*[g.strip() for g in m.groups()]), src)
    return re.sub(r"\{\{run:(\w+)\}\}", lambda m: f"```text\n{outputs[m.group(1)]}\n```\n<p class=\"src\">Output of <code>python workshop/examples/{m.group(1)}.py</code>, from the run that built this guide</p>\n", src)


CSS = """
/* layout: one reading column with a sticky contents rail on wide screens; an engineer's handbook */
:root{
  --bg:#f6f8f9; --paper:#ffffff; --ink:#17232b; --ink2:#43525c; --muted:#76838c; --rule:#d9e1e5;
  --water:#1f6f8b; --water-soft:#e3f0f4; --note:#fff7e3; --note-rule:#d9a441; --code-bg:#0f1d24; --code-fg:#dde8ee;
  --k:#b9a3ff; --s:#f2cf7a; --c:#7f939e; --n:#72d6c9; --b:#8cc4ff; --f:#cfe3ff;
  --display:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e171c; --paper:#132027; --ink:#e3ecf0; --ink2:#b3c2ca; --muted:#8597a1; --rule:#24363f;
  --water:#6cc0dc; --water-soft:#16323d; --note:#2a2414; --note-rule:#b98c32; --code-bg:#0a1216; --code-fg:#dde8ee; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e171c; --paper:#132027; --ink:#e3ecf0; --ink2:#b3c2ca; --muted:#8597a1; --rule:#24363f;
  --water:#6cc0dc; --water-soft:#16323d; --note:#2a2414; --note-rule:#b98c32; --code-bg:#0a1216; --code-fg:#dde8ee; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{background:var(--bg);color:var(--ink);font:16.5px/1.65 var(--body);
  padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px));padding-block:0}
.wrap{max-width:1180px;margin:0 auto;display:flex;gap:40px;align-items:flex-start}
nav.toc{position:sticky;top:calc(env(safe-area-inset-top, 0px) + 16px);flex:0 0 240px;max-height:calc(100vh - 32px);overflow:auto;
  padding:28px 0 20px;font-size:13.5px;line-height:1.4}
nav.toc .label{font:600 11.5px/1 var(--display);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
nav.toc a{display:block;color:var(--ink2);text-decoration:none;padding:4px 0 4px 10px;border-left:2px solid var(--rule)}
nav.toc a.h3{padding-left:22px;color:var(--muted);font-size:12.5px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--water);border-color:var(--water);outline:none}
main{flex:1;min-width:0;max-width:760px;padding-block:28px 96px}
header.cover{border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:8px}
header.cover .eyebrow{font:600 12px/1 var(--display);letter-spacing:.14em;text-transform:uppercase;color:var(--water)}
h1{font:600 40px/1.1 var(--display);margin:10px 0 12px;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 28px/1.2 var(--display);margin:56px 0 10px;padding-top:14px;border-top:3px solid var(--water);text-wrap:balance;scroll-margin-top:16px}
h3{font:600 19px/1.3 var(--display);margin:34px 0 6px;text-wrap:balance;scroll-margin-top:16px}
h3 .slide{font:500 12px/1 var(--mono);color:var(--water);background:var(--water-soft);padding:3px 7px;border-radius:4px;margin-right:8px;vertical-align:2px}
h4{font:600 15px/1.3 var(--body);margin:22px 0 4px;color:var(--ink2);text-transform:uppercase;letter-spacing:.06em;font-size:12.5px}
p,li{max-width:68ch}
a{color:var(--water)}
code{font:0.88em var(--mono);background:var(--water-soft);padding:1px 5px;border-radius:3px}
pre{background:var(--code-bg);color:var(--code-fg);font:13.2px/1.55 var(--mono);padding:14px 16px;border-radius:6px;overflow-x:auto;margin:12px 0 4px}
pre code{background:none;padding:0;font:inherit;color:inherit}
pre .k{color:var(--k)}pre .s{color:var(--s)}pre .c{color:var(--c);font-style:italic}pre .n{color:var(--n)}pre .b{color:var(--b)}pre .f{color:var(--f)}
pre.text{background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-left:4px solid var(--water)}
p.src{font:12px/1.4 var(--mono);color:var(--muted);margin:2px 0 14px;overflow-wrap:anywhere}
p code,li code,td code{overflow-wrap:anywhere}
main p,main li{overflow-wrap:break-word}
blockquote{margin:16px 0;padding:10px 16px;background:var(--note);border-left:4px solid var(--note-rule);border-radius:0 6px 6px 0}
blockquote p{margin:4px 0}
table{border-collapse:collapse;margin:12px 0;font-size:14.5px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--rule);padding:6px 10px;text-align:left;vertical-align:top}
th{font:600 12.5px/1.3 var(--body);color:var(--ink2)}
hr{border:none;border-top:1px solid var(--rule);margin:30px 0}
img{max-width:100%;border:1px solid var(--rule);border-radius:6px;background:#fff}
@media (max-width:900px){nav.toc{display:none}h1{font-size:31px}h2{font-size:24px}}
"""


def to_html(md_text: str) -> str:
    toc = []
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

    def head(m):
        level, attrs, inner = m.group(1), m.group(2) or "", m.group(3)
        text = re.sub("<[^>]+>", "", inner)
        anchor = re.sub(r"[^a-z0-9]+", "-", html.unescape(text).lower()).strip("-")[:50]
        if level in ("2", "3"):
            toc.append((level, html.unescape(text), anchor))
        inner = re.sub(r"^Slide (\d+(?:–\d+)?) · ", r'<span class="slide">slide \1</span>', inner)
        return f'<h{level} id="{anchor}">{inner}</h{level}>'
    body = re.sub(r"<h([1-4])( [^>]*)?>(.*?)</h\1>", head, body)

    def code(m):
        lang, src = m.group(1), html.unescape(m.group(2))
        if lang == "python":
            return f"<pre><code>{highlight(src)}</code></pre>"
        return f'<pre class="text"><code>{html.escape(src)}</code></pre>'
    body = re.sub(r'<pre><code class="language-(\w+)">(.*?)</code></pre>', code, body, flags=re.S)
    body = re.sub(r"<!--code [^>]*-->", "", body)
    nav = "".join(f'<a class="h{lv}" href="#{a}">{html.escape(t)}</a>' for lv, t, a in toc)
    first_h1 = re.search(r"<h1[^>]*>.*?</h1>", body, re.S)
    if first_h1:
        cover = body[:first_h1.end()]
        body = '<header class="cover"><div class="eyebrow">Teaching guide · the BSM1 workshop</div>' + cover + "</header>" + body[first_h1.end():]
    return (f'<title>BSM1 Lesson Guide</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">\n'
            f"<style>{CSS}</style>\n"
            f'<div class="wrap"><nav class="toc" aria-label="Contents"><div class="label">Contents</div>{nav}</nav><main>{body}</main></div>\n')


def main():
    outputs = run_outputs()
    md_text = fill(SRC.read_text(encoding="utf8"), outputs)
    (WS / "TEACHING_GUIDE.md").write_text(re.sub(r"<!--code [^>]*-->\n", "", md_text), encoding="utf8")
    (WS / "teaching_guide.html").write_text(to_html(md_text), encoding="utf8")
    print("wrote", WS / "TEACHING_GUIDE.md", "and", WS / "teaching_guide.html")


if __name__ == "__main__":
    main()
