"""Rebuild workshop/SPEECHIFY_OUTLINE.md from the deck: one entry per slide, in order.

Keeps every narration already written (matched by slide title) and writes one for dividers,
code maps, quizzes and any slide without one.

    python workshop/tools/build_outline.py
"""

import html as H
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
OUT = WS / "SPEECHIFY_OUTLINE.md"
LESSON = {"lesson1": "the four training steps", "lesson2": "download, read, check and fix",
          "lesson3": "choose, split, scale and cut", "lesson4": "shapes, baselines, linear, MLP and CNN",
          "lesson5": "train, validate, test once and save", "lesson6": "environment, reward, act and learn"}


#: narration written by hand for slides whose text alone does not make a good voice-over
EXTRA = {
    "The plant and its two control loops": (
        "This is the plant as the project drew it. Water flows from left to right. Tanks 1 and 2 get no air, so "
        "bacteria there remove nitrate. Tanks 3 to 5 are aerated, so other bacteria turn ammonium into nitrate, and the "
        "internal recycle carries that nitrate back to tank 1. The settler lets the sludge sink, and most of it goes back "
        "to the start. Our controllers move two things: the air in tank 5, to hold 2 milligrams of oxygen per litre, and "
        "the recycle pump, to hold 1 milligram of nitrate per litre in tank 2."),
    "The second held-out test: two fixes, one open problem": (
        "These are the project's latest results. The first held-out test found two flaws, so two fixes were made: "
        "anti-windup for the daily learning control, and a robust PID with a filter on the oxygen reading and half the "
        "gains. They were tested on new influent and new sensor noise, with the plan committed before any run. "
        "Anti-windup removed the learning control's harm under overload and cost nothing in normal weather. The robust PID "
        "never failed and was 83 percent better than the paper's PID on new influent. But with the official sensor noise, "
        "nothing we built beats the paper's own gentle PID. That is the open problem, and it is reported."),
}


def txt(s):
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def main():
    old = OUT.read_text(encoding="utf8")
    intro = old[:old.index("## Deck") + len("## Deck")]
    narr = {}
    for blk in re.split(r"(?=### Slide \d+ ·)", old)[1:]:
        t = re.match(r"### Slide \d+ · (.*)", blk).group(1).strip()
        m = re.search(r"\*\*Narration:\*\* (.*)", blk, re.S)
        if m:
            narr[t] = m.group(1).strip().split("\n\n")[0]
    h = (WS / "slides" / "index.html").read_text(encoding="utf8")
    deck = (WS / "slides" / "deck.js").read_text(encoding="utf8")
    qblock = deck[deck.index("const QUIZ = {"):deck.index("const QSTATE")]
    quiz = {}
    for key, body in re.findall(r"(\w+): \[(\[.*?\])\],?(?=\s*\n\s*(?:\w+: \[\[|\}))", qblock, re.S):
        quiz[key] = re.findall(r'\["(.*?)", "(.*?)"\]', body)
    secs = re.findall(r'<section class="(slide[^"]*)"([^>]*)>(.*?)</section>', h, re.S)
    intro = re.sub(r"\d+ slides", f"{len(secs)} slides", intro)
    out = [intro, "\n\n"]
    for i, (cls, attrs, body) in enumerate(secs, 1):
        m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", body, re.S)
        title = txt(m.group(1)) if m else f"Slide {i}"
        k = re.search(r'<div class="kicker">(.*?)</div>', body)
        kick = txt(k.group(1)) if k else ""
        lis = [txt(x) for x in re.findall(r"<li[^>]*>(.*?)</li>", body, re.S)][:6]
        anim = (re.search(r'data-anim="(\w+)"', attrs) or [None, ""])[1]
        key = (re.search(r'data-key="(\w+)"', attrs) or [None, ""])[1]
        bul = lis
        if anim == "quiz":
            qs = quiz.get(key, [])
            bul = [f"Q: {q} → A: {a}" for q, a in qs]
            lead = "Questions learners ask, with short answers." if key == "faq" else \
                f"Lesson {key[1:]} ends with three questions. Ask the room first, then reveal each answer."
            nar = lead + " " + " ".join(f"{q} {a}" for q, a in qs)
        elif anim == "codemap":
            scr = re.search(r"examples/(\w+\.py)", body).group(1)
            bul = [f"The same steps ({LESSON[key]}), each in its own colour, in four panels: the hands-on script, "
                   "the notebook, and the project's own code", f"Run it yourself: python workshop/examples/{scr}"]
            nar = (f"Where is the code? These four panels are real files. Each step of this lesson, {LESSON[key]}, "
                   "has one colour, and it lights up in every panel at once: first in the short script you can run, "
                   "then in the notebook, then in the project's own code. What you just learned is exactly what the "
                   "research code does.")
        elif "divider" in cls:
            goal = txt(re.search(r"After this lesson you can…</h3><p[^>]*>(.*?)</p>", body, re.S).group(1))
            term = txt(re.search(r"Terminal: (.*?)</p>", body, re.S).group(1))
            nb = txt(re.search(r"Notebook: (.*?)</p>", body, re.S).group(1))
            bul = [f"Goal: {goal}", f"Terminal: {term}", f"Notebook: {nb}"]
            nar = f"{kick.split('·')[0].strip()}: {title} After this lesson you will be able to say: {goal}"
        else:
            nar = EXTRA.get(title) or narr.get(title) or (f"{title}. " + (" ".join(lis[:3]) if lis else txt(body)[:300]))
        if not bul:
            p = re.search(r'<p class="sub"[^>]*>(.*?)</p>', body, re.S)
            bul = [txt(p.group(1))] if p else []
        out.append(f"### Slide {i} · {title}\n" + (f"- Section: {kick}\n" if kick else "")
                   + "".join(f"- {b}\n" for b in bul) + f"- Image: `stills/slide{i:02d}.jpg`\n"
                   + ("- Interactive in the HTML deck: use the live deck for this slide.\n" if anim else "")
                   + f"\n**Narration:** {nar}\n\n")
    OUT.write_text("".join(out), encoding="utf8")
    print("wrote", OUT, len(secs), "slides")


if __name__ == "__main__":
    main()
