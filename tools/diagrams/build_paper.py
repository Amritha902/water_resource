"""Paper diagrams, version 2: no titles (the caption names the figure), short labels,
straight orthogonal arrows, no crossings. Every box placed by hand. Writes v2/*.drawio."""
from dio import Diagram

SUB = lambda a, b: f"{a}<sub>{b}</sub>"
DOT = "shape=ellipse;aspect=fixed;fillColor=#1B2A33;strokeColor=none;"


def plant():
    d = Diagram("BSM1 plant", 1180, 360)
    t = []
    for k, x in enumerate((70, 225, 380, 535, 690)):
        air = k >= 2
        t.append(d.box(x, 110, 135, 140, f"<b>Tank {k+1}</b><br>{'aerobic' if air else 'anoxic'}<br>"
                       f"{'1333' if air else '1000'} m³", "tankair" if air else "tank"))
    s = d.box(890, 110, 170, 100, "<b>Settler</b><br>10 layers", "settler")
    d.box(930, 210, 90, 50, "", "hopper")
    d.line((8, 180), (68, 180), "")
    d.box(4, 140, 62, 28, "influent", "textc", "fontSize=13;")
    for a, b in zip(t, t[1:]):
        d.edge(a, b, ex=(1, 0.5), en=(0, 0.5))
    d.edge(t[4], s, ex=(1, 0.5), en=(0, 0.7))
    d.line((1060, 130), (1172, 130), "effluent")
    d.edge(t[4], t[0], f"internal recycle {SUB('<i>Q</i>', 'a')}  (input 2)", kind="new",
           ex=(0.5, 0), en=(0.5, 0), pts=((757, 60), (137, 60)))
    d.line((975, 262), (137, 252), f"return sludge {SUB('<i>Q</i>', 'r')}", kind="fb", pts=((975, 330), (137, 330)))
    d.line((975, 300), (1172, 300), "waste", kind="fb")
    d.box(971, 296, 8, 8, "", "white", DOT)
    for k, x in enumerate((447, 602, 757)):
        last = k == 2
        d.line((x, 300), (x, 252), "", kind="new" if last else "air")
        d.box(x + 6, 268, 130, 26, ("<b>" + SUB("<i>K<sub>L</sub>a</i>", "5") + "</b> (input 1)") if last
              else "<i>K<sub>L</sub>a</i> = 240", "text", "fontSize=13;" + ("fontColor=#0E3B2A;" if last else ""))
    d.box(245, 122, 95, 26, SUB("<i>S</i>", "NO,2"), "white", "fontSize=13;")
    d.box(710, 122, 95, 26, SUB("<i>S</i>", "O,5"), "white", "fontSize=13;")
    d.save("paper/plant.drawio")


def maacc():
    d = Diagram("MAACC (paper 1)", 930, 450)
    d.box(20, 20, 640, 410, "Agent <i>i</i>  (<i>i</i> = 1 oxygen, <i>i</i> = 2 nitrate)", "group")
    act = d.box(50, 80, 170, 60, "Action network<br>Δ<i>a<sub>i</sub></i> = π<sub><i>i</i></sub>(<i>z<sub>i</sub></i>)", "paper")
    itg = d.box(270, 80, 130, 60, "Integrate<br><i>a<sub>i</sub></i> ← <i>a<sub>i</sub></i> + Δ<i>a<sub>i</sub></i>", "paper")
    sm = d.box(490, 85, 50, 50, "Σ", "sum")
    pid = d.box(440, 200, 150, 60, "Expert PID<br><i>b<sub>i</sub></i>(<i>z<sub>i</sub></i>)", "paper")
    cr = d.box(170, 230, 190, 60, "Critic <i>Q<sub>i</sub></i>(<i>z</i>, <i>a</i><sub>1</sub>, <i>a</i><sub>2</sub>)<br>one-step TD update", "paper")
    z = d.box(195, 350, 140, 50, "<i>z</i> = <i>y</i> − <i>r</i>", "white")
    pl = d.box(730, 70, 170, 80, "<b>BSM1</b>", "plant")
    d.edge(act, itg, ex=(1, 0.5), en=(0, 0.5))
    d.edge(itg, sm, "<i>a<sub>i</sub></i>", ex=(1, 0.5), en=(0, 0.5))
    d.edge(pid, sm, "<i>b<sub>i</sub></i>", ex=(0.5, 0), en=(0.5, 1))
    d.edge(sm, pl, "<i>u<sub>i</sub></i> = <i>a<sub>i</sub></i> + <i>b<sub>i</sub></i>", ex=(1, 0.5), en=(0, 0.5))
    d.edge(cr, act, "∂<i>Q</i>/∂<i>a</i>", kind="learn", ex=(0, 0.5), en=(0.5, 1), pts=((135, 260),))
    d.edge(pl, z, "<i>y</i>", kind="fb", ex=(0.5, 1), en=(1, 0.5), pts=((815, 375),))
    d.edge(z, cr, "", kind="fb", ex=(0.5, 0), en=(0.5, 1))
    d.edge(z, act, "", kind="fb", ex=(0, 0.5), en=(0, 0.5), pts=((32, 375), (32, 110)))
    d.save("paper/maacc.drawio")


def ours():
    d = Diagram("Our controller", 1100, 420)
    d.box(14, 156, 70, 28, "<i>r</i> = 2", "textc", "fontSize=14;")
    s1 = d.box(110, 155, 50, 50, "Σ", "sum")
    d.line((84, 180), (108, 180), "")
    pid = d.box(205, 150, 150, 60, "<b>Tuned PID</b>", "new")
    s2 = d.box(450, 155, 50, 50, "Σ", "sum")
    lim = d.box(570, 150, 130, 60, "Limit<br>0–240 d<sup>−1</sup>", "white")
    pl = d.box(800, 140, 170, 80, "<b>BSM1</b><br>tank 5", "plant")
    ilc = d.box(400, 30, 150, 60, "<b>Daily ILC</b>", "new")
    sen = d.box(820, 290, 130, 60, "Sensor<br>delay + noise", "white")
    flt = d.box(540, 290, 150, 60, "<b>Filter</b><br>τ = 0.75 min", "new")
    d.edge(s1, pid, "<i>e</i>", ex=(1, 0.5), en=(0, 0.5))
    d.edge(pid, s2, SUB("<i>u</i>", "PID"), ex=(1, 0.5), en=(0, 0.5))
    d.edge(s2, lim, "", ex=(1, 0.5), en=(0, 0.5))
    d.edge(lim, pl, "<i>K<sub>L</sub>a</i><sub>5</sub>", ex=(1, 0.5), en=(0, 0.5))
    d.line((970, 180), (1090, 180), SUB("<i>S</i>", "O,5"))
    d.edge(pid, ilc, SUB("<i>u</i>", "PID"), kind="new", ex=(0.5, 0), en=(0, 0.5), pts=((280, 60),))
    d.edge(ilc, s2, "<i>f</i>(<i>t</i>)", kind="new", ex=(0.5, 1), en=(0.5, 0))
    d.edge(lim, ilc, "at limit", kind="upd", ex=(0.5, 0), en=(1, 0.5), pts=((635, 60),))
    d.edge(pl, sen, "", kind="fb", ex=(0.5, 1), en=(0.5, 0))
    d.edge(sen, flt, "", kind="fb", ex=(0, 0.5), en=(1, 0.5))
    d.edge(flt, s1, SUB("<i>y</i>", "m"), kind="fb", ex=(0, 0.5), en=(0.5, 1), pts=((135, 320),))
    d.box(88, 150, 20, 20, "+", "textc", "fontSize=14;")
    d.box(140, 208, 20, 20, "−", "textc", "fontSize=16;")
    d.box(20, 380, 18, 18, "", "new")
    d.box(46, 376, 300, 26, "added in this work", "text", "fontSize=13;")
    d.save("paper/ours.drawio")


def ilc():
    d = Diagram("Daily ILC", 1060, 230)
    a = d.box(20, 30, 280, 100, "<b>Day <i>d</i></b><br>record " + SUB("<i>u</i>", "PID") + "(<i>j</i>), <i>j</i> = 1 … 1920", "paper")
    m = d.box(390, 30, 280, 100, "<b>Midnight</b><br><i>f</i> ← 𝒮[<i>f</i> + β ℒ(" + SUB("<i>u</i>", "PID") + " − ū)]", "new")
    b = d.box(760, 30, 280, 100, "<b>Day <i>d</i>+1</b><br><i>u</i> = " + SUB("<i>u</i>", "PID") + " + <i>f</i>(<i>j</i>)", "paper")
    d.edge(a, m, ex=(1, 0.5), en=(0, 0.5))
    d.edge(m, b, ex=(1, 0.5), en=(0, 0.5))
    d.edge(b, a, "next day", kind="upd", ex=(0.5, 1), en=(0.5, 1), pts=((900, 190), (160, 190)))
    d.save("paper/ilc.drawio")


for f in (plant, maacc, ours, ilc):
    f()
print("ok")
