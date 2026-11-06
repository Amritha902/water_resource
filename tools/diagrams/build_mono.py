"""Black-and-white paper diagrams in classic control block-diagram style. Writes mono/*.drawio."""
import dio
from dio import Diagram

F = "fontFamily=Times New Roman;"
BOX = "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;strokeWidth=1.2;fontColor=#000000;fontSize=17;" + F
dio.STY.update({
    "m": BOX,
    "mdash": BOX + "dashed=1;",
    "msum": "shape=ellipse;perimeter=ellipsePerimeter;aspect=fixed;html=1;fillColor=#FFFFFF;strokeColor=#000000;"
            "strokeWidth=1.2;fontColor=#000000;fontSize=18;" + F,
    "mtxt": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;"
            "fontColor=#000000;fontSize=16;" + F,
    "mdot": "shape=ellipse;aspect=fixed;fillColor=#000000;strokeColor=none;",
    "mtank": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;strokeWidth=1.2;fontColor=#000000;"
             "fontSize=16;verticalAlign=bottom;spacingBottom=8;" + F,
    "mhop": "shape=triangle;perimeter=trianglePerimeter;direction=south;html=1;fillColor=#FFFFFF;strokeColor=#000000;strokeWidth=1.2;",
})
E = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=6;strokeColor=#000000;"
     "strokeWidth=1.2;fontColor=#000000;fontSize=16;labelBackgroundColor=#FFFFFF;" + F)
dio.EDGE_STY.update({"m": E, "mdash": E + "dashed=1;"})


def dot(d, x, y):
    d.box(x - 4, y - 4, 8, 8, "", "mdot")


def sign(d, x, y, s):
    d.box(x, y, 16, 16, s, "mtxt", "fontSize=15;")


def base():
    d = Diagram("MAACC", 760, 380)
    d.line((15, 260), (66, 260), "<i>r<sub>i</sub></i>", "m")
    s1 = d.box(68, 238, 44, 44, "Σ", "msum")
    pid = d.box(170, 235, 120, 50, "PID", "m")
    s2 = d.box(398, 238, 44, 44, "Σ", "msum")
    pl = d.box(500, 225, 140, 70, "BSM1", "m")
    act = d.box(170, 125, 120, 50, "Action net π<sub><i>i</i></sub>", "m")
    itg = d.box(360, 125, 120, 50, "1 / (1 − <i>z</i><sup>−1</sup>)", "m")
    cr = d.box(170, 20, 120, 60, "Critic <i>Q<sub>i</sub></i>", "mdash")
    d.edge(s1, pid, "", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(pid, s2, "<i>b<sub>i</sub></i>", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(s2, pl, "<i>u<sub>i</sub></i>", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(act, itg, "Δ<i>a<sub>i</sub></i>", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(itg, s2, "<i>a<sub>i</sub></i>", "m", ex=(0.5, 1), en=(0.5, 0))
    d.edge(cr, act, "∂<i>Q<sub>i</sub></i>/∂<i>a<sub>i</sub></i>", "mdash", ex=(0.5, 1), en=(0.5, 0))
    d.line((140, 260), (168, 150), "", "m", pts=((140, 150),))
    dot(d, 140, 260)
    d.line((640, 260), (745, 260), "", "m")
    d.box(708, 232, 30, 22, "<i>y<sub>i</sub></i>", "mtxt")
    d.box(144, 232, 26, 22, "<i>e<sub>i</sub></i>", "mtxt")
    d.line((700, 260), (90, 284), "", "m", pts=((700, 350), (90, 350)))
    dot(d, 700, 260)
    sign(d, 48, 236, "+")
    sign(d, 96, 288, "−")
    d.save("mono/base.drawio")


def proposed():
    d = Diagram("Proposed", 860, 420)
    d.line((15, 260), (66, 260), "<i>r</i>", "m")
    s1 = d.box(68, 238, 44, 44, "Σ", "msum")
    pid = d.box(170, 235, 120, 50, "PID", "m")
    s2 = d.box(398, 238, 44, 44, "Σ", "msum")
    sat = d.box(490, 235, 80, 50, "sat", "m")
    pl = d.box(630, 225, 130, 70, "BSM1", "m")
    ilc = d.box(360, 125, 120, 50, "Daily ILC", "m")
    sen = d.box(630, 345, 130, 50, "Sensor", "m")
    flt = d.box(380, 345, 140, 50, "1 / (τ<i>s</i> + 1)", "m")
    d.edge(s1, pid, "<i>e</i>", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(pid, s2, "", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(s2, sat, "", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(sat, pl, "<i>u</i>", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(ilc, s2, "<i>f</i>", "m", ex=(0.5, 1), en=(0.5, 0))
    d.line((330, 260), (358, 150), "", "m", pts=((330, 150),))
    dot(d, 330, 260)
    d.box(296, 236, 40, 20, "<i>u</i><sub>PID</sub>", "mtxt", "fontSize=14;")
    d.edge(sat, ilc, "AW", "mdash", ex=(0.5, 0), en=(1, 0.5), pts=((530, 150),))
    d.line((760, 260), (845, 260), "", "m")
    d.box(808, 232, 30, 22, "<i>y</i>", "mtxt")
    d.line((800, 260), (762, 370), "", "m", pts=((800, 370),))
    dot(d, 800, 260)
    d.edge(sen, flt, "", "m", ex=(0, 0.5), en=(1, 0.5))
    d.edge(flt, s1, "<i>y</i><sub>m</sub>", "m", ex=(0, 0.5), en=(0.5, 1), pts=((90, 370),))
    sign(d, 48, 236, "+")
    sign(d, 96, 288, "−")
    d.save("mono/proposed.drawio")


def plant():
    d = Diagram("BSM1", 1180, 330)
    t = []
    for k, x in enumerate((70, 225, 380, 535, 690)):
        air = k >= 2
        t.append(d.box(x, 100, 135, 130, f"Tank {k+1}<br>{'aerobic' if air else 'anoxic'}", "mtank"))
    s = d.box(890, 100, 170, 90, "Settler", "m")
    d.box(930, 190, 90, 50, "", "mhop")
    d.line((8, 165), (68, 165), "<i>Q</i><sub>in</sub>", "m")
    for a, b in zip(t, t[1:]):
        d.edge(a, b, "", "m", ex=(1, 0.5), en=(0, 0.5))
    d.edge(t[4], s, "", "m", ex=(1, 0.5), en=(0, 0.72))
    d.line((1060, 120), (1172, 120), "effluent", "m")
    d.edge(t[4], t[0], "<i>Q</i><sub>a</sub> (manipulated)", "m", ex=(0.5, 0), en=(0.5, 0), pts=((757, 50), (137, 50)))
    d.line((975, 242), (137, 232), "<i>Q</i><sub>r</sub>", "m", pts=((975, 305), (137, 305)))
    d.line((975, 275), (1172, 275), "<i>Q</i><sub>w</sub>", "m")
    dot(d, 975, 275)
    for k, x in enumerate((447, 602, 757)):
        d.line((x, 280), (x, 232), "", "m" if k == 2 else "mdash")
        d.box(x + 4, 248, 130, 26, ("<i>K<sub>L</sub>a</i><sub>5</sub> (manipulated)" if k == 2 else "<i>K<sub>L</sub>a</i> = 240 d<sup>−1</sup>"),
              "mtxt", "align=left;fontSize=15;")
    d.box(240, 112, 105, 26, "<i>S</i><sub>NO,2</sub>", "m", "fontSize=15;")
    d.box(705, 112, 105, 26, "<i>S</i><sub>O,5</sub>", "m", "fontSize=15;")
    d.save("mono/plant.drawio")


for f in (base, proposed, plant):
    f()
print("ok")
