"""Tiny helper to write draw.io (.drawio, uncompressed mxGraphModel) files by hand-placed geometry."""

from xml.sax.saxutils import escape
import os

NOTITLE = os.environ.get("NOTITLE") == "1"

FONT = "fontFamily=Helvetica;"
BASE = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;" + FONT + "fontSize=14;"
STY = {
    "paper": BASE + "fillColor=#F1F1EE;strokeColor=#7A7A74;fontColor=#1B2A33;strokeWidth=1.5;",
    "new": BASE + "fillColor=#E2F3EA;strokeColor=#1B7F5A;fontColor=#0E3B2A;strokeWidth=2.5;",
    "plant": BASE + "fillColor=#DCEEF4;strokeColor=#1C6E8C;fontColor=#0B3C49;strokeWidth=1.5;",
    "learn": BASE + "fillColor=#FDEEE4;strokeColor=#C8611E;fontColor=#5A2A0A;strokeWidth=2;dashed=1;",
    "bad": BASE + "fillColor=#FBE4E4;strokeColor=#B03A2E;fontColor=#6B1A12;strokeWidth=1.5;",
    "white": BASE + "fillColor=#FFFFFF;strokeColor=#1B2A33;fontColor=#1B2A33;strokeWidth=1.5;",
    "group": "rounded=1;arcSize=3;whiteSpace=wrap;html=1;" + FONT + "fillColor=#FAFAF8;strokeColor=#A7B1B5;"
             "dashed=1;verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontStyle=1;fontSize=14;fontColor=#33444C;",
    "groupnew": "rounded=1;arcSize=3;whiteSpace=wrap;html=1;" + FONT + "fillColor=#F4FBF7;strokeColor=#1B7F5A;"
                "dashed=1;verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontStyle=1;fontSize=14;fontColor=#0E3B2A;",
    "text": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;" + FONT
            + "fontSize=13;fontColor=#4A5A62;",
    "textc": "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;" + FONT
             + "fontSize=13;fontColor=#4A5A62;",
    "title": "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;" + FONT
             + "fontSize=18;fontStyle=1;fontColor=#0B3C49;",
    "sum": "shape=ellipse;perimeter=ellipsePerimeter;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#1B2A33;strokeWidth=1.8;"
           + FONT + "fontSize=20;fontStyle=1;fontColor=#1B2A33;",
    "tag": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#1B7F5A;strokeColor=none;" + FONT
           + "fontSize=11;fontStyle=1;fontColor=#FFFFFF;",
    "tagpaper": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#7A7A74;strokeColor=none;" + FONT
                + "fontSize=11;fontStyle=1;fontColor=#FFFFFF;",
    "tank": "rounded=0;whiteSpace=wrap;html=1;" + FONT + "fontSize=13;fillColor=#DCEEF4;strokeColor=#1C6E8C;"
            "strokeWidth=1.8;fontColor=#0B3C49;verticalAlign=bottom;spacingBottom=8;",
    "tankair": "rounded=0;whiteSpace=wrap;html=1;" + FONT + "fontSize=13;fillColor=#C5E4EE;strokeColor=#1C6E8C;"
               "strokeWidth=1.8;fontColor=#0B3C49;verticalAlign=bottom;spacingBottom=8;",
    "settler": "rounded=0;whiteSpace=wrap;html=1;" + FONT + "fontSize=13;fillColor=#E8E4D6;"
               "strokeColor=#7A6A40;strokeWidth=1.8;fontColor=#3A3420;",
    "hopper": "shape=triangle;perimeter=trianglePerimeter;direction=south;whiteSpace=wrap;html=1;fillColor=#E8E4D6;"
              "strokeColor=#7A6A40;strokeWidth=1.8;",
}
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;html=1;endArrow=block;endFill=1;endSize=7;"
        "strokeColor=#1B2A33;strokeWidth=1.8;" + FONT + "fontSize=13;fontColor=#1B2A33;labelBackgroundColor=#FFFFFF;")
EDGE_STY = {
    "sig": EDGE,
    "fb": EDGE + "strokeColor=#1C6E8C;fontColor=#1C6E8C;",
    "learn": EDGE + "strokeColor=#C8611E;dashed=1;fontColor=#8A3E10;",
    "new": EDGE + "strokeColor=#1B7F5A;fontColor=#0E3B2A;strokeWidth=2.2;",
    "upd": EDGE + "strokeColor=#7A7A74;dashed=1;fontColor=#55606A;strokeWidth=1.4;",
    "air": EDGE + "strokeColor=#5AA9C8;endArrow=block;strokeWidth=1.6;fontColor=#1C6E8C;",
}


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.cells = []
        self.n = 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, label, kind="paper", extra=""):
        i = self._id()
        if NOTITLE and kind == "title":
            return i
        self.cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{STY[kind]}{extra}" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
                          f'</mxCell>')
        return i

    def edge(self, s, t, label="", kind="sig", pts=(), ex=None, en=None, extra=""):
        i = self._id()
        st = EDGE_STY[kind] + extra
        if ex is not None:
            st += f"exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;"
        if en is not None:
            st += f"entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;"
        p = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
        arr = f'<Array as="points">{p}</Array>' if pts else ""
        self.cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" edge="1" parent="1" '
                          f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
        return i

    def line(self, p0, p1, label="", kind="sig", pts=(), extra=""):
        """Edge between free points (no terminals)."""
        i = self._id()
        p = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
        arr = f'<Array as="points">{p}</Array>' if pts else ""
        self.cells.append(f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{EDGE_STY[kind]}{extra}" edge="1" '
                          f'parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{p0[0]}" y="{p0[1]}" as="sourcePoint"/>'
                          f'<mxPoint x="{p1[0]}" y="{p1[1]}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')
        return i

    def xml(self):
        body = "".join(self.cells)
        return (f'<mxfile host="drawio" agent="hand-written"><diagram id="{self.name}" name="{self.name}">'
                f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
                f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')

    def save(self, path):
        if NOTITLE:
            path = os.path.join("notitle", path)
        open(path, "w").write(self.xml())
