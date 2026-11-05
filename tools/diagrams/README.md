# Architecture diagrams (draw.io)

The diagrams in `docs/diagrams/*.drawio` are drawn by hand. `build.py` places
every box, arrow and label by coordinates, using the helper `dio.py`. Each
file is a plain, uncompressed draw.io file: open it at
https://app.diagrams.net (File → Open) or in draw.io desktop to edit or
export it.

The PNGs in `docs/diagrams/` were rendered with mxGraph 4.2.2, draw.io's
own rendering engine, in a headless browser (`render.js`, which needs
`npm install mxgraph playwright`).

```bash
python build.py                      # writes d0_plant.drawio ... d5_method.drawio
node render.js d*.drawio             # writes the PNGs
NOTITLE=1 python build.py && node render.js notitle/*.drawio   # versions without titles, used on the slides
```

`audit.js` checks every diagram inside mxGraph for problems: text that
overflows its box, labels lying on a box, boxes that overlap, lines that
cross a box, and arrowheads with too little room. All six diagrams pass;
the only overlaps it reports are the sensors, drawn inside their tanks on
purpose.

```bash
node audit.js d*.drawio
```

`deck.js` builds `docs/Paper1_Architecture_Deck.pptx`. It needs pptxgenjs and
expects two files in its folder:
* `sizes.json`: the cropped title-less PNGs and their sizes;
* `numbers.json`: the paired percentages, computed from the result files
  with the functions in `experiments/45_paper1_figures.py`.

## Paper versions

`build_paper.py` writes the paper diagrams in `docs/diagrams/paper/` (no titles, short labels, no crossing arrows). Run it from this folder, then `node render.js paper/*.drawio` and `node audit.js paper/*.drawio`.
