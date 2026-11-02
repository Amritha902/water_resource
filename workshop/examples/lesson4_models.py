"""Lesson 4, hands-on: the shapes choose the model; look inside it, layer by layer.

Run it:   python workshop/examples/lesson4_models.py      (needs workshop/data/prepared.npz: lesson 3 or notebook 1)

  1 SHAPES     (N, 96, 3) → (N, 8): a sequence in, 8 numbers out → regression on a sequence
  2 BASELINES  rules with no learning, the bar every model must clear
  3 LINEAR     288 inputs × 8 outputs: count the knobs
  4 MLP        layers, ReLU, and the shape after every layer
  5 CNN        a causal convolution: the time order, few knobs, never the future
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

d = np.load(Path(__file__).resolve().parents[1] / "data" / "prepared.npz")
X = {s: d[f"X_{s}"] for s in ("train", "val")}
Y = {s: d[f"y_{s}"] for s in ("train", "val")}
MEAN, STD = d["mean"], d["std"]
flow = lambda z: np.exp(np.asarray(z) * STD[0] + MEAN[0])                 # back to m³/d
mae = lambda p: np.abs(flow(p) - flow(Y["val"])).mean(0)                 # error per horizon, m³/d
count = lambda m: sum(p.numel() for p in m.parameters())

# ── 1 SHAPES
print("1 SHAPES     X", X["train"].shape, "→ y", Y["train"].shape, "  (examples, time steps, channels) → (examples, 8 future flows)")

# ── 2 BASELINES: no learning at all
for name, pred in (("persistence (= now)", np.repeat(X["val"][:, -1:, 0], 8, 1)),
                   ("same time yesterday", X["val"][:, :8, 0])):
    e = mae(pred)
    print(f"2 BASELINE   {name:22s} error 15 min {e[0]:6.0f}   2 h {e[-1]:6.0f} m³/d   knobs: 0")

# ── 3 LINEAR: flatten the window into 288 numbers; ŷ = W·x + b
lin = Ridge(alpha=1.0).fit(X["train"].reshape(len(X["train"]), -1), Y["train"])
e = mae(lin.predict(X["val"].reshape(len(X["val"]), -1)))
print(f"3 LINEAR     knobs {lin.coef_.size + lin.intercept_.size}  (= 288 × 8 + 8)   error 15 min {e[0]:6.0f}   2 h {e[-1]:6.0f} m³/d")

# ── 4 MLP: the same, with two hidden layers and ReLU in between
torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(288, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
print(f"4 MLP        knobs {count(mlp)}  = (288·64 + 64) + (64·64 + 64) + (64·8 + 8)")
h = torch.from_numpy(X["val"][:1])
print(f"             {'input':26s} {tuple(h.shape)}")
for layer in mlp:
    h = layer(h)
    extra = f"   {(h == 0).float().mean():.0%} switched off by ReLU" if isinstance(layer, nn.ReLU) else ""
    print(f"             {str(layer)[:26]:26s} {tuple(h.shape)}{extra}")


# ── 5 CNN: a small filter slides along time; padding only on the left = causal
class CausalBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.pad, self.conv = 2 * dilation, nn.Conv1d(ch, ch, 3, dilation=dilation)

    def forward(self, x):
        return x + F.gelu(self.conv(F.pad(x, (self.pad, 0))))           # pad LEFT only: no future


cnn = nn.Sequential(nn.Conv1d(3, 24, 1), *[CausalBlock(24, d) for d in (1, 2, 4, 8, 16, 32)])
head = nn.Linear(24, 8)
print(f"5 CNN        knobs {count(cnn) + count(head)}  (fewer than the MLP, for the same input)")
x = torch.from_numpy(X["val"][:1]).transpose(1, 2)                      # (batch, channels, time)
out = cnn(x)
print(f"             (1, 3, 96) → convolutions → {tuple(out.shape)} → last step {tuple(out[:, :, -1].shape)} → head {tuple(head(out[:, :, -1]).shape)}")
x2 = x.clone(); x2[:, :, 50] += 5.0                                     # change one moment in the middle
changed = (cnn(x2) - out).abs().sum(1)[0] > 1e-6
print(f"             change time step 50 → outputs change from step {int(changed.nonzero()[0])} on, never before it (causal)")

# ✏️ Try: hidden layers of 512 instead of 64. How many knobs? Compare with ~1 900 training examples.
