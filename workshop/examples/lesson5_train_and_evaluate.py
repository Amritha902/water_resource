"""Lesson 5, hands-on: train with validation, catch overfitting, test once, save with the scaler.

Run it:   python workshop/examples/lesson5_train_and_evaluate.py      (needs workshop/data/prepared.npz; about 20 s)

  1 TRAIN     the four-step loop (lesson 1) on batches, for many epochs
  2 VALIDATE  after every epoch, on data not trained on; keep the best epoch
  3 TEST      once, at the very end, on data nothing was chosen on
  4 SAVE      weights + scaler + settings together; reload and predict from raw values
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parents[1] / "data"
d = np.load(DATA / "prepared.npz")
X = {s: torch.from_numpy(d[f"X_{s}"]) for s in ("train", "val", "test")}
Y = {s: torch.from_numpy(d[f"y_{s}"]) for s in ("train", "val", "test")}
MEAN, STD = d["mean"], d["std"]
flow = lambda z: np.exp(np.asarray(z) * STD[0] + MEAN[0])
mae2h = lambda pred, s: float(np.abs(flow(pred)[:, -1] - flow(Y[s].numpy())[:, -1]).mean())


def mlp(hidden):
    torch.manual_seed(0)                                        # same starting knobs every run: same numbers for everyone
    layers, n = [nn.Flatten()], 288
    for h in hidden:
        layers += [nn.Linear(n, h), nn.ReLU()]; n = h
    return nn.Sequential(*layers, nn.Linear(n, 8))


def fit(model, epochs, n_train=None, wd=1e-4, lr=3e-3):
    torch.manual_seed(0)
    xt, yt = X["train"][:n_train], Y["train"][:n_train]
    loader = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, curve = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:                                   # ── 1 TRAIN: the four steps, batch by batch
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():                                   # ── 2 VALIDATE: no learning here
            tr = F.mse_loss(model(xt), yt).item()
            va = F.mse_loss(model(X["val"]), Y["val"]).item()
        curve.append((tr, va))
        if va < best:                                           #    keep the best epoch
            best, best_state, best_epoch = va, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return np.array(curve), best_epoch


net = mlp((64, 64))
curve, be = fit(net, 40)
print(f"1-2 TRAIN+VALIDATE  MLP: best epoch {be}; train loss {curve[be, 0]:.3f}, validation loss {curve[be, 1]:.3f}")

big = mlp((512, 512))
curve_big, be_big = fit(big, 80, n_train=200, wd=0.0, lr=1e-3)
print(f"    OVERFITTING     415k knobs on 200 examples: train loss {curve_big[0, 0]:.3f} → {curve_big[-1, 0]:.4f}, "
      f"validation {curve_big[be_big, 1]:.3f} at epoch {be_big} → {curve_big[-1, 1]:.3f} at the end  (it memorised)")

# ── 3 TEST: once. Persistence, linear, MLP on storm 2
lin = Ridge(alpha=1.0).fit(X["train"].reshape(len(X["train"]), -1).numpy(), Y["train"].numpy())
with torch.no_grad():
    tests = {"persistence": X["test"][:, -1:, 0].repeat(1, 8).numpy(),
             "linear": lin.predict(X["test"].reshape(len(X["test"]), -1).numpy()),
             "MLP": net(X["test"]).numpy()}
print("3 TEST (once)       error 2 h ahead on storm 2: " + "   ".join(f"{k} {mae2h(v, 'test'):.0f}" for k, v in tests.items()) + "  m³/d")

# ── 4 SAVE everything the model needs; then use it like the plant would
torch.save({"state_dict": net.state_dict(), "mean": MEAN, "std": STD, "hidden": (64, 64),
            "channels": ["Q", "S_NH", "S_S"]}, DATA / "lesson5_mlp.pt")
blob = torch.load(DATA / "lesson5_mlp.pt", weights_only=False)
model = mlp(blob["hidden"]); model.load_state_dict(blob["state_dict"]); model.eval()
for i, when in ((len(d["X_test"]) - 1, "calm, end of test"), (100, "during storm 2")):
    raw = np.exp(d["X_test"][i] * STD + MEAN)                           # 24 h of raw sensor values
    z = (np.log(raw) - blob["mean"]) / blob["std"]                      # the SAVED scaler, not a new one
    with torch.no_grad():
        out = flow(model(torch.from_numpy(z[None].astype(np.float32)))[0].numpy())
    print(f"4 SAVE + RELOAD     {when:18s} forecast in 2 h: {out[-1]:6.0f} m³/d   what happened: {flow(d['y_test'][i])[-1]:6.0f} m³/d")
print("                    (no storm in the training data: the storm is where every model struggles)")

# Try this: fit(big, 80, n_train=None)  (all ~1 900 examples). Does the validation loss still rise?
