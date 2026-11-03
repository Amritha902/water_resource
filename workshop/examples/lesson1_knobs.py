"""Lesson 1, hands-on: a model is a function with knobs, and training turns the knobs.

Run it:   python workshop/examples/lesson1_knobs.py

Part A does it by hand with numpy, so every step is visible.
Part B does the same with PyTorch, in the same four lines that every training script
in this repository uses (notebook 2, src/wwtp/forecast/train.py, src/wwtp/rl/ddpg.py ...).
"""

import numpy as np
import torch

# ── the data: 20 examples. x = inflow (scaled), y = ammonium load (scaled). FIXED, never changed.
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 20)
y = 0.7 * x + 0.15 + 0.05 * rng.normal(size=20)     # the "true" rule, hidden from the model

# ═══════════════════════════════ Part A: by hand
w, b = 0.0, 0.0          # THE KNOBS (parameters). Training changes only these two numbers.
lr = 0.5                 # LEARNING RATE: how far we turn the knobs each step

print("Part A: by hand")
for step in range(201):
    pred = w * x + b                          # 1 GUESS: run the model on the data (forward)
    loss = np.mean((pred - y) ** 2)           # 2 HOW WRONG: one number, the mean squared error
    if step in (0, 1, 2, 5, 10, 50, 200):
        print(f"  step {step:3d}   w = {w:6.3f}   b = {b:6.3f}   loss = {loss:.5f}")
    # 3 WHICH WAY: the gradient = how the loss changes when a knob is nudged.
    #   loss = mean((w·x + b − y)²); by the chain rule  d loss/dw = mean(2·(pred − y)·x),  d loss/db = mean(2·(pred − y)).
    #   PyTorch computes this for you with loss.backward() (Part B).
    grad_w = np.mean(2 * (pred - y) * x)      # 3 WHICH WAY: the gradient, for each knob
    grad_b = np.mean(2 * (pred - y))
    w = w - lr * grad_w                       # 4 TURN THE KNOBS a little, downhill
    b = b - lr * grad_b
print("  the hidden rule was w = 0.7, b = 0.15\n")

# ═══════════════════════════════ Part B: the same with PyTorch (how the repository writes it)
xt = torch.tensor(x, dtype=torch.float32)[:, None]
yt = torch.tensor(y, dtype=torch.float32)[:, None]
torch.manual_seed(0)
model = torch.nn.Linear(1, 1)                 # THE KNOBS: model.weight (= w) and model.bias (= b)
opt = torch.optim.SGD(model.parameters(), lr=0.5)

print("Part B: PyTorch")
for step in range(201):
    pred = model(xt)                                  # 1 guess          (notebook 2, cell 9, line 19)
    loss = torch.nn.functional.mse_loss(pred, yt)     # 2 how wrong      (notebook 2, cell 9, line 20)
    if step in (0, 50, 200):
        print(f"  step {step:3d}   w = {model.weight.item():6.3f}   b = {model.bias.item():6.3f}   loss = {loss.item():.5f}")
    opt.zero_grad()
    loss.backward()                                   # 3 which way      (notebook 2, cell 9, line 22)
    opt.step()                                        # 4 turn the knobs (notebook 2, cell 9, line 24)

# Try this:  lr = 0.05 (slow)   lr = 1.2 (too big: watch the loss explode)   then 4.0
