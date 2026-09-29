"""Training loop for the neural digital twin.

The twin is trained with scheduled sampling: with probability ``p_free`` a
step consumes the model's own previous prediction instead of the measured
state.  ``p_free`` is annealed from 0 to ``free_max`` so the model is fit for
the free-running rollouts the controller actually performs, not only for
one-step teacher forcing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dataset import (COARSE_STRIDE, D_SCALE, NH_SCALE, S_SCALE, U_SCALE,
                      TwinCorpus, build_corpus)
from .models import DigitalTwin


def _rollout_loss(model: DigitalTwin, s, u, d, s_next, nh, p_free: float,
                  rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, _ = s.shape
    h = model.init_hidden(b, s.device)
    s_in = s[:, 0]
    pred_s, pred_nh = [], []
    for k in range(t):
        s_out, nh_out, h = model.step(s_in, u[:, k], d[:, k], h)
        pred_s.append(s_out)
        pred_nh.append(nh_out)
        if k + 1 < t:
            use_free = rng.random() < p_free
            s_in = s_out if use_free else s[:, k + 1]
    pred_s = torch.stack(pred_s, 1)
    pred_nh = torch.stack(pred_nh, 1)
    return F.smooth_l1_loss(pred_s, s_next), F.smooth_l1_loss(pred_nh, nh)


def train_twin(out_dir: str | Path, n_rollouts: int = 24, epochs: int = 30,
               batch_size: int = 64, lr: float = 2e-3, seed: int = 0,
               val_frac: float = 0.15, free_max: float = 0.6,
               nh_weight: float = 0.3, device: str = "cpu",
               corpus: TwinCorpus | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if corpus is None:
        corpus = build_corpus(n_rollouts=n_rollouts, seed=seed)
    n = len(corpus)
    idx = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def make(sel, shuffle):
        ds = TensorDataset(*[torch.from_numpy(a[sel]) for a in
                             (corpus.s, corpus.u, corpus.d, corpus.s_next,
                              corpus.nh)])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader, val_loader = make(train_idx, True), make(val_idx, False)

    model = DigitalTwin().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    for epoch in range(epochs):
        p_free = free_max * min(1.0, epoch / max(epochs * 0.5, 1))
        model.train()
        tr = 0.0
        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            ls, lnh = _rollout_loss(model, *batch, p_free, rng)
            loss = ls + nh_weight * lnh
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss) * len(batch[0])
        tr /= max(len(train_loader.dataset), 1)
        sched.step()

        model.eval()
        va, rmse_free, seen = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [t.to(device) for t in batch]
                s, u, d, s_next, nh = batch
                ls, lnh = _rollout_loss(model, s, u, d, s_next, nh, 0.0, rng)
                va += float(ls + nh_weight * lnh) * len(s)
                # free-running horizon error in physical units
                h0 = model.init_hidden(len(s), s.device)
                _, _, h0 = model(s[:, :16], u[:, :16], d[:, :16], h0)
                pred, _, _ = model.rollout(s[:, 16], u[:, 16:], d[:, 16:], h0)
                err = (pred - s_next[:, 16:]) * torch.tensor(
                    S_SCALE, dtype=pred.dtype, device=pred.device)
                rmse_free += float(torch.sqrt((err ** 2).mean())) * len(s)
                seen += len(s)
        history.append({"epoch": epoch, "train": tr, "val": va / max(seen, 1),
                        "val_free_rmse": rmse_free / max(seen, 1),
                        "p_free": p_free})
        print(f"epoch {epoch:2d}  train {tr:.5f}  val {history[-1]['val']:.5f} "
              f" free-run RMSE {history[-1]['val_free_rmse']:.4f} mg/L", flush=True)

    torch.save({"state_dict": model.state_dict(),
                "s_scale": S_SCALE, "u_scale": U_SCALE, "d_scale": D_SCALE,
                "nh_scale": NH_SCALE, "coarse_stride": COARSE_STRIDE},
               out_dir / "twin.pt")
    (out_dir / "twin_history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "n_sequences": n}
