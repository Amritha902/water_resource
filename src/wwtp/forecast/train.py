"""Training loop for :class:`~wwtp.forecast.models.InfluentForecaster`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dataset import (CHANNELS, HORIZON, QUANTILES, REGIMES, ForecastCorpus,
                      build_corpus)
from .models import InfluentForecaster, quantile_loss


def _loaders(corpus: ForecastCorpus, batch_size: int, val_frac: float,
             seed: int):
    n = len(corpus)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_frac * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def make(sel, shuffle):
        ds = TensorDataset(torch.from_numpy(corpus.x[sel]),
                           torch.from_numpy(corpus.y[sel]),
                           torch.from_numpy(corpus.regime[sel]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return make(train_idx, True), make(val_idx, False)


def train_forecaster(out_dir: str | Path, n_scenarios: int = 160,
                     epochs: int = 12, batch_size: int = 256,
                     lr: float = 2e-3, seed: int = 0, val_frac: float = 0.15,
                     regime_weight: float = 0.2,
                     device: str = "cpu") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    corpus = build_corpus(n_scenarios=n_scenarios, seed=seed)
    train_loader, val_loader = _loaders(corpus, batch_size, val_frac, seed)

    model = InfluentForecaster().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(len(train_loader), 1))

    history = []
    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for xb, yb, rb in train_loader:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            q, logits, _ = model(xb)
            loss = (quantile_loss(q, yb)
                    + regime_weight * torch.nn.functional.cross_entropy(logits, rb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr += float(loss) * len(xb)
        tr /= max(len(train_loader.dataset), 1)

        model.eval()
        va, acc, n_seen, mae = 0.0, 0.0, 0, 0.0
        with torch.no_grad():
            for xb, yb, rb in val_loader:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                q, logits, _ = model(xb)
                va += float(quantile_loss(q, yb)) * len(xb)
                acc += float((logits.argmax(-1) == rb).sum())
                mae += float((q[..., 1] - yb).abs().mean()) * len(xb)
                n_seen += len(xb)
        history.append({"epoch": epoch, "train_loss": tr,
                        "val_pinball": va / max(n_seen, 1),
                        "val_mae_norm": mae / max(n_seen, 1),
                        "val_regime_acc": acc / max(n_seen, 1)})
        print(f"epoch {epoch:2d}  train {tr:.4f}  val_pinball "
              f"{history[-1]['val_pinball']:.4f}  regime_acc "
              f"{history[-1]['val_regime_acc']:.3f}", flush=True)

    torch.save({"state_dict": model.state_dict(),
                "mean": corpus.mean, "std": corpus.std,
                "channels": CHANNELS, "horizon": HORIZON,
                "quantiles": QUANTILES, "regimes": REGIMES},
               out_dir / "forecaster.pt")
    (out_dir / "forecaster_history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "n_windows": len(corpus)}
