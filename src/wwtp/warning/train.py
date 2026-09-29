"""Training loop for :class:`~wwtp.warning.models.EarlyWarningNet`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dataset import CHANNELS, HORIZONS, QUANTILES, Normaliser, WarningCorpus
from .models import EarlyWarningNet, pinball


def _loader(c: WarningCorpus, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(c.x), torch.from_numpy(c.level),
                       torch.from_numpy(c.event))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _mask(x: torch.Tensor, drop: tuple[int, ...]) -> torch.Tensor:
    if drop:
        x = x.clone()
        x[..., list(drop)] = 0.0          # normalised mean
    return x


@torch.no_grad()
def predict(model: EarlyWarningNet, x: np.ndarray, batch_size: int = 1024,
            drop: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    lv, pe = [], []
    for i in range(0, len(x), batch_size):
        xb = _mask(torch.from_numpy(x[i:i + batch_size]), drop)
        a, b = model(xb)
        lv.append(a.numpy())
        pe.append(b.numpy())
    return np.concatenate(lv), np.concatenate(pe)


def train_warning(train: WarningCorpus, val: WarningCorpus, norm: Normaliser,
                  out: str | Path, epochs: int = 14, batch_size: int = 256,
                  lr: float = 2e-3, event_weight: float = 1.0, seed: int = 0,
                  drop_channels: tuple[str, ...] = ()) -> dict:
    """Fit the network; ``drop_channels`` trains an ablated variant."""
    torch.manual_seed(seed)
    drop = tuple(CHANNELS.index(c) for c in drop_channels)
    model = EarlyWarningNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tl = _loader(train, batch_size, True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(tl))

    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for xb, lb, eb in tl:
            levels, p = model(_mask(xb, drop))
            loss = (pinball(levels, lb)
                    + event_weight * F.binary_cross_entropy(p, eb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(xb)

        lv, pe = predict(model, val.x, drop=drop)
        v_pin = float(pinball(torch.from_numpy(lv), torch.from_numpy(val.level)))
        v_bce = float(F.binary_cross_entropy(torch.from_numpy(pe),
                                             torch.from_numpy(val.event)))
        v = v_pin + event_weight * v_bce
        history.append({"epoch": epoch, "train": tot / len(train),
                        "val_pinball": v_pin, "val_bce": v_bce})
        print(f"epoch {epoch:2d} train {tot / len(train):.4f} "
              f"val pinball {v_pin:.4f} bce {v_bce:.4f}", flush=True)
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **norm.state(),
                "channels": CHANNELS, "horizons": HORIZONS,
                "quantiles": QUANTILES, "drop_channels": drop_channels}, out)
    out.with_suffix(".json").write_text(json.dumps(history, indent=2))
    return {"model": model, "history": history, "drop": drop}


def load_warning(path: str | Path) -> tuple[EarlyWarningNet, Normaliser, tuple]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = EarlyWarningNet()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    drop = tuple(CHANNELS.index(c) for c in blob.get("drop_channels", ()))
    return model, Normaliser(blob["mean"], blob["std"]), drop
