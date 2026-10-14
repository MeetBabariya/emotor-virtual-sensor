"""Recurrent thermal-state model (GRU) with an explicit initial temperature state.

Feature models (EWMA + regression) implicitly assume every recording starts with the motor in
equilibrium with the coolant. Real vehicles restart hot. A state-space model can instead be
*initialised* with a temperature estimate — in a vehicle, the value the controller stored at
shutdown. We evaluate two initialisations:

  * "known start"  – true initial temperatures (as in the published TNN benchmark)
  * "cold assumption" – all four temperatures initialised to the coolant reading

    python -m src.seq_model
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import MEASURABLE, ROOT, TARGETS, TEST_PROFILES, add_physics_features, load

torch.manual_seed(0)
np.random.seed(0)

INPUTS = MEASURABLE + ["i_s", "u_s", "s_el", "i_s_sq", "s_x_w"]
# Validation profiles (taken from the training pool, never the test set). Deliberately includes
# 53 and 48, which start 32 K and 10 K above coolant, so early stopping also sees hot restarts.
VAL_PROFILES = [53, 48, 58, 6, 20, 70]
CHUNK = 1024          # truncated back-prop length (≈ 8.5 min at 2 Hz)
HIDDEN = 48
EPOCHS = 60
PATIENCE = 10
DEVICE = torch.device("cpu")   # small recurrent model: CPU is faster than MPS here


class ThermalGRU(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = HIDDEN):
        super().__init__()
        self.init = nn.Sequential(nn.Linear(n_out + 2, hidden), nn.Tanh())  # (T0, coolant0, ambient0) → h0
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def initial_state(self, t0, cool0, amb0):
        return self.init(torch.cat([t0, cool0, amb0], dim=-1)).unsqueeze(0)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return self.head(out), h


class Scaler:
    def __init__(self, df: pd.DataFrame, cols: list[str]):
        self.mu = df[cols].mean().to_numpy(np.float32)
        self.sd = df[cols].std().to_numpy(np.float32) + 1e-6

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mu) / self.sd


def to_batch(df: pd.DataFrame, pids: list[int], xs: Scaler, ys: Scaler):
    groups = [df[df.profile_id == p] for p in pids]
    T = max(len(g) for g in groups)
    X = np.zeros((len(pids), T, len(INPUTS)), np.float32)
    Y = np.zeros((len(pids), T, len(TARGETS)), np.float32)
    M = np.zeros((len(pids), T), np.float32)
    for i, g in enumerate(groups):
        n = len(g)
        X[i, :n] = xs(g[INPUTS].to_numpy(np.float32))
        Y[i, :n] = ys(g[TARGETS].to_numpy(np.float32))
        M[i, :n] = 1
    first = pd.concat([g.iloc[[0]] for g in groups])
    return (torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M), first)


def init_tensors(first: pd.DataFrame, ys: Scaler, mode: str):
    if mode == "known":
        t0 = first[TARGETS].to_numpy(np.float32)
    else:  # cold assumption: every component starts at coolant temperature
        t0 = np.repeat(first[["coolant"]].to_numpy(np.float32), len(TARGETS), axis=1)
    t0 = ys(t0)
    c = (first[["coolant"]].to_numpy(np.float32) - ys.mu.mean()) / ys.sd.mean()
    a = (first[["ambient"]].to_numpy(np.float32) - ys.mu.mean()) / ys.sd.mean()
    return torch.from_numpy(t0), torch.from_numpy(c), torch.from_numpy(a)


@torch.no_grad()
def predict(model, batch, ys: Scaler, mode: str) -> np.ndarray:
    X, _, _, first = batch
    h = model.initial_state(*init_tensors(first, ys, mode))
    outs = []
    for s in range(0, X.shape[1], CHUNK * 8):
        o, h = model(X[:, s:s + CHUNK * 8], h)
        outs.append(o)
    return torch.cat(outs, 1).numpy() * ys.sd + ys.mu


def masked_metrics(P: np.ndarray, batch, ys: Scaler) -> dict:
    _, Y, M, _ = batch
    Yt = Y.numpy() * ys.sd + ys.mu
    m = M.numpy().astype(bool)
    e = (P - Yt)[m]
    return {"mse_avg": float((e ** 2).mean()), "mse_pm": float((e[:, 0] ** 2).mean()), "linf": float(np.abs(e).max())}


def main():
    df = add_physics_features(load())
    all_p = sorted(df.profile_id.unique())
    train_p = [p for p in all_p if p not in TEST_PROFILES + VAL_PROFILES]
    xs = Scaler(df[df.profile_id.isin(train_p)], INPUTS)
    ys = Scaler(df[df.profile_id.isin(train_p)], TARGETS)
    tr, va, te = (to_batch(df, p, xs, ys) for p in (train_p, VAL_PROFILES, TEST_PROFILES))

    model = ThermalGRU(len(INPUTS), len(TARGETS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    X, Y, M, first = tr
    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        # Train half the time from the true initial state, half from the cold assumption,
        # so the model learns to correct a wrong starting estimate from the signals.
        mode = "known" if ep % 2 == 0 else "cold"
        h = model.initial_state(*init_tensors(first, ys, mode))
        total = 0.0
        for s in range(0, X.shape[1], CHUNK):
            out, h = model(X[:, s:s + CHUNK], h)
            m = M[:, s:s + CHUNK].unsqueeze(-1)
            loss = (((out - Y[:, s:s + CHUNK]) ** 2) * m).sum() / (m.sum() * len(TARGETS) + 1e-8)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            h = h.detach()
            total += loss.item()
        model.eval()
        v_known = masked_metrics(predict(model, va, ys, "known"), va, ys)
        v_cold = masked_metrics(predict(model, va, ys, "cold"), va, ys)
        score = (v_known["mse_avg"] + v_cold["mse_avg"]) / 2
        sched.step(score)
        print(f"ep {ep:02d} train {total:.3f}  val known {v_known['mse_avg']:.2f}  val cold {v_cold['mse_avg']:.2f} K²  ({time.time()-t0:.0f}s)", flush=True)
        if score < best:
            best, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "x_mu": xs.mu, "x_sd": xs.sd, "y_mu": ys.mu, "y_sd": ys.sd,
                "inputs": INPUTS, "targets": TARGETS, "hidden": HIDDEN}, ROOT / "models" / "thermal_gru.pt")
    res = {}
    for name, batch in [("val", va), ("test", te)]:
        for mode in ["known", "cold"]:
            res[f"{name}_{mode}"] = masked_metrics(predict(model, batch, ys, mode), batch, ys)
    (ROOT / "reports" / "gru_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
