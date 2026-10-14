"""Thermal Neural Network (TNN): a lumped-parameter thermal network whose losses and
conductances are small neural networks, integrated with explicit Euler at 2 Hz.

Architecture after Kirchgässner, Wallscheid & Böcker, "Thermal neural networks: Lumped-parameter
thermal modeling with state-space machine learning", Engineering Applications of AI (2023),
reference code: github.com/wkirgsn/thermal-nn (MIT). Differences here:
  * inputs exclude torque (not measurable in a production vehicle)
  * trained from both the true initial state and a "cold assumption" (all nodes = coolant)
  * early stopping on validation profiles that include hot restarts

State per step (4 nodes: magnet, stator yoke/tooth/winding; 2 boundary nodes: coolant, ambient):
    T[k+1] = T[k] + dt * exp(c) * ( sum_j G_ij (T_j - T_i) + P_i )
with G = |sigmoid(MLP)| (conductances) and P = |MLP| (power losses), both fed by [inputs, T[k]].

    python -m src.tnn [--seed N]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .data import MEASURABLE, ROOT, TARGETS, TEST_PROFILES, add_physics_features, load
from .seq_model import VAL_PROFILES

INPUTS = MEASURABLE + ["i_s", "u_s", "s_el"]
BOUNDARY = ["coolant", "ambient"]
TEMP_SCALE = 200.0      # °C → model units, as in the reference implementation
DT = 0.5                # s
TBPTT = 512
EPOCHS = 80
PATIENCE = 12
torch.set_num_threads(4)


class TNNCell(nn.Module):
    def __init__(self, n_in: int, n_out: int, boundary_idx: List[int], hidden: int = 16):
        super().__init__()
        n_temps = n_out + len(boundary_idx)
        n_conds = n_temps * (n_temps - 1) // 2
        self.caps = nn.Parameter(torch.empty(n_out).normal_(-9.2, 0.5))
        self.conductance = nn.Sequential(nn.Linear(n_in + n_out, n_conds), nn.Sigmoid())
        self.ploss = nn.Sequential(nn.Linear(n_in + n_out, hidden), nn.Tanh(), nn.Linear(hidden, n_out))
        adj = np.zeros((n_temps, n_temps), dtype=np.int64)
        iu = np.triu_indices(n_temps, 1)
        adj[iu] = np.arange(n_conds)
        adj += adj.T
        self.register_buffer("adj", torch.from_numpy(adj[:n_out]))
        self.boundary_idx = boundary_idx
        self.dt: float = DT

    def forward(self, x: Tensor, T: Tensor) -> Tuple[Tensor, Tensor]:
        temps = torch.cat([T, x[:, self.boundary_idx]], dim=1)
        z = torch.cat([x, T], dim=1)
        G = torch.abs(self.conductance(z))[:, self.adj]                  # (B, n_out, n_temps)
        P = torch.abs(self.ploss(z))
        flow = torch.sum((temps.unsqueeze(1) - T.unsqueeze(-1)) * G, dim=-1)
        T_next = T + self.dt * torch.exp(self.caps) * (flow + P)
        return T, torch.clamp(T_next, -1.0, 5.0)


class TNN(nn.Module):
    def __init__(self, cell: TNNCell):
        super().__init__()
        self.cell = cell

    def forward(self, x: Tensor, T: Tensor) -> Tuple[Tensor, Tensor]:
        outs: List[Tensor] = []
        for xt in x.unbind(0):                     # time-major (T, B, F)
            out, T = self.cell(xt, T)
            outs.append(out)
        return torch.stack(outs), T


def prepare(df: pd.DataFrame, pids: List[int], x_scale: np.ndarray):
    groups = [df[df.profile_id == p] for p in pids]
    L = max(len(g) for g in groups)
    X = np.zeros((L, len(pids), len(INPUTS)), np.float32)
    Y = np.zeros((L, len(pids), len(TARGETS)), np.float32)
    M = np.zeros((L, len(pids)), np.float32)
    for i, g in enumerate(groups):
        n = len(g)
        xi = g[INPUTS].to_numpy(np.float32)
        X[:n, i] = xi / x_scale
        Y[:n, i] = g[TARGETS].to_numpy(np.float32) / TEMP_SCALE
        M[:n, i] = 1.0
    first = pd.concat([g.iloc[[0]] for g in groups])
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M), first


def initial_state(first: pd.DataFrame, mode: str) -> Tensor:
    if mode == "known":
        t0 = first[TARGETS].to_numpy(np.float32)
    else:
        t0 = np.repeat(first[["coolant"]].to_numpy(np.float32), len(TARGETS), axis=1)
    return torch.from_numpy(t0 / TEMP_SCALE)


@torch.no_grad()
def predict(model, batch, mode: str) -> np.ndarray:
    X, _, _, first = batch
    out, _ = model(X, initial_state(first, mode))
    return out.numpy() * TEMP_SCALE


def metrics(P: np.ndarray, batch) -> dict:
    _, Y, M, _ = batch
    m = M.numpy().astype(bool)
    e = (P - Y.numpy() * TEMP_SCALE)[m]
    return {"mse_avg": float((e ** 2).mean()), "mse_pm": float((e[:, 0] ** 2).mean()), "linf": float(np.abs(e).max())}


def main(seed: int = 0, lr: float = 1e-3, epochs: int = EPOCHS, resume: str | None = None, tag: str | None = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = add_physics_features(load())
    train_p = [p for p in sorted(df.profile_id.unique()) if p not in TEST_PROFILES + VAL_PROFILES]
    x_scale = df[df.profile_id.isin(train_p)][INPUTS].abs().max().to_numpy(np.float32)
    # temperatures in the input vector use the same scale as the state
    for c in BOUNDARY:
        x_scale[INPUTS.index(c)] = TEMP_SCALE
    tr, va, te = (prepare(df, p, x_scale) for p in (train_p, VAL_PROFILES, TEST_PROFILES))

    cell = TNNCell(len(INPUTS), len(TARGETS), [INPUTS.index(c) for c in BOUNDARY])
    model = torch.jit.script(TNN(cell))
    if resume:
        model.load_state_dict(torch.load(resume, weights_only=False)["state"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    X, Y, M, first = tr
    best, best_state, bad, log = np.inf, None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        mode = "known" if ep % 2 == 0 else "cold"
        T = initial_state(first, mode)
        for s in range(0, X.shape[0], TBPTT):
            out, T = model(X[s:s + TBPTT], T.detach())
            m = M[s:s + TBPTT].unsqueeze(-1)
            loss = (((out - Y[s:s + TBPTT]) ** 2) * m).sum() / (m.sum() * len(TARGETS) + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vk, vc = metrics(predict(model, va, "known"), va), metrics(predict(model, va, "cold"), va)
        score = (vk["mse_avg"] + vc["mse_avg"]) / 2
        sched.step(score)
        log.append({"epoch": ep, "val_known": vk["mse_avg"], "val_cold": vc["mse_avg"]})
        print(f"seed {seed} ep {ep:02d} val known {vk['mse_avg']:.2f}  cold {vc['mse_avg']:.2f} K²  "
              f"l∞ {vk['linf']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        if score < best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "x_scale": x_scale, "inputs": INPUTS, "targets": TARGETS,
                "boundary": BOUNDARY, "temp_scale": TEMP_SCALE, "seed": seed, "lr": lr, "val_score": best},
               ROOT / "models" / f"tnn_{tag or f'seed{seed}'}.pt")
    res = {"seed": seed, "val_score": best, "params": sum(p.numel() for p in model.parameters())}
    for name, batch in [("val", va), ("test", te)]:
        for mode in ["known", "cold"]:
            res[f"{name}_{mode}"] = metrics(predict(model, batch, mode), batch)
    (ROOT / "reports" / f"tnn_{tag or f'seed{seed}'}.json").write_text(json.dumps({"result": res, "log": log}, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("test")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    main(a.seed, a.lr, a.epochs, a.resume, a.tag)
