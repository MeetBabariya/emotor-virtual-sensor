"""Head-to-head comparison of the corrected EWMA regression and the TNN on identical splits.

Both models are trained on the same profiles (everything except validation + test). Choices
(which TNN seed, whether to ensemble) are made on the VALIDATION profiles; test is reported
for the chosen configuration only.

    python -m src.compare
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import MEASURABLE, ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES, add_physics_features, load, make_features
from .metrics import PUBLISHED
from .seq_model import VAL_PROFILES
from .select_cv import SPAN_SETS
from .start_correction import correct, initial_temps, ridge
from . import tnn as T

FIG = ROOT / "reports" / "figures"


def score(P, Y) -> dict:
    e = P - Y
    return {"mse_avg": round(float((e ** 2).mean()), 2), "mse_pm": round(float((e[:, 0] ** 2).mean()), 2),
            "linf": round(float(np.abs(e).max()), 2)}


def tnn_predict(ckpt: Path, df: pd.DataFrame, pids: list[int], mode: str) -> np.ndarray:
    """Predictions in the row order of df[df.profile_id.isin(pids)] (grouped by profile, original order)."""
    c = torch.load(ckpt, weights_only=False)
    cell = T.TNNCell(len(c["inputs"]), len(c["targets"]), [c["inputs"].index(b) for b in c["boundary"]])
    model = T.TNN(cell)
    model.load_state_dict(c["state"])
    model.eval()
    out = []
    for p in pids:
        batch = T.prepare(df, [p], c["x_scale"])
        out.append(T.predict(model, batch, mode)[: int(batch[2].sum()), 0])
    return np.concatenate(out)


def main():
    raw = load()
    df = add_physics_features(raw)
    order = lambda pids: pd.concat([df[df.profile_id == p] for p in pids])
    train_p = [p for p in sorted(df.profile_id.unique()) if p not in TEST_PROFILES + VAL_PROFILES]

    X, _ = make_features(raw, MEASURABLE, spans=SPAN_SETS["short (7–53 min)"], with_std=True)
    taus = list(json.loads((ROOT / "reports" / "start_correction.json").read_text())["tau_s"].values())
    is_tr = raw.profile_id.isin(train_p).to_numpy()
    models = {t: ridge().fit(X[is_tr], raw.loc[is_tr, t]) for t in TARGETS}

    seeds = sorted(ROOT.glob("models/tnn_seed*.pt"))
    best_seed = min(seeds, key=lambda p: torch.load(p, weights_only=False)["val_score"])
    print("TNN seeds:", [s.name for s in seeds], "→ chosen by val:", best_seed.name)

    out, preds = {}, {}
    for split_name, pids in [("val", VAL_PROFILES), ("test", TEST_PROFILES)]:
        d = order(pids)
        idx = np.concatenate([np.flatnonzero(raw.profile_id.to_numpy() == p) for p in pids])
        Y = d[TARGETS].to_numpy()
        Pr = np.column_stack([models[t].predict(X.iloc[idx]) for t in TARGETS])
        for mode in ["known", "cold"]:
            t0 = initial_temps(d.reset_index(drop=True), mode)
            Prc = correct(Pr, d.profile_id.to_numpy(), t0, taus)
            Pt = tnn_predict(best_seed, df, pids, mode)
            Pe = (Prc + Pt) / 2
            for name, P in [("ridge+corr", Prc), ("tnn", Pt), ("ensemble", Pe)]:
                out[f"{split_name}_{mode}_{name}"] = score(P, Y)
                preds[(split_name, mode, name)] = P
        preds[(split_name, "y")] = Y
        preds[(split_name, "pid")] = d.profile_id.to_numpy()

    # choose the deployable configuration on validation (known start = vehicle stores its estimate)
    cands = ["ridge+corr", "tnn", "ensemble"]
    chosen = min(cands, key=lambda n: out[f"val_known_{n}"]["mse_avg"])
    res = {"chosen_on_val": chosen, "tnn_checkpoint": best_seed.name, "scores": out}
    (ROOT / "reports" / "comparison.json").write_text(json.dumps(res, indent=2))
    tbl = pd.DataFrame(out).T
    print(tbl.to_string())
    print("\nChosen on validation:", chosen)
    print("Published:\n", PUBLISHED.to_string(index=False))

    # Figure: hot restart (val profile 53) — magnet temperature, cold assumption vs known start
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    pid = preds[("val", "pid")]
    s = pid == 53
    tmin = np.arange(s.sum()) / SAMPLE_RATE_HZ / 60
    for ax, mode, title in [(axes[0], "cold", "Cold assumption (start = coolant)"), (axes[1], "known", "Stored estimate at restart")]:
        ax.plot(tmin, preds[("val", "y")][s, 0], color="#111", lw=1.4, label="measured")
        ax.plot(tmin, preds[("val", mode, "ridge+corr")][s, 0], color="#d9480f", lw=1.1, label="EWMA regression + correction")
        ax.plot(tmin, preds[("val", mode, "tnn")][s, 0], color="#3b5bdb", lw=1.1, label="thermal neural network")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 60)
        ax.grid(alpha=0.2)
        ax.set_xlabel("minutes after restart")
    axes[0].set_ylabel("magnet °C")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Hot restart (profile 53: magnet starts 32 K above coolant)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "hot_restart.png", dpi=160)
    print("saved", FIG / "hot_restart.png")


if __name__ == "__main__":
    main()
