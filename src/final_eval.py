"""Fit the CV-selected configuration on all training profiles and score the held-out
test profiles exactly once. Saves predictions, metrics, the model, and figures.

    python -m src.final_eval
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MEASURABLE, ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES, load, make_features, split
from .metrics import PUBLISHED, report
from .select_cv import SPAN_SETS

FIG = ROOT / "reports" / "figures"


def build(kind: str, alpha: float = 1.0):
    if kind == "ridge":
        return lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def main():
    cv = json.loads((ROOT / "reports" / "cv_selection.json").read_text())
    best = min(cv, key=lambda r: r["cv_mse"])
    print("CV-selected config:", best)

    df = load()
    tr, te = split(df)
    X, cols = make_features(df, MEASURABLE, spans=SPAN_SETS[best["spans"]], with_std=best["std"])
    X_tr, X_te = X[tr].to_numpy(np.float32), X[te].to_numpy(np.float32)
    models = {t: build(best["model"], best.get("alpha", 1.0))().fit(X_tr, df.loc[tr, t]) for t in TARGETS}
    P = np.column_stack([models[t].predict(X_te) for t in TARGETS])

    row = report(df.loc[te], P, f"{best['model']} ({best['spans']}, std={best['std']})")
    (ROOT / "reports" / "test_metrics.json").write_text(json.dumps(row, indent=2))
    joblib.dump({"models": models, "features": cols, "config": best}, ROOT / "models" / "virtual_sensor.joblib")

    pred = df.loc[te, ["profile_id"] + TARGETS].copy()
    for i, t in enumerate(TARGETS):
        pred[f"{t}_hat"] = P[:, i]
    pred.to_parquet(ROOT / "reports" / "test_predictions.parquet")

    print("\nTest (profiles 60/62/74), scored once:")
    print({k: round(v, 2) for k, v in row.items() if k != "model"})
    print("\nPublished:\n", PUBLISHED.to_string(index=False))

    # Figure: magnet temperature, truth vs estimate, per test profile
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TEST_PROFILES), 1, figsize=(10, 7), sharey=False)
    for ax, pid in zip(axes, TEST_PROFILES):
        d = pred[pred.profile_id == pid]
        t_min = np.arange(len(d)) / SAMPLE_RATE_HZ / 60
        ax.plot(t_min, d["pm"], lw=1.4, label="measured (lab only)", color="#111")
        ax.plot(t_min, d["pm_hat"], lw=1.1, label="virtual sensor", color="#d9480f")
        ax.set_ylabel(f"#{pid} · °C")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("minutes")
    fig.suptitle("Permanent-magnet temperature on unseen test profiles")
    fig.tight_layout()
    fig.savefig(FIG / "pm_test_profiles.png", dpi=160)
    print("saved", FIG / "pm_test_profiles.png")


if __name__ == "__main__":
    main()
