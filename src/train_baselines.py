"""Train feature-based baselines and evaluate on the held-out generalization profiles.

    python -m src.train_baselines
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MEASURABLE, ROOT, TARGETS, WITH_TORQUE, load, make_features, split
from .metrics import PUBLISHED, report


def fit_predict(model_fn, X_tr, Y_tr, X_te):
    preds = []
    for t in TARGETS:
        m = model_fn()
        m.fit(X_tr, Y_tr[t])
        preds.append(m.predict(X_te))
    return np.column_stack(preds)


def main():
    df = load()
    tr, te = split(df)
    results = []
    for setup, inputs in [("no torque", MEASURABLE), ("with torque", WITH_TORQUE)]:
        X, cols = make_features(df, inputs)
        X_tr, X_te = X[tr].to_numpy(np.float32), X[te].to_numpy(np.float32)
        Y_tr, Y_te = df.loc[tr, TARGETS], df.loc[te, TARGETS]
        print(f"[{setup}] {len(cols)} features, train {len(X_tr):,} rows, test {len(X_te):,} rows")

        models = {
            "Ridge (OLS + EWMA)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Gradient boosting + EWMA": lambda: HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                l2_regularization=1.0, random_state=0),
        }
        for name, fn in models.items():
            t0 = time.time()
            P = fit_predict(fn, X_tr, Y_tr, X_te)
            row = report(Y_te, P, f"{name} [{setup}]")
            row["fit_s"] = round(time.time() - t0, 1)
            results.append(row)
            print(f"  {name:28s} MSE {row['mse_avg']:.2f} K²  pm {row['mse_pm']:.2f}  l∞ {row['linf']:.2f} K  ({row['fit_s']}s)")

    res = pd.DataFrame(results)
    out = ROOT / "reports" / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nPublished (EAAI 2023, Table III):\n", PUBLISHED.to_string(index=False))
    print("\nOurs:\n", res[["model", "mse_avg", "mse_pm", "linf"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
