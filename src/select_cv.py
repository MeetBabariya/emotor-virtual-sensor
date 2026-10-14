"""Model / feature selection by grouped cross-validation on the TRAINING profiles only.

Whole measurement profiles are held out per fold, so the score reflects generalisation to
unseen drive cycles. The test profiles (60, 62, 74) are not touched here.

    python -m src.select_cv
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MEASURABLE, ROOT, TARGETS, load, make_features, split

SPAN_SETS = {
    "short (7–53 min)": [840, 1320, 3360, 6360],
    "long (11–79 min)": [1320, 3360, 6360, 9480],
    "wide (4–79 min)": [480, 1320, 3360, 9480],
}


def cv_score(model_fn, X: np.ndarray, Y: pd.DataFrame, groups: np.ndarray, n_splits: int = 5) -> dict:
    sq_err, abs_max = [], 0.0
    for tr, va in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        P = np.column_stack([model_fn().fit(X[tr], Y.iloc[tr][t]).predict(X[va]) for t in TARGETS])
        e = P - Y.iloc[va][TARGETS].to_numpy()
        sq_err.append(e ** 2)
        abs_max = max(abs_max, float(np.abs(e).max()))
    se = np.concatenate(sq_err)
    return {"cv_mse": float(se.mean()), "cv_mse_pm": float(se[:, 0].mean()), "cv_linf": abs_max}


def main():
    df = load()
    tr_mask, _ = split(df)
    dtr = df[tr_mask].reset_index(drop=True)
    groups = dtr["profile_id"].to_numpy()
    Y = dtr[TARGETS]
    rows = []

    ridge = lambda a: (lambda: make_pipeline(StandardScaler(), Ridge(alpha=a)))
    for (span_name, spans), with_std in itertools.product(SPAN_SETS.items(), [False, True]):
        X, _ = make_features(dtr, MEASURABLE, spans=spans, with_std=with_std)
        X = X.to_numpy(np.float32)
        for alpha in [0.1, 1.0, 10.0]:
            r = {"model": "ridge", "spans": span_name, "std": with_std, "alpha": alpha, **cv_score(ridge(alpha), X, Y, groups)}
            rows.append(r)
            print(f"ridge  {span_name:18s} std={with_std!s:5} a={alpha:<5} CV MSE {r['cv_mse']:.2f}  pm {r['cv_mse_pm']:.2f}  l∞ {r['cv_linf']:.1f}", flush=True)

    best = min((r for r in rows if r["model"] == "ridge"), key=lambda r: r["cv_mse"])
    X, _ = make_features(dtr, MEASURABLE, spans=SPAN_SETS[best["spans"]], with_std=best["std"])
    X = X.to_numpy(np.float32)
    hgb = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    t0 = time.time()
    r = {"model": "hgb", "spans": best["spans"], "std": best["std"], **cv_score(hgb, X, Y, groups)}
    rows.append(r)
    print(f"hgb    {best['spans']:18s} std={best['std']!s:5}         CV MSE {r['cv_mse']:.2f}  pm {r['cv_mse_pm']:.2f}  l∞ {r['cv_linf']:.1f}  ({time.time()-t0:.0f}s)", flush=True)

    (ROOT / "reports" / "cv_selection.json").write_text(json.dumps(rows, indent=2))
    print("\nBest ridge:", best)


if __name__ == "__main__":
    main()
