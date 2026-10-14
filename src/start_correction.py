"""Start-of-run correction for the EWMA regression.

At the beginning of a recording the regression has no thermal history, so its error is a
bias that fades as the EWMAs fill. Given an initial temperature estimate T0 (in a vehicle: the
value the controller stored at shutdown), we remove that bias and let the correction decay:

    T_hat(t) = f(x)(t) + (T0 - f(x)(0)) * exp(-t / tau)

tau is chosen per target from out-of-fold predictions on the training profiles (grouped CV);
the test profiles are only scored at the end.

    python -m src.start_correction
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MEASURABLE, ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES, load, make_features, split
from .metrics import report
from .select_cv import SPAN_SETS

TAUS_S = [30, 60, 120, 300, 600, 900, 1200, 1800, 2700, 3600]
HOT_START = [53, 47, 48]  # training profiles whose magnet starts ≥10 K above coolant


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def correct(pred: np.ndarray, pids: np.ndarray, t0: np.ndarray, taus: list[float]) -> np.ndarray:
    """pred, t0: (n, 4). t0 rows hold the initial-temperature estimate of each row's profile."""
    out = pred.copy()
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        t = np.arange(len(idx)) / SAMPLE_RATE_HZ
        bias0 = t0[idx[0]] - pred[idx[0]]
        for j, tau in enumerate(taus):
            out[idx, j] += bias0[j] * np.exp(-t / tau)
    return out


def initial_temps(df: pd.DataFrame, mode: str) -> np.ndarray:
    first = df.groupby("profile_id", sort=False).transform("first")
    if mode == "known":
        return first[TARGETS].to_numpy()
    return np.repeat(first[["coolant"]].to_numpy(), len(TARGETS), axis=1)  # cold assumption


def main(rest_start: bool = True, score_test: bool = True):
    df = load()
    X, _ = make_features(df, MEASURABLE, spans=SPAN_SETS["short (7–53 min)"], with_std=True, rest_start=rest_start)
    X = X.to_numpy(np.float32)
    tr, te = split(df)
    dtr = df[tr].reset_index(drop=True)
    Xtr = X[tr.to_numpy()]
    pids_tr = dtr["profile_id"].to_numpy()

    # Out-of-fold predictions on training profiles (whole profiles held out per fold)
    oof = np.zeros((len(dtr), len(TARGETS)))
    for a, b in GroupKFold(5).split(Xtr, groups=pids_tr):
        for j, t in enumerate(TARGETS):
            oof[b, j] = ridge().fit(Xtr[a], dtr[t].iloc[a]).predict(Xtr[b])

    Y = dtr[TARGETS].to_numpy()
    t0_known = initial_temps(dtr, "known")
    best_tau = []
    for j, t in enumerate(TARGETS):
        scores = {tau: np.mean((correct(oof[:, [j]], pids_tr, t0_known[:, [j]], [tau])[:, 0] - Y[:, j]) ** 2) for tau in TAUS_S}
        best_tau.append(min(scores, key=scores.get))
        print(f"{t:15s} OOF MSE uncorrected {np.mean((oof[:, j]-Y[:, j])**2):.2f} → best tau {best_tau[-1]}s: {scores[best_tau[-1]]:.2f}")

    rows = {}
    hot = np.isin(pids_tr, HOT_START)
    for mode in ["none", "cold", "known"]:
        P = oof if mode == "none" else correct(oof, pids_tr, initial_temps(dtr, mode), best_tau)
        rows[f"cv_all_{mode}"] = report(dtr, P, mode)
        rows[f"cv_hot_{mode}"] = report(dtr[hot], P[hot], mode)

    if not score_test:
        cv = {k: {m: round(v[m], 2) for m in ["mse_avg", "mse_pm", "linf"]} for k, v in rows.items()}
        print(json.dumps(cv, indent=2))
        return cv

    # Final: fit on all training profiles, score test once per initialisation mode
    dte = df[te].reset_index(drop=True)
    Pte = np.column_stack([ridge().fit(Xtr, dtr[t]).predict(X[te.to_numpy()]) for t in TARGETS])
    pids_te = dte["profile_id"].to_numpy()
    for mode in ["none", "cold", "known"]:
        P = Pte if mode == "none" else correct(Pte, pids_te, initial_temps(dte, mode), best_tau)
        rows[f"test_{mode}"] = report(dte, P, mode)
        if mode == "known":
            out = dte[["profile_id"] + TARGETS].copy()
            for j, t in enumerate(TARGETS):
                out[f"{t}_hat"] = P[:, j]
                out[f"{t}_hat_raw"] = Pte[:, j]
            out.to_parquet(ROOT / "reports" / "test_predictions_corrected.parquet")

    summary = {k: {m: round(v[m], 2) for m in ["mse_avg", "mse_pm", "linf"]} for k, v in rows.items()}
    (ROOT / "reports" / "start_correction.json").write_text(json.dumps({"tau_s": dict(zip(TARGETS, best_tau)), "results": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(rest_start="--first-value-start" not in sys.argv, score_test="--cv-only" not in sys.argv)
