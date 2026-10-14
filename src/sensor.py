"""The deployable virtual sensor: mean of (EWMA ridge + start correction) and the TNN.

    python -m src.sensor        # fit + save models/virtual_sensor.joblib, cache predictions

`VirtualSensor.estimate(profile_df, start=...)` runs on any single drive cycle, including
modified inputs for what-if analysis.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch

from . import tnn as T
from .data import MEASURABLE, ROOT, TARGETS, TEST_PROFILES, add_physics_features, load, make_features
from .seq_model import VAL_PROFILES
from .select_cv import SPAN_SETS
from .start_correction import correct, ridge

SPANS = SPAN_SETS["short (7–53 min)"]
ARTIFACT = ROOT / "models" / "virtual_sensor.joblib"
CACHE = ROOT / "reports" / "virtual_sensor_predictions.parquet"


class VirtualSensor:
    def __init__(self, ridge_models: dict, taus: list[float], tnn_ckpt: dict):
        self.ridge_models = ridge_models
        self.taus = taus
        self.tnn_ckpt = tnn_ckpt
        c = tnn_ckpt
        cell = T.TNNCell(len(c["inputs"]), len(c["targets"]), [c["inputs"].index(b) for b in c["boundary"]])
        self.tnn = T.TNN(cell)
        self.tnn.load_state_dict(c["state"])
        self.tnn.eval()

    @classmethod
    def load(cls) -> "VirtualSensor":
        path = ARTIFACT if ARTIFACT.exists() else ROOT / "artifacts" / "virtual_sensor.joblib"
        a = joblib.load(path)
        return cls(a["ridge_models"], a["taus"], a["tnn_ckpt"])

    def estimate(self, cycle: pd.DataFrame, start: str = "stored", t0: list[float] | None = None) -> pd.DataFrame:
        """cycle: rows of ONE drive cycle with the measurable columns (and targets if known).

        start="stored": initial temperatures come from the cycle's first row (in a vehicle, the
        estimate saved at shutdown). start="cold": every component starts at coolant temperature.
        t0: optional explicit initial temperatures for the four components (overrides start).
        """
        cycle = cycle.reset_index(drop=True).copy()
        cycle["profile_id"] = cycle.get("profile_id", 0)
        if t0 is not None:  # explicit initial temperatures (e.g. the value stored at the last shutdown)
            for t, v in zip(TARGETS, t0):
                cycle.loc[0, t] = v
            start = "stored"
        mode = "known" if start == "stored" else "cold"
        if mode == "known" and not set(TARGETS).issubset(cycle.columns):
            raise ValueError("start='stored' needs initial temperatures; use start='cold'")
        X, _ = make_features(cycle, MEASURABLE, spans=SPANS, with_std=True)
        P_r = np.column_stack([self.ridge_models[t].predict(X) for t in TARGETS])
        if mode == "known":
            t0 = np.repeat(cycle[TARGETS].iloc[[0]].to_numpy(), len(cycle), axis=0)
        else:
            t0 = np.repeat(cycle[["coolant"]].iloc[[0]].to_numpy(), len(cycle) * len(TARGETS)).reshape(len(cycle), -1)
        P_r = correct(P_r, cycle["profile_id"].to_numpy(), t0, self.taus)
        d = add_physics_features(cycle)
        if not set(TARGETS).issubset(d.columns):
            for t in TARGETS:
                d[t] = d["coolant"]
        batch = T.prepare(d, [d.profile_id.iloc[0]], self.tnn_ckpt["x_scale"])
        P_t = T.predict(self.tnn, batch, mode)[: len(cycle), 0]
        P = (P_r + P_t) / 2
        return pd.DataFrame(P, columns=TARGETS)


def build():
    raw = load()
    train_p = [p for p in sorted(raw.profile_id.unique()) if p not in TEST_PROFILES + VAL_PROFILES]
    X, _ = make_features(raw, MEASURABLE, spans=SPANS, with_std=True)
    is_tr = raw.profile_id.isin(train_p).to_numpy()
    ridge_models = {t: ridge().fit(X[is_tr], raw.loc[is_tr, t]) for t in TARGETS}
    taus = list(json.loads((ROOT / "reports" / "start_correction.json").read_text())["tau_s"].values())
    choice = json.loads((ROOT / "reports" / "comparison.json").read_text())
    tnn_ckpt = torch.load(ROOT / "models" / choice["tnn_checkpoint"], weights_only=False)
    joblib.dump({"ridge_models": ridge_models, "taus": taus, "tnn_ckpt": tnn_ckpt,
                 "train_profiles": train_p, "val_profiles": VAL_PROFILES, "test_profiles": TEST_PROFILES}, ARTIFACT)

    # Cache per-sample estimates for every profile (stored-start), used by the assistant's tools.
    vs = VirtualSensor.load()
    parts = []
    for pid, g in raw.groupby("profile_id", sort=True):
        est = vs.estimate(g, start="stored")
        part = g[["profile_id"] + TARGETS + ["coolant", "ambient", "motor_speed", "i_d", "i_q"]].reset_index(drop=True)
        for t in TARGETS:
            part[f"{t}_est"] = est[t].to_numpy()
        parts.append(part)
    pd.concat(parts, ignore_index=True).to_parquet(CACHE)
    print("saved", ARTIFACT, "and", CACHE)


if __name__ == "__main__":
    build()
