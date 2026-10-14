"""Data loading, splits and feature engineering for the Paderborn PMSM dataset.

Dataset: "Electric Motor Temperature" (Kirchgässner et al., Paderborn University, LEA),
CC-BY-SA-4.0, https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TARGETS = ["pm", "stator_yoke", "stator_tooth", "stator_winding"]
# Signals available in a production vehicle. Torque is NOT reliably measurable in a car,
# so the realistic setup excludes it; the published benchmark included it.
MEASURABLE = ["ambient", "coolant", "u_d", "u_q", "motor_speed", "i_d", "i_q"]
WITH_TORQUE = MEASURABLE + ["torque"]
TEMPERATURE_INPUTS = {"ambient", "coolant"}

# Held-out "generalization set" used by the dataset authors in the Thermal Neural Network
# paper (Kirchgässner, Wallscheid, Böcker, EAAI 2023, Table III). Never used for model selection.
TEST_PROFILES = [60, 62, 74]

# Default exponentially-weighted spans in samples (2 Hz → 7, 11, 28, 53 minutes); chosen by grouped CV.
# Temperature responds to the *history* of losses, which raw instantaneous signals miss.
EWM_SPANS = [840, 1320, 3360, 6360]
SAMPLE_RATE_HZ = 2


DEMO_DATA = ROOT / "artifacts" / "demo_data.parquet"  # compact copy shipped with the demo app


def load() -> pd.DataFrame:
    pq = DATA / "measures_v2.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    if (DATA / "measures_v2.csv").exists():
        df = pd.read_csv(DATA / "measures_v2.csv")
        df.to_parquet(pq)
        return df
    return load_demo()[RAW_COLUMNS]


RAW_COLUMNS = ["u_q", "coolant", "stator_winding", "u_d", "stator_tooth", "motor_speed", "i_d", "i_q",
               "pm", "stator_yoke", "ambient", "torque", "profile_id"]


def load_demo() -> pd.DataFrame:
    """Raw signals + virtual-sensor estimates from the int-quantised demo file."""
    q = pd.read_parquet(DEMO_DATA)
    return pd.DataFrame({c: q[c].astype("int64") if c == "profile_id" else q[c].astype("float64") / 100 for c in q.columns})


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived electrical quantities that relate to copper/iron losses."""
    out = df.copy()
    out["i_s"] = np.hypot(df["i_d"], df["i_q"])            # current magnitude → copper losses ∝ i_s²
    out["u_s"] = np.hypot(df["u_d"], df["u_q"])            # voltage magnitude
    out["s_el"] = out["i_s"] * out["u_s"]                  # apparent electrical power
    out["i_s_sq"] = out["i_s"] ** 2
    out["s_x_w"] = out["s_el"] * df["motor_speed"]          # speed-dependent (iron) loss proxy
    return out


def make_features(
    df: pd.DataFrame,
    inputs: list[str],
    spans: list[int] = EWM_SPANS,
    with_std: bool = True,
    rest_start: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Raw + physics features plus causal exponentially-weighted means (and std) per profile.

    `adjust=False` treats the time before a recording as if the motor had been sitting at its
    first observed operating point. Without it, early EWMAs only "remember" a few seconds of
    history and estimates in the first ~10% of each profile are badly wrong (MSE 11.2 → 3.8 K²
    for a linear model). All windows look backwards only: no future information leaks in.
    """
    df = add_physics_features(df)
    base = inputs + ["i_s", "u_s", "s_el", "i_s_sq", "s_x_w"]
    cols = {c: df[c] for c in base}
    g = df.groupby("profile_id", sort=False)

    def ew(s: pd.Series, span: int, stat: str) -> pd.Series:
        # rest_start: electrical quantities were zero before the recording (motor off at key-on);
        # temperatures (coolant, ambient) keep their first reading. Implemented as one padded sample.
        if rest_start and s.name not in TEMPERATURE_INPUTS:
            padded = pd.concat([pd.Series([0.0]), s.reset_index(drop=True)], ignore_index=True)
            r = getattr(padded.ewm(span=span, adjust=False), stat)().iloc[1:]
            return pd.Series(r.to_numpy(), index=s.index)
        return getattr(s.ewm(span=span, adjust=False), stat)()

    for span in spans:
        mean = g[base].transform(lambda s, sp=span: ew(s, sp, "mean"))
        for c in base:
            cols[f"{c}_ewm{span}"] = mean[c]
        if with_std:
            std = g[base].transform(lambda s, sp=span: ew(s, sp, "std"))
            for c in base:
                cols[f"{c}_ews{span}"] = std[c].fillna(0.0)
    feats = pd.DataFrame(cols, index=df.index)
    return feats, list(feats.columns)


def split(df: pd.DataFrame):
    test = df["profile_id"].isin(TEST_PROFILES)
    return ~test, test
