"""Pack everything the demo app needs into small, committable files under artifacts/.

The raw dataset (300 MB CSV) is too large to ship. Values are stored as int32 hundredths
(0.01 °C / A / V / rpm resolution, finer than the sensors) with zstd compression: 24 MB.

    python -m src.build_demo_data
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from .data import ROOT, TARGETS, load

ART = ROOT / "artifacts"
SCALE = 100


def main():
    ART.mkdir(exist_ok=True)
    est = pd.read_parquet(ROOT / "reports" / "virtual_sensor_predictions.parquet")
    # The prediction cache is grouped by profile_id (stable within each profile); order raw rows the same way.
    raw = load().sort_values("profile_id", kind="stable").reset_index(drop=True)
    assert (raw["profile_id"].to_numpy() == est["profile_id"].to_numpy()).all()
    assert np.allclose(raw[TARGETS].to_numpy(), est[TARGETS].to_numpy()), "row alignment check failed"
    df = raw.copy()
    for t in TARGETS:
        df[f"{t}_est"] = est[f"{t}_est"].to_numpy()
    q = pd.DataFrame({c: df[c].astype("int16") if c == "profile_id" else np.round(df[c] * SCALE).astype("int32") for c in df.columns})
    q.to_parquet(ART / "demo_data.parquet", compression="zstd", compression_level=19, index=False)
    shutil.copy(ROOT / "models" / "virtual_sensor.joblib", ART / "virtual_sensor.joblib")
    for f in ["assistant_eval.json", "comparison.json", "start_correction.json"]:
        shutil.copy(ROOT / "reports" / f, ART / f)
    print("wrote", sorted(p.name for p in ART.iterdir()))


if __name__ == "__main__":
    main()
