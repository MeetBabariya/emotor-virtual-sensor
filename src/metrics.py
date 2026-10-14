"""Metrics in the same units as the published benchmark: MSE in K² and worst-case |error| in K."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGETS

# Kirchgässner et al., "Thermal neural networks", EAAI 2023, Table III (generalization set 60/62/74).
# MSE averaged over the four targets; l_inf = worst-case absolute error.
PUBLISHED = pd.DataFrame(
    [
        ("MLP", 5.58, 14.29),
        ("OLS", 4.47, 9.85),
        ("CNN", 4.43, 15.54),
        ("LPTN (physics)", 3.64, 7.37),
        ("TNN (small)", 3.18, 5.84),
        ("TNN (best)", 2.87, 6.02),
    ],
    columns=["model", "mse_avg", "linf"],
)


def report(y_true: pd.DataFrame, y_pred: np.ndarray, name: str) -> dict:
    err = y_pred - y_true[TARGETS].to_numpy()
    row = {"model": name}
    for i, t in enumerate(TARGETS):
        row[f"mse_{t}"] = float(np.mean(err[:, i] ** 2))
    row["mse_avg"] = float(np.mean(err ** 2))
    row["linf"] = float(np.max(np.abs(err)))
    row["linf_pm"] = float(np.max(np.abs(err[:, 0])))
    return row
