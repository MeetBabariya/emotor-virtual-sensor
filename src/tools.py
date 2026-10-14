"""Deterministic analysis tools the engineering assistant can call.

Every number the assistant reports must come from one of these functions. They read the
measurement data and the cached virtual-sensor estimates; `what_if` re-runs the sensor on
modified inputs. All functions return plain JSON-serialisable dicts.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES, load
from .seq_model import VAL_PROFILES

COMPONENTS = {"magnet": "pm", "stator_yoke": "stator_yoke", "stator_tooth": "stator_tooth", "stator_winding": "stator_winding"}
TEST_MSE_K2, TEST_LINF_K = 1.50, 5.73


@lru_cache(maxsize=1)
def _cache() -> pd.DataFrame:
    full = ROOT / "reports" / "virtual_sensor_predictions.parquet"
    if full.exists():
        return pd.read_parquet(full)
    from .data import load_demo  # deployed demo: compact file in artifacts/
    return load_demo()


@lru_cache(maxsize=1)
def _raw() -> pd.DataFrame:
    return load()


def _split(pid: int) -> str:
    if pid in TEST_PROFILES:
        return "test"
    if pid in VAL_PROFILES:
        return "validation"
    return "train"


def _cycle(pid: int) -> pd.DataFrame:
    c = _cache()
    g = c[c.profile_id == pid]
    if g.empty:
        valid = sorted(c.profile_id.unique().tolist())
        raise ValueError(f"Drive cycle {pid} does not exist. Valid ids: {valid[0]}–{valid[-1]} ({len(valid)} cycles, with gaps).")
    return g.reset_index(drop=True)


def _windows(mask: np.ndarray, min_s: float = 5.0) -> list[tuple[int, int]]:
    """Contiguous True runs as (start_idx, end_idx_exclusive), dropping blips shorter than min_s."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(s, e) for s, e in zip(starts, ends) if (e - s) / SAMPLE_RATE_HZ >= min_s]


def list_drive_cycles() -> dict:
    """One row per recorded drive cycle with duration, operating range and peak magnet temperature."""
    c = _cache()
    g = c.groupby("profile_id")
    tab = pd.DataFrame({
        "duration_min": g.size() / SAMPLE_RATE_HZ / 60,
        "max_speed_rpm": g.motor_speed.max(),
        "coolant_start_c": g.coolant.first(),
        "magnet_start_c": g.pm.first(),
        "peak_magnet_measured_c": g.pm.max(),
        "peak_magnet_estimated_c": g.pm_est.max(),
    }).round(1)
    tab["split"] = [_split(p) for p in tab.index]
    return {"n_cycles": int(len(tab)), "total_hours": round(float(tab.duration_min.sum() / 60), 1),
            "cycles": tab.reset_index().to_dict(orient="records")}


def find_hot_restarts(min_delta_k: float = 10.0) -> dict:
    """Drive cycles whose magnet starts at least min_delta_k hotter than the coolant (hot restarts).

    These are the cases where assuming a cold start is badly wrong, so the virtual sensor needs
    the temperature stored at the last shutdown."""
    c = _cache()
    first = c.groupby("profile_id").first()
    delta = (first.pm - first.coolant).round(1)
    hits = delta[delta >= min_delta_k].sort_values(ascending=False)
    return {"min_delta_k": min_delta_k, "n_cycles": int(len(hits)),
            "cycles": [{"profile_id": int(p), "magnet_start_c": round(float(first.pm[p]), 1),
                        "coolant_start_c": round(float(first.coolant[p]), 1), "delta_k": float(d), "split": _split(int(p))}
                       for p, d in hits.items()]}


def get_cycle_summary(profile_id: int) -> dict:
    """Operating conditions and virtual-sensor accuracy for one drive cycle."""
    g = _cycle(profile_id)
    i_s = np.hypot(g.i_d, g.i_q)
    acc = {}
    for name, col in COMPONENTS.items():
        e = g[f"{col}_est"] - g[col]
        acc[name] = {"peak_measured_c": round(float(g[col].max()), 1), "peak_estimated_c": round(float(g[f"{col}_est"].max()), 1),
                     "mse_k2": round(float((e ** 2).mean()), 2), "max_abs_error_k": round(float(e.abs().max()), 2)}
    split = _split(profile_id)
    return {
        "profile_id": profile_id, "split": split,
        "note": None if split == "test" else f"This cycle is in the {split} set, so accuracy here is optimistic compared with unseen data.",
        "duration_min": round(len(g) / SAMPLE_RATE_HZ / 60, 1),
        "speed_rpm": {"mean": round(float(g.motor_speed.mean())), "max": round(float(g.motor_speed.max()))},
        "current_magnitude_a": {"mean": round(float(i_s.mean()), 1), "max": round(float(i_s.max()), 1)},
        "coolant_c": {"start": round(float(g.coolant.iloc[0]), 1), "min": round(float(g.coolant.min()), 1), "max": round(float(g.coolant.max()), 1)},
        "magnet_start_minus_coolant_k": round(float(g.pm.iloc[0] - g.coolant.iloc[0]), 1),
        "components": acc,
    }


def find_overheating(threshold_c: float, component: str = "magnet", source: str = "virtual_sensor") -> dict:
    """Drive cycles and time windows where a component exceeds threshold_c."""
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {list(COMPONENTS)}")
    if source not in ("virtual_sensor", "measured"):
        raise ValueError("source must be 'virtual_sensor' or 'measured'")
    col = COMPONENTS[component] + ("_est" if source == "virtual_sensor" else "")
    hits = []
    for pid, g in _cache().groupby("profile_id"):
        w = _windows(g[col].to_numpy() > threshold_c)
        if not w:
            continue
        tot = sum(e - s for s, e in w) / SAMPLE_RATE_HZ / 60
        hits.append({
            "profile_id": int(pid), "split": _split(int(pid)),
            "minutes_above": round(float(tot), 1), "peak_c": round(float(g[col].max()), 1),
            "windows_min": [[round(s / SAMPLE_RATE_HZ / 60, 1), round(e / SAMPLE_RATE_HZ / 60, 1)] for s, e in w[:5]],
            "n_windows": len(w),
        })
    hits.sort(key=lambda h: -h["minutes_above"])
    return {"threshold_c": threshold_c, "component": component, "source": source,
            "n_cycles_exceeding": len(hits), "cycles": hits}


def what_if_frames(profile_id: int, coolant_delta_c: float = 0.0, current_scale: float = 1.0):
    """Baseline and modified virtual-sensor estimates for one drive cycle (full time series)."""
    if not (-30 <= coolant_delta_c <= 30):
        raise ValueError("coolant_delta_c must be within ±30 °C")
    if not (0.5 <= current_scale <= 1.5):
        raise ValueError("current_scale must be within 0.5–1.5")
    raw = _raw()
    g = raw[raw.profile_id == profile_id].reset_index(drop=True)
    if g.empty:
        _cycle(profile_id)  # raises the helpful error
    vs = _sensor()
    base = vs.estimate(g, start="stored")
    mod = g.copy()
    mod["coolant"] = mod["coolant"] + coolant_delta_c
    mod[["i_d", "i_q"]] = mod[["i_d", "i_q"]] * current_scale
    # initial component temperatures move with the coolant offset (motor in equilibrium at start)
    for t in TARGETS:
        mod.loc[0, t] = g.loc[0, t] + coolant_delta_c
    new = vs.estimate(mod, start="stored")

    lo, hi = float(raw.coolant.min()), float(raw.coolant.max())
    warnings = []
    if mod.coolant.max() > hi or mod.coolant.min() < lo:
        warnings.append(f"Modified coolant leaves the recorded range ({lo:.0f}–{hi:.0f} °C): the estimate is an extrapolation.")
    if current_scale > 1.0 and np.hypot(mod.i_d, mod.i_q).max() > np.hypot(raw.i_d, raw.i_q).max():
        warnings.append("Scaled currents exceed anything in the training data: the estimate is an extrapolation.")
    warnings.append("What-if results are model estimates, not measurements. Scaling currents does not re-solve the motor control, so voltages and speed stay unchanged.")
    return g, base, new, warnings


def what_if(profile_id: int, coolant_delta_c: float = 0.0, current_scale: float = 1.0) -> dict:
    """Re-run the virtual sensor on a drive cycle with shifted coolant temperature and/or scaled
    phase currents (a crude proxy for more/less load). Returns magnet and winding peaks before/after."""
    _, base, new, warnings = what_if_frames(profile_id, coolant_delta_c, current_scale)
    out = {"profile_id": profile_id, "coolant_delta_c": coolant_delta_c, "current_scale": current_scale}
    for name, col in [("magnet", "pm"), ("stator_winding", "stator_winding")]:
        out[name] = {"baseline_peak_c": round(float(base[col].max()), 1), "modified_peak_c": round(float(new[col].max()), 1),
                     "baseline_mean_c": round(float(base[col].mean()), 1), "modified_mean_c": round(float(new[col].mean()), 1)}
        out[name]["peak_change_k"] = round(out[name]["modified_peak_c"] - out[name]["baseline_peak_c"], 1)
    out["warnings"] = warnings
    return out


@lru_cache(maxsize=1)
def _sensor():
    from .sensor import VirtualSensor
    return VirtualSensor.load()


def model_card() -> dict:
    """How the virtual sensor works, how accurate it is, and its known limitations."""
    return {
        "what": "Estimates permanent-magnet (rotor) and three stator temperatures of a PMSM from signals measurable in a vehicle: currents, voltages, speed, coolant and ambient temperature. Torque is not used.",
        "model": "Average of (1) ridge regression on exponentially-weighted load history with a start-of-run correction and (2) a Thermal Neural Network (lumped thermal network with learned losses/conductances).",
        "data": "Paderborn University LEA PMSM test bench, 69 drive cycles, 185 h at 2 Hz (CC-BY-SA-4.0).",
        "accuracy_test": {"profiles": TEST_PROFILES, "mse_k2_avg_4_components": TEST_MSE_K2, "max_abs_error_k": TEST_LINF_K,
                          "published_best_mse_k2": 2.87, "published_best_max_error_k": 6.02},
        "limitations": [
            "Needs an initial temperature at start-up (the value stored at the last shutdown). Assuming a cold start after a hot restart can be ~30 K wrong for ~30 minutes.",
            "Trained on one motor on one test bench; other motors need re-training.",
            "Test set has only 3 cycles; accuracy on training-set cycles is optimistic.",
            "What-if analyses outside the recorded operating range are extrapolations.",
        ],
    }


TOOL_FUNCS = {f.__name__: f for f in [list_drive_cycles, get_cycle_summary, find_overheating, find_hot_restarts, what_if, model_card]}

TOOL_SCHEMAS = [
    {
        "name": "list_drive_cycles",
        "description": "List all recorded drive cycles (id, duration, max speed, start temperatures, peak magnet temperature measured and estimated, and whether the cycle is train/validation/test). Use it for overviews and to find cycles by property.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "get_cycle_summary",
        "description": "Detailed summary of ONE drive cycle: operating conditions, start temperatures, and the virtual sensor's accuracy per component (peak measured vs estimated, MSE, max error). Use before making claims about a specific cycle.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {"profile_id": {"type": "integer", "description": "Drive cycle id"}},
                         "required": ["profile_id"], "additionalProperties": False},
    },
    {
        "name": "find_overheating",
        "description": "Find drive cycles and time windows where a component's temperature exceeds a threshold. source='virtual_sensor' uses the estimate a vehicle would have; source='measured' uses lab ground truth.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {
            "threshold_c": {"type": "number", "description": "Temperature threshold in °C"},
            "component": {"type": "string", "enum": list(COMPONENTS)},
            "source": {"type": "string", "enum": ["virtual_sensor", "measured"]}},
            "required": ["threshold_c", "component", "source"], "additionalProperties": False},
    },
    {
        "name": "find_hot_restarts",
        "description": "Find drive cycles that start hot: the magnet is at least min_delta_k hotter than the coolant at the first sample. Use for questions about hot restarts or start-up temperature differences instead of computing from list_drive_cycles.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {"min_delta_k": {"type": "number", "description": "Minimum magnet-minus-coolant difference at start, in K"}},
                         "required": ["min_delta_k"], "additionalProperties": False},
    },
    {
        "name": "what_if",
        "description": "Re-run the virtual sensor on one drive cycle with coolant shifted by coolant_delta_c (°C, within ±30) and/or phase currents scaled by current_scale (0.5–1.5). Returns magnet and winding peak/mean before and after, plus extrapolation warnings that must be reported to the user.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {
            "profile_id": {"type": "integer"},
            "coolant_delta_c": {"type": "number"},
            "current_scale": {"type": "number"}},
            "required": ["profile_id", "coolant_delta_c", "current_scale"], "additionalProperties": False},
    },
    {
        "name": "model_card",
        "description": "How the virtual sensor works, its measured accuracy against published models, and known limitations. Use when asked how reliable or how built the estimates are.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
]
