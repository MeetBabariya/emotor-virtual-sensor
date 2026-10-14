"""Export compact, precomputed JSON for the static case-study page (no server needed).

    python -m src.export_web /path/to/portfolio-site/public/data/emotor
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

from .data import ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES, load_demo
from .tools import what_if_frames

CYCLE_STEP = 60       # one point per 30 s for whole-cycle charts
RESTART_STEP = 20     # one point per 10 s for the first hour after a hot restart
COOLANT = [-20, -10, 0, 10, 20]
CURRENT = [0.8, 0.9, 1.0, 1.1, 1.2]


def r1(a) -> list[float]:
    return [round(float(x), 1) for x in a]


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = load_demo()
    from .sensor import VirtualSensor

    vs = VirtualSensor.load()

    # 1) test cycles: measured vs estimated, magnet + stator winding
    cycles = {}
    for pid in TEST_PROFILES:
        g = df[df.profile_id == pid].reset_index(drop=True)
        s = slice(None, None, CYCLE_STEP)
        cycles[str(pid)] = {
            "minutes": r1(np.arange(len(g))[s] / SAMPLE_RATE_HZ / 60),
            "speed_krpm": r1(g.motor_speed.to_numpy()[s] / 1000),
            **{f"{t}": r1(g[t].to_numpy()[s]) for t in ["pm", "stator_winding"]},
            **{f"{t}_est": r1(g[f"{t}_est"].to_numpy()[s]) for t in ["pm", "stator_winding"]},
            "metrics": {t: {"mse": round(float(np.mean((g[f"{t}_est"] - g[t]) ** 2)), 2),
                            "max_err": round(float(np.max(np.abs(g[f"{t}_est"] - g[t]))), 2),
                            "peak": round(float(g[t].max()), 1), "peak_est": round(float(g[f"{t}_est"].max()), 1)}
                        for t in TARGETS},
        }
    (out / "cycles.json").write_text(json.dumps(cycles, separators=(",", ":")))

    # 2) hot restart (profile 53): first hour for a grid of assumed start temperatures
    g = df[df.profile_id == 53].reset_index(drop=True)
    hour = g.iloc[: int(60 * 60 * SAMPLE_RATE_HZ)].reset_index(drop=True)
    true0 = hour.loc[0, TARGETS].to_numpy(float)
    cool0 = float(hour.loc[0, "coolant"])
    starts = list(range(int(round(cool0)), 61))
    curves = {}
    for sp in starts:
        frac = (sp - cool0) / (true0[0] - cool0)
        t0 = [cool0 + frac * (v - cool0) for v in true0]
        est = vs.estimate(hour, t0=t0)["pm"].to_numpy()
        err = np.abs(est - hour.pm.to_numpy())
        bad = np.flatnonzero(err > 3)
        curves[str(sp)] = {"pm_est": r1(est[::RESTART_STEP]), "worst": round(float(err.max()), 1),
                           "settle_min": round(float((bad[-1] + 1) / SAMPLE_RATE_HZ / 60), 1) if len(bad) else 0.0}
    restart = {"profile": 53, "true_start": round(float(true0[0]), 1), "coolant_start": round(cool0, 1),
               "minutes": r1(np.arange(len(hour))[::RESTART_STEP] / SAMPLE_RATE_HZ / 60),
               "pm": r1(hour.pm.to_numpy()[::RESTART_STEP]), "curves": curves}
    (out / "hot_restart.json").write_text(json.dumps(restart, separators=(",", ":")))

    # 3) what-if grid on the test cycles
    grid = {}
    for pid, dc, cs in product(TEST_PROFILES, COOLANT, CURRENT):
        g0, base, new, warn = what_if_frames(pid, float(dc), float(cs))
        key = f"{pid}|{dc}|{cs}"
        grid[key] = {"pm": r1(new["pm"].to_numpy()[::CYCLE_STEP]), "pm_peak": round(float(new["pm"].max()), 1),
                     "pm_mean": round(float(new["pm"].mean()), 1), "sw_peak": round(float(new["stator_winding"].max()), 1),
                     "min_above_100": round(float((new["pm"] > 100).sum() / SAMPLE_RATE_HZ / 60), 1),
                     "extrapolation": any("extrapolation" in w for w in warn)}
        print("what-if", key, grid[key]["pm_peak"], flush=True)
    (out / "what_if.json").write_text(json.dumps({"coolant": COOLANT, "current": CURRENT, "profiles": TEST_PROFILES,
                                                  "minutes": {str(p): cycles[str(p)]["minutes"] for p in TEST_PROFILES},
                                                  "grid": grid}, separators=(",", ":")))

    # 4) recorded assistant evaluation
    ev = json.loads((ROOT / "artifacts" / "assistant_eval.json").read_text())
    runs = [{"id": c["id"], "question": c["question"], "passed": c["passed"], "answer": c["answer"],
             "tools": [{"name": t["name"], "input": t["input"]} for t in c["tool_calls"]]} for c in ev["cases"]]
    (out / "assistant.json").write_text(json.dumps({"summary": ev["summary"], "runs": runs}, separators=(",", ":")))

    for f in sorted(out.iterdir()):
        print(f.name, f"{f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
