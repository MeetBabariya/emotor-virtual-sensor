"""Precompute everything the serverless assistant (web-api/) needs, so it runs without
pandas/PyTorch: exact per-cycle summaries, overheating results for every integer threshold,
hot-restart deltas and what-if grids for the evaluation cycles.

    python -m src.export_api_data web-api/data
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

from . import tools as T
from .data import TEST_PROFILES
from .seq_model import VAL_PROFILES

THRESHOLDS = range(40, 161)                 # °C, integer steps
WHATIF_PROFILES = TEST_PROFILES + VAL_PROFILES
COOLANT = [-20, -10, 0, 10, 20]
CURRENT = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), default=float))
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB", flush=True)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    listing = T.list_drive_cycles()
    summaries = {str(c["profile_id"]): T.get_cycle_summary(c["profile_id"]) for c in listing["cycles"]}
    dump(out / "cycles.json", {"listing": listing, "summaries": summaries})

    over = {}
    for comp, src in product(T.COMPONENTS, ["virtual_sensor", "measured"]):
        for th in THRESHOLDS:
            r = T.find_overheating(th, comp, src)
            if r["n_cycles_exceeding"]:
                over[f"{comp}|{src}|{th}"] = r["cycles"]
    dump(out / "overheating.json", over)

    hot = T.find_hot_restarts(0.0)  # all positive deltas; the function filters by threshold
    dump(out / "hot_restarts.json", hot["cycles"])

    grid = {}
    for pid, dc, cs in product(WHATIF_PROFILES, COOLANT, CURRENT):
        grid[f"{pid}|{dc}|{cs:.1f}"] = T.what_if(pid, float(dc), float(cs))
        print("what-if", pid, dc, cs, flush=True)
    dump(out / "what_if.json", {"profiles": WHATIF_PROFILES, "coolant": COOLANT, "current": CURRENT, "grid": grid})
    dump(out / "model_card.json", T.model_card())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
