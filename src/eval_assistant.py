"""Evaluation set for the engineering assistant.

Every expected answer is computed from the data/tools at run time (no hand-typed numbers), and
grading is deterministic: required drive-cycle ids, numbers within tolerance, required caveats,
forbidden content, and the tool that should have been used.

    python -m src.eval_assistant            # all cases
    python -m src.eval_assistant --only 3   # one case
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import time
from dataclasses import dataclass, field

from .assistant import MODEL, PROVIDER, ask
from .data import ROOT
from .tools import find_overheating, get_cycle_summary, list_drive_cycles, model_card, what_if

# Claude Opus 5.5 list prices, USD per million tokens (input, output). Gemini free tier: $0.
PRICE_IN, PRICE_OUT = (4.0, 20.0) if PROVIDER == "claude" else (0.0, 0.0)


@dataclass
class Case:
    id: str
    question: str
    must_ids: set[int] = field(default_factory=set)        # drive-cycle ids that must appear
    must_not_ids: set[int] = field(default_factory=set)    # ids that must NOT appear
    numbers: list[tuple[float, float]] = field(default_factory=list)  # (value, tolerance) that must appear
    must_text: list[list[str]] = field(default_factory=list)  # each inner list: at least one phrase must appear
    must_not_text: list[str] = field(default_factory=list)
    expect_tool: str | tuple[str, ...] | None = None  # any of these tools counts


def build_cases() -> list[Case]:
    cyc = {c["profile_id"]: c for c in list_drive_cycles()["cycles"]}
    meas100 = {c["profile_id"] for c in find_overheating(100, "magnet", "measured")["cycles"]}
    est100 = {c["profile_id"] for c in find_overheating(100, "magnet", "virtual_sensor")["cycles"]}
    missed = meas100 - est100
    win130 = {c["profile_id"] for c in find_overheating(130, "stator_winding", "measured")["cycles"]}
    hot = {p for p, c in cyc.items() if c["magnet_start_c"] - c["coolant_start_c"] >= 10}
    longest = max(cyc.values(), key=lambda c: c["duration_min"])
    s74, s60, s65 = get_cycle_summary(74), get_cycle_summary(60), get_cycle_summary(65)
    w62 = what_if(62, 10, 1.0)
    w74 = what_if(74, 0, 1.3)
    card = model_card()

    return [
        Case("overheat_measured", "Which drive cycles actually pushed the magnet above 100 °C in the lab measurements?",
             must_ids=meas100, expect_tool="find_overheating"),
        Case("sensor_misses", "If a car relied on the virtual sensor with a 100 °C magnet limit, would it have caught every drive cycle where the real magnet went above 100 °C?",
             must_ids=missed, must_text=[["no", "not", "miss"]], expect_tool="find_overheating"),
        Case("peak_74", "What was the peak measured magnet temperature in drive cycle 74?",
             numbers=[(s74["components"]["magnet"]["peak_measured_c"], 0.15)], expect_tool="get_cycle_summary"),
        Case("error_60", "What is the virtual sensor's maximum magnet error on drive cycle 60?",
             numbers=[(s60["components"]["magnet"]["max_abs_error_k"], 0.06)], expect_tool="get_cycle_summary"),
        Case("whatif_coolant", "In drive cycle 62, how much would the peak magnet temperature change if the coolant were 10 °C warmer?",
             numbers=[(w62["magnet"]["peak_change_k"], 0.15)], expect_tool="what_if"),
        Case("whatif_current", "Estimate the peak magnet temperature in drive cycle 74 if the phase currents were 30% higher.",
             numbers=[(w74["magnet"]["modified_peak_c"], 0.15)],
             must_text=[["extrapolat", "outside", "beyond", "exceed"]], expect_tool="what_if"),
        Case("longest", "Which drive cycle is the longest, and how long is it in minutes?",
             must_ids={longest["profile_id"]}, numbers=[(longest["duration_min"], 0.15)], expect_tool="list_drive_cycles"),
        Case("hot_restarts", "Which drive cycles start with the magnet at least 10 K hotter than the coolant?",
             must_ids=hot, expect_tool="find_hot_restarts"),
        Case("winding_130", "Which drive cycles had a measured stator winding temperature above 130 °C?",
             must_ids=win130, expect_tool="find_overheating"),
        Case("speed_65", "What is the maximum motor speed in drive cycle 65?",
             numbers=[(s65["speed_rpm"]["max"], 1.5)], expect_tool=("get_cycle_summary", "list_drive_cycles")),
        Case("accuracy_vs_published", "How accurate is the virtual sensor compared with the best published model?",
             numbers=[(card["accuracy_test"]["mse_k2_avg_4_components"], 0.005), (card["accuracy_test"]["published_best_mse_k2"], 0.005)],
             expect_tool="model_card"),
        Case("nonexistent_cycle", "What was the peak magnet temperature in drive cycle 999?",
             must_text=[["does not exist", "doesn't exist", "no drive cycle", "not exist", "no cycle", "isn't", "not found", "no such"]],
             must_not_text=["°C"]),
        Case("out_of_scope", "Predict the remaining useful life of the motor bearings for drive cycle 20.",
             must_text=[["can't", "cannot", "not able", "unable", "no tool", "don't have", "do not have", "not available", "outside",
                         "does not provide", "doesn't provide", "not designed", "does not support", "not supported"]],
             must_not_text=["hours remaining", "remaining useful life is"]),
    ]


def numbers_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def grade(case: Case, answer: str, tools_used: list[str]) -> dict:
    low = answer.lower()
    ids = {int(x) for x in re.findall(r"(?:cycle|profile|#|id)s?\s*(?:ids?\s*)?[:#]?\s*(\d{1,3})", low)}
    ids |= {int(x) for x in re.findall(r"\b(\d{1,3})\b", answer)}  # tables / lists of ids
    nums = numbers_in(answer)
    checks = {}
    if case.must_ids:
        checks["ids_present"] = case.must_ids <= ids
    if case.must_not_ids:
        checks["ids_absent"] = not (case.must_not_ids & ids)
    for i, (v, tol) in enumerate(case.numbers):
        checks[f"number_{i}={v}"] = any(abs(n - v) <= tol for n in nums)
    for i, group in enumerate(case.must_text):
        checks[f"text_{i}"] = any(p.lower() in low for p in group)
    for p in case.must_not_text:
        checks[f"no:{p}"] = p.lower() not in low
    if case.expect_tool:
        allowed = (case.expect_tool,) if isinstance(case.expect_tool, str) else case.expect_tool
        checks[f"tool:{'|'.join(allowed)}"] = any(t in tools_used for t in allowed)
    return {"passed": all(checks.values()), "checks": checks}


class QuestionTimeout(BaseException):
    """BaseException so the request-level retry (which catches Exception) cannot swallow it."""


def ask_with_deadline(question: str, seconds: int = 180, attempts: int = 2):
    """Hard wall-clock limit per question (a hung HTTP request must not stall the whole run)."""
    def on_alarm(*_):
        raise QuestionTimeout(f"no answer within {seconds}s")
    signal.signal(signal.SIGALRM, on_alarm)
    for i in range(attempts):
        signal.alarm(seconds)
        try:
            return ask(question)
        except QuestionTimeout:
            if i == attempts - 1:
                raise
            print("  (question timed out, retrying once)", flush=True)
        finally:
            signal.alarm(0)


def regrade() -> None:
    """Re-apply the (current) grader to stored answers without calling the model again."""
    path = ROOT / "reports" / "assistant_eval.json"
    d = json.loads(path.read_text())
    cases = {c.id: c for c in build_cases()}
    for row in d["cases"]:
        g = grade(cases[row["id"]], row["answer"], [t["name"] for t in row["tool_calls"]])
        row.update(g)
    d["summary"]["passed"] = sum(r["passed"] for r in d["cases"])
    d["summary"]["regraded_at"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(d, indent=2))
    for r in d["cases"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
    print(f"{d['summary']['passed']}/{d['summary']['total']} passed (regraded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="run only case number N (0-based)")
    ap.add_argument("--regrade", action="store_true", help="re-grade stored answers, no model calls")
    a = ap.parse_args()
    if a.regrade:
        return regrade()
    cases = build_cases()
    if a.only is not None:
        cases = [cases[a.only]]
    rows, tin, tout = [], 0, 0
    for i, c in enumerate(cases):
        if i and PROVIDER == "gemini":
            time.sleep(6)  # stay inside free-tier requests-per-minute limits
        try:
            r = ask_with_deadline(c.question)
        except Exception as e:  # count infrastructure failures, don't crash the run
            rows.append({"id": c.id, "question": c.question, "passed": False, "checks": {"error": False},
                         "answer": f"ERROR {type(e).__name__}: {e}", "tool_calls": [], "seconds": 0, "stop_reason": "error"})
            print(f"ERR   {c.id:24s} {type(e).__name__}", flush=True)
            continue
        g = grade(c, r.answer, [t["name"] for t in r.tool_calls])
        tin, tout = tin + r.input_tokens, tout + r.output_tokens
        rows.append({"id": c.id, "question": c.question, **g, "answer": r.answer,
                     "tool_calls": r.tool_calls, "seconds": r.seconds, "stop_reason": r.stop_reason})
        print(f"{'PASS' if g['passed'] else 'FAIL'}  {c.id:24s} {len(r.tool_calls)} tools  {r.seconds:5.1f}s  "
              + ", ".join(k for k, v in g["checks"].items() if not v), flush=True)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    summary = {"provider": PROVIDER, "model": MODEL, "passed": sum(r["passed"] for r in rows), "total": len(rows),
               "input_tokens": tin, "output_tokens": tout, "approx_cost_usd": round(cost, 2),
               "run_at": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "reports" / "assistant_eval.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    print(f"\n{summary['passed']}/{summary['total']} passed · {tin:,} in / {tout:,} out tokens · ≈${cost:.2f}")


if __name__ == "__main__":
    main()
