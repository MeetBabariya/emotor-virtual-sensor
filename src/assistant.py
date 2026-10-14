"""Engineering assistant: an LLM answers questions about the motor by calling analysis tools.

Hand-written agent loop (no framework) so every step is visible:
request → function calls → run Python tools → results → repeat until the model answers.

Two interchangeable backends, chosen by LLM_PROVIDER in .env:
  * "gemini" (default)  – Google Gemini via the Interactions API (free tier)
  * "claude"            – Anthropic Claude Opus 5.5 via the Messages API (paid API credit)

    python -m src.assistant "Which drive cycles push the magnet above 100 °C?"
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import os

from dotenv import load_dotenv

from .data import ROOT
from .tools import TOOL_FUNCS, TOOL_SCHEMAS

load_dotenv(ROOT / ".env")

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # free tier; gemini-3-flash-preview allows only 20 requests/day
MODEL = GEMINI_MODEL if PROVIDER == "gemini" else CLAUDE_MODEL
MAX_TURNS = 8
REQUEST_TIMEOUT_S = 60  # per HTTP request; free-tier calls can otherwise hang indefinitely

SYSTEM = """You are an engineering assistant for a thermal virtual sensor on a permanent-magnet synchronous motor (PMSM) used in electric vehicles. Engineers ask about drive cycles, temperatures, overheating risk and what-if scenarios.

Ground rules:
- Every number you state must come from a tool result in this conversation. Do not estimate, recall or invent values. If the tools cannot answer, say so.
- "Magnet" means the rotor permanent-magnet temperature (column pm), which cannot be measured in a production vehicle; the virtual sensor estimates it. Make clear whether a figure is measured (lab ground truth) or estimated (virtual sensor).
- Cycles in the train or validation split were seen during development, so accuracy there is optimistic; say so when it matters.
- Always pass on what_if warnings, especially extrapolation warnings.
- Answer in a few short paragraphs or a small table. Lead with the answer, cite drive-cycle ids, then add caveats. Use °C for temperatures and K for differences."""


@dataclass
class Result:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str = ""


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"Unknown tool {name}", True
    try:
        return json.dumps(fn(**args), default=float), False
    except (ValueError, TypeError) as e:
        return f"Error: {e}", True


def ask(question: str, client=None, verbose: bool = False) -> Result:
    return (_ask_gemini if PROVIDER == "gemini" else _ask_claude)(question, client, verbose)


def _record(res: Result, name: str, args: dict, verbose: bool) -> tuple[str, bool]:
    out, is_err = run_tool(name, args)
    res.tool_calls.append({"name": name, "input": args, "error": is_err})
    if verbose:
        print(f"  → {name}({json.dumps(args)}){'  [error]' if is_err else ''}", file=sys.stderr)
    return out, is_err


def _gemini_tools() -> list[dict]:
    def clean(schema):  # Gemini function schemas: plain JSON schema without additionalProperties
        if isinstance(schema, dict):
            return {k: clean(v) for k, v in schema.items() if k != "additionalProperties"}
        return schema
    return [{"type": "function", "name": t["name"], "description": t["description"],
             "parameters": clean(t["input_schema"])} for t in TOOL_SCHEMAS]


def _with_retry(fn, attempts: int = 3, waits=(10, 30)):
    """Free-tier endpoints occasionally drop connections or rate-limit; retry with backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # SDK raises several connection / status error types
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if "per day" in str(e):  # daily free-tier quota: waiting minutes will not help
                raise RuntimeError(f"Free-tier daily quota exhausted for {GEMINI_MODEL}: {e}") from e
            retryable = status in (None, 408, 429, 500, 502, 503, 504)
            if i == attempts - 1 or not retryable:
                raise
            print(f"  (retrying after {type(e).__name__}: waiting {waits[min(i, len(waits) - 1)]}s)", file=sys.stderr)
            time.sleep(waits[min(i, len(waits) - 1)])


def _ask_gemini(question: str, client=None, verbose: bool = False) -> Result:
    from google import genai
    from google.genai import types

    # attempts=1: disable the SDK's silent exponential-backoff retries (it can wait for hours on a
    # rate-limit error); _with_retry above handles transient failures visibly instead.
    client = client or genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    tools = _gemini_tools()
    res = Result(answer="")
    t0 = time.time()
    interaction = _with_retry(lambda: client.interactions.create(
        model=GEMINI_MODEL, system_instruction=SYSTEM, input=question, tools=tools, timeout=REQUEST_TIMEOUT_S))
    for _ in range(MAX_TURNS):
        u = interaction.usage
        if u is not None:
            res.input_tokens += u.total_input_tokens or 0
            res.output_tokens += (u.total_output_tokens or 0) + (u.total_thought_tokens or 0)
        res.stop_reason = str(interaction.status)
        calls = [s for s in (interaction.steps or []) if s.type == "function_call"]
        if not calls:
            res.answer = (interaction.output_text or "").strip()
            break
        results = []
        for c in calls:  # parallel calls are answered together in one request
            out, is_err = _record(res, c.name, dict(c.arguments or {}), verbose)
            results.append({"type": "function_result", "name": c.name, "call_id": c.id,
                            "result": [{"type": "text", "text": out}], "is_error": is_err})
        prev = interaction.id
        interaction = _with_retry(lambda: client.interactions.create(
            model=GEMINI_MODEL, system_instruction=SYSTEM, input=results, tools=tools, previous_interaction_id=prev, timeout=REQUEST_TIMEOUT_S))
    else:
        res.answer = "Stopped: too many tool-call rounds without a final answer."
    res.seconds = round(time.time() - t0, 1)
    return res


def _ask_claude(question: str, client=None, verbose: bool = False) -> Result:
    import anthropic

    client = client or anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    res = Result(answer="")
    t0 = time.time()
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        res.input_tokens += response.usage.input_tokens
        res.output_tokens += response.usage.output_tokens
        res.stop_reason = response.stop_reason

        if response.stop_reason == "refusal":
            res.answer = "The request was declined by the model's safety system."
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            res.answer = "\n".join(b.text for b in response.content if b.type == "text").strip()
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:  # Claude may call several tools at once; answer them in one message
            out, is_err = _record(res, tu.name, tu.input, verbose)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": is_err})
        messages.append({"role": "user", "content": results})
    else:
        res.answer = res.answer or "Stopped: too many tool-call rounds without a final answer."
    res.seconds = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which drive cycles push the magnet above 100 °C, and does the virtual sensor catch all of them?"
    r = ask(q, verbose=True)
    print(r.answer)
    print(f"\n[{len(r.tool_calls)} tool calls · {r.input_tokens:,} in / {r.output_tokens:,} out tokens · {r.seconds}s]", file=sys.stderr)
