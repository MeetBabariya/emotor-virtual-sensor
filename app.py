"""E-motor virtual temperature sensor — interactive demo.

    streamlit run app.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data import ROOT, SAMPLE_RATE_HZ, TARGETS, TEST_PROFILES
from src.seq_model import VAL_PROFILES

st.set_page_config(page_title="E-motor Virtual Sensor", page_icon="🌡️", layout="wide")
load_dotenv(ROOT / ".env")  # local development; on a host, set GEMINI_API_KEY as a secret instead

ART = ROOT / "artifacts"
ACCENT, INK, MUTED, GRID = "#ff8a3d", "#e9e9ec", "#8b8e98", "rgba(255,255,255,0.07)"
BLUE = "#6c8cff"
NAMES = {"pm": "Magnet (rotor)", "stator_yoke": "Stator yoke", "stator_tooth": "Stator tooth", "stator_winding": "Stator winding"}

st.markdown(
    """
<style>
  .block-container {padding-top: 3.2rem; max-width: 1200px;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  .kicker {font: 500 0.75rem ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .08em;
           text-transform: uppercase; color: #ff8a3d; margin-bottom: .2rem;}
  .lede {color: #a7a9b1; font-size: 1.08rem; max-width: 46rem; line-height: 1.55;}
  .card {border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 1rem 1.1rem; background: #121418; height: 100%;}
  .card .v {font-size: 1.9rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1;}
  .card .l {color: #8b8e98; font-size: .85rem; margin-top: .25rem;}
  .card .s {color: #ff8a3d; font: 500 .75rem ui-monospace, Menlo, monospace; margin-top: .45rem;}
  .step {border-left: 2px solid #ff8a3d; padding: .2rem 0 .2rem .9rem; margin-bottom: .9rem;}
  .step b {display:block; margin-bottom: .15rem;}
  .step span {color: #a7a9b1; font-size: .92rem;}
  .chip {display:inline-block; font: 500 .75rem ui-monospace, Menlo, monospace; padding: .12rem .5rem; border-radius: 6px;
         border: 1px solid rgba(255,255,255,.12); color: #c9cbd2; margin: 0 .3rem .3rem 0;}
  .pass {color:#5fd39a;} .fail {color:#ff6b6b;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading 185 hours of test-bench data…")
def data() -> pd.DataFrame:
    """Raw signals + virtual-sensor estimates (compact 0.01-resolution copy shipped in artifacts/)."""
    from src.data import load_demo

    return load_demo()


@st.cache_resource(show_spinner=False)
def sensor():
    from src.sensor import VirtualSensor

    return VirtualSensor.load()


def split_of(pid: int) -> str:
    return "test" if pid in TEST_PROFILES else "validation" if pid in VAL_PROFILES else "train"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13), legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def card(value: str, label: str, sub: str = "") -> str:
    return f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div>{f"<div class=s>{sub}</div>" if sub else ""}</div>'


# ---------------------------------------------------------------- pages
def overview():
    st.markdown('<div class="kicker">Virtual sensing · PMSM · Thermal ML + LLM assistant</div>', unsafe_allow_html=True)
    st.title("Estimating what a car can't measure: rotor-magnet temperature")
    st.markdown(
        '<p class="lede">The permanent magnets inside an electric-vehicle motor lose strength for good if they overheat, '
        "but you can't put a sensor on a spinning rotor. This project estimates magnet and stator temperatures from "
        "signals a car <i>can</i> measure (currents, voltages, speed and coolant temperature), validated on a published "
        "benchmark, and adds an AI assistant that answers engineering questions using the model.</p>",
        unsafe_allow_html=True,
    )
    c = st.columns(4)
    c[0].markdown(card("1.50 K²", "Average error on unseen test cycles (MSE)", "published best: 2.87"), unsafe_allow_html=True)
    c[1].markdown(card("5.73 K", "Worst-case error on test cycles", "published best: 6.02"), unsafe_allow_html=True)
    c[2].markdown(card("185 h", "Test-bench recordings · 69 drive cycles", "Paderborn University (CC-BY-SA)"), unsafe_allow_html=True)
    ev = json.loads((ART / "assistant_eval.json").read_text())["summary"]
    c[3].markdown(card(f"{ev['passed']}/{ev['total']}", "Assistant evaluation questions passed", "graded automatically"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Against the published benchmark")
        rows = [("MLP", 5.58, 14.29), ("Linear regression (OLS)", 4.47, 9.85), ("CNN", 4.43, 15.54),
                ("Lumped thermal network", 3.64, 7.37), ("Thermal Neural Network (best published)", 2.87, 6.02),
                ("This project (ensemble)", 1.50, 5.73)]
        names, mse, linf = zip(*rows)
        colors = [MUTED] * 5 + [ACCENT]
        fig = go.Figure(go.Bar(x=mse, y=names, orientation="h", marker_color=colors, text=[f"{v:.2f}" for v in mse],
                               textposition="outside", hovertemplate="%{y}: %{x:.2f} K²<extra></extra>"))
        fig.update_layout(xaxis_title="Mean squared error, K² (lower is better)", hovermode="closest", xaxis_range=[0, 6.6])
        st.plotly_chart(style(fig, 330), use_container_width=True)
        st.caption("Generalization set: drive cycles 60, 62, 74; average over magnet + 3 stator temperatures. Published numbers: "
                   "Kirchgässner, Wallscheid & Böcker, *Thermal neural networks*, Eng. Appl. of AI (2023), Table III. "
                   "Published models also used torque as an input; this project does not, because a car cannot measure it reliably.")
    with right:
        st.subheader("How it works")
        for b, t in [
            ("1 · Load history features", "Temperature depends on what the motor did over the last minutes to hours, so currents and voltages are summarised as exponentially weighted averages over 7–53 minutes."),
            ("2 · Two complementary models", "A regularised linear model on those features, plus a Thermal Neural Network: a heat-flow network whose losses and conductances are small neural nets, integrated every 0.5 s."),
            ("3 · Start from a stored temperature", "Both models start from the temperature saved at the last shutdown. A wrong start assumption costs ~30 K for half an hour."),
            ("4 · Average the two", "Their errors tend to point in opposite directions, so the average beats both: 1.50 K² vs 2.09 and 2.70."),
            ("5 · An assistant on top", "An LLM answers questions by calling six analysis tools; every number it states comes from code, and a 13-question evaluation checks it."),
        ]:
            st.markdown(f'<div class="step"><b>{b}</b><span>{t}</span></div>', unsafe_allow_html=True)


def explorer():
    st.markdown('<div class="kicker">Drive-cycle explorer</div>', unsafe_allow_html=True)
    st.header("Virtual sensor vs. lab measurement")
    df = data()
    pids = sorted(df.profile_id.unique().tolist(), key=lambda p: (split_of(p) != "test", p))
    dur = df.groupby("profile_id").size() / SAMPLE_RATE_HZ / 60
    a, b = st.columns([1, 2])
    pid = a.selectbox("Drive cycle", pids, format_func=lambda p: f"#{p} · {split_of(p)} · {dur[p]:.0f} min")
    comp = b.radio("Component", TARGETS, format_func=NAMES.get, horizontal=True)
    if split_of(pid) != "test":
        st.info(f"Cycle {pid} is in the **{split_of(pid)}** set, so the model saw it during development: accuracy here is optimistic. "
                "Cycles 60, 62 and 74 are the untouched test cycles.", icon="ℹ️")
    g = df[df.profile_id == pid].reset_index(drop=True)
    y, yh = g[comp].to_numpy(), g[f"{comp}_est"].to_numpy()
    e = yh - y
    m = st.columns(4)
    m[0].metric("Peak measured", f"{y.max():.1f} °C")
    m[1].metric("Peak estimated", f"{yh.max():.1f} °C", f"{yh.max() - y.max():+.1f} K", delta_color="off")
    m[2].metric("Mean squared error", f"{np.mean(e ** 2):.2f} K²")
    m[3].metric("Worst-case error", f"{np.abs(e).max():.2f} K")

    step = 4  # plot at 0.5 Hz for speed; metrics above use every sample
    t = np.arange(len(g))[::step] / SAMPLE_RATE_HZ / 60
    fig = go.Figure()
    fig.add_scatter(x=t, y=y[::step], name="Measured (lab only)", line=dict(color=INK, width=1.6))
    fig.add_scatter(x=t, y=yh[::step], name="Virtual sensor", line=dict(color=ACCENT, width=1.4))
    fig.update_layout(yaxis_title="°C", xaxis_title="minutes")
    st.plotly_chart(style(fig, 360), use_container_width=True)

    with st.expander("Operating conditions and error over time"):
        f2 = go.Figure()
        f2.add_scatter(x=t, y=e[::step], name="Estimate − measured (K)", line=dict(color=ACCENT, width=1))
        f2.add_scatter(x=t, y=g.motor_speed.to_numpy()[::step] / 1000, name="Speed (×1000 rpm)", line=dict(color=BLUE, width=1), yaxis="y2")
        f2.add_scatter(x=t, y=g.coolant.to_numpy()[::step], name="Coolant (°C)", line=dict(color=MUTED, width=1, dash="dot"), yaxis="y2")
        f2.update_layout(yaxis=dict(title="error, K"), yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis_title="minutes")
        st.plotly_chart(style(f2, 300), use_container_width=True)


def hot_restart():
    st.markdown('<div class="kicker">The finding the benchmark hides</div>', unsafe_allow_html=True)
    st.header("Hot restarts: why the car must remember its last estimate")
    st.markdown(
        '<p class="lede">All three official test cycles start with a cool motor. Real cars restart hot: drive, stop for coffee, '
        "drive again. In cycle 53 the magnet starts at 50.6 °C while the coolant reads 18.6 °C. Pick the starting temperature "
        "the car's controller believes and watch how long the estimate takes to recover.</p>",
        unsafe_allow_html=True,
    )
    df = data()
    g = df[df.profile_id == 53].reset_index(drop=True)
    true0 = g.loc[0, TARGETS].to_numpy(float)
    cool0 = float(g.loc[0, "coolant"])
    guess = st.slider("Magnet temperature the controller assumes at restart (°C)", float(round(cool0)), 60.0, float(round(true0[0], 1)), 0.5)

    @st.cache_data(show_spinner="Re-running the virtual sensor…")
    def run(start_pm: float) -> np.ndarray:
        scale = start_pm - cool0
        frac = scale / (true0[0] - cool0) if true0[0] != cool0 else 1.0
        t0 = [cool0 + frac * (v - cool0) for v in true0]  # stator nodes scaled consistently with the magnet guess
        return sensor().estimate(g, t0=t0)["pm"].to_numpy()

    est = run(round(guess, 1))
    n = int(60 * 60 * SAMPLE_RATE_HZ)
    t = np.arange(n) / SAMPLE_RATE_HZ / 60
    err = np.abs(est[:n] - g.pm.to_numpy()[:n])
    ok = np.flatnonzero(err > 3)
    settle = (ok[-1] + 1) / SAMPLE_RATE_HZ / 60 if len(ok) else 0.0
    m = st.columns(3)
    m[0].metric("Assumed start error", f"{guess - true0[0]:+.1f} K")
    m[1].metric("Worst error in first hour", f"{err.max():.1f} K")
    m[2].metric("Minutes until within ±3 K", f"{settle:.0f}" if settle < 60 else "> 60")
    fig = go.Figure()
    fig.add_scatter(x=t, y=g.pm.to_numpy()[:n], name="Measured magnet", line=dict(color=INK, width=1.6))
    fig.add_scatter(x=t, y=est[:n], name="Virtual sensor", line=dict(color=ACCENT, width=1.4))
    fig.update_layout(yaxis_title="°C", xaxis_title="minutes after restart", yaxis_range=[15, 75])
    st.plotly_chart(style(fig, 380), use_container_width=True)
    st.caption("Engineering takeaway: store the estimate at shutdown and model cool-down while parked. The start-of-run correction "
               "and the Thermal Neural Network both take this stored value as their initial state.")


def what_if_page():
    st.markdown('<div class="kicker">Scenario analysis</div>', unsafe_allow_html=True)
    st.header("What if the coolant were warmer, or the load higher?")
    df = data()
    pids = TEST_PROFILES + [p for p in VAL_PROFILES]
    a, b, c = st.columns([1.2, 1, 1])
    pid = a.selectbox("Drive cycle", pids, format_func=lambda p: f"#{p} · {split_of(p)}")
    dc = b.slider("Coolant offset (°C)", -20, 20, 10, 1)
    cs = c.slider("Phase-current scale", 0.7, 1.3, 1.0, 0.05)

    @st.cache_data(show_spinner="Re-running the virtual sensor on the modified cycle…")
    def run(pid: int, dc: float, cs: float):
        from src.tools import what_if_frames

        g, base, new, warn = what_if_frames(pid, dc, cs)
        return base["pm"].to_numpy(), new["pm"].to_numpy(), base["stator_winding"].to_numpy(), new["stator_winding"].to_numpy(), warn

    bp, np_, bw, nw, warn = run(pid, float(dc), float(cs))
    m = st.columns(4)
    m[0].metric("Magnet peak", f"{np_.max():.1f} °C", f"{np_.max() - bp.max():+.1f} K", delta_color="inverse")
    m[1].metric("Magnet mean", f"{np_.mean():.1f} °C", f"{np_.mean() - bp.mean():+.1f} K", delta_color="inverse")
    m[2].metric("Winding peak", f"{nw.max():.1f} °C", f"{nw.max() - bw.max():+.1f} K", delta_color="inverse")
    m[3].metric("Time magnet > 100 °C", f"{(np_ > 100).sum() / SAMPLE_RATE_HZ / 60:.1f} min")
    step = 4
    t = np.arange(len(bp))[::step] / SAMPLE_RATE_HZ / 60
    fig = go.Figure()
    fig.add_scatter(x=t, y=bp[::step], name="Magnet · as recorded", line=dict(color=MUTED, width=1.3))
    fig.add_scatter(x=t, y=np_[::step], name="Magnet · scenario", line=dict(color=ACCENT, width=1.5))
    fig.add_hline(y=100, line=dict(color="#ff6b6b", width=1, dash="dot"), annotation_text="100 °C", annotation_font_color="#ff6b6b")
    fig.update_layout(yaxis_title="°C", xaxis_title="minutes")
    st.plotly_chart(style(fig, 360), use_container_width=True)
    for w in warn:
        (st.warning if "extrapolation" in w else st.caption)(w)


def assistant_page():
    st.markdown('<div class="kicker">LLM + tools · evaluated</div>', unsafe_allow_html=True)
    st.header("Ask the engineering assistant")
    st.markdown(
        '<p class="lede">The assistant never answers from memory. It chooses among six analysis tools (cycle summaries, overheating '
        "search, hot-restart detection, what-if simulation, model card), the code computes the numbers, and the LLM explains them. "
        "A 13-question evaluation with automatically computed answers checks that it stays correct.</p>",
        unsafe_allow_html=True,
    )
    key = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") and _has_secrets() else None)
    live, recorded = st.tabs(["Ask a question", "Recorded evaluation runs"])
    with recorded:
        ev = json.loads((ART / "assistant_eval.json").read_text())
        s = ev["summary"]
        st.caption(f"{s['passed']}/{s['total']} passed · model {s['model']} · run {s.get('run_at', '')}")
        for c in ev["cases"]:
            badge = '<span class="pass">PASS</span>' if c["passed"] else '<span class="fail">FAIL</span>'
            with st.expander(f"{'✅' if c['passed'] else '❌'}  {c['question']}"):
                st.markdown(" ".join(f'<span class="chip">{t["name"]}({", ".join(f"{k}={v}" for k, v in t["input"].items())})</span>'
                                     for t in c["tool_calls"]) or '<span class="chip">no tool call</span>', unsafe_allow_html=True)
                st.markdown(c["answer"])
                st.markdown(badge + " &nbsp; " + " · ".join(f"{k} {'✓' if v else '✗'}" for k, v in c["checks"].items()), unsafe_allow_html=True)
    with live:
        if not key:
            st.info("Live questions need a free Gemini API key (set `GEMINI_API_KEY`). Meanwhile, browse the recorded evaluation runs.", icon="🔑")
            return
        st.session_state.setdefault("asked", 0)
        examples = ["Which drive cycles push the magnet above 100 °C, and does the virtual sensor catch all of them?",
                    "Which cycles are hot restarts, and why do they matter?",
                    "In cycle 62, what happens to the magnet peak if the coolant is 10 °C warmer?"]
        cols = st.columns(3)
        picked = next((q for q, col in zip(examples, cols) if col.button(q, use_container_width=True)), None)
        q = st.chat_input("Ask about drive cycles, temperatures, overheating or what-if scenarios…") or picked
        if q:
            if st.session_state.asked >= 5:
                st.warning("This free demo allows 5 questions per session.")
                return
            st.session_state.asked += 1
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                from src.assistant import ask

                with st.status("Thinking and calling tools…", expanded=False) as status:
                    try:
                        r = ask(q)
                    except Exception as e:  # quota exhausted, network, etc.
                        status.update(label="The free model is unavailable right now", state="error")
                        st.error(f"{type(e).__name__}: {e}")
                        return
                    for t in r.tool_calls:
                        st.markdown(f'<span class="chip">{t["name"]}({", ".join(f"{k}={v}" for k, v in t["input"].items())})</span>', unsafe_allow_html=True)
                    status.update(label=f"Used {len(r.tool_calls)} tool call(s) · {r.seconds}s", state="complete")
                st.markdown(r.answer)


def _has_secrets() -> bool:
    try:
        return "GEMINI_API_KEY" in st.secrets
    except Exception:
        return False


def about():
    st.markdown('<div class="kicker">About</div>', unsafe_allow_html=True)
    st.header("Method, honesty notes and credits")
    st.markdown(
        """
**Evaluation discipline.** Test cycles 60/62/74 (the dataset authors' generalization set) were scored once per final
configuration. Model choices were made with grouped cross-validation on the training cycles, plus a separate validation set
that deliberately includes hot restarts. One early bug (EWMA warm-up) was found while inspecting test errors; after that,
selection moved entirely to cross-validation.

**What did not work.** A plain GRU (2.79 K²) learned to ignore its initial state. Gradient boosting was worse than the
linear model in cross-validation. The first assistant version failed a question because the LLM had to do arithmetic over
69 rows; the fix was a new tool, not a better prompt.

**Limitations.** One motor on one test bench; only three test cycles; what-if analyses outside the recorded range are
extrapolations; at a 100 °C threshold the sensor misses one cycle whose real peak is 102.1 °C, so a deployed system needs margin.

**Credits.** Data: Kirchgässner, Wallscheid & Böcker, Paderborn University LEA — *Electric Motor Temperature* (Kaggle, CC-BY-SA-4.0).
TNN architecture after Kirchgässner et al., *Thermal neural networks*, Eng. Appl. of AI (2023), reference code github.com/wkirgsn/thermal-nn.

Built by **Meet Babariya** · [portfolio](https://meetbabariya.github.io) · [GitHub](https://github.com/MeetBabariya)
"""
    )


pg = st.navigation(
    [
        st.Page(overview, title="Overview", icon="🏁", default=True),
        st.Page(explorer, title="Drive-cycle explorer", icon="📈", url_path="explorer"),
        st.Page(hot_restart, title="Hot restart", icon="🔥", url_path="hot-restart"),
        st.Page(what_if_page, title="What-if", icon="🧪", url_path="what-if"),
        st.Page(assistant_page, title="Assistant", icon="💬", url_path="assistant"),
        st.Page(about, title="Method & credits", icon="📄", url_path="about"),
    ]
)
pg.run()
