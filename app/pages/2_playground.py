"""
2_playground.py — Budget reallocation playground.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import predict_revenue_from_spend, optimize_budget
from src.saturation import SATURATION_DEFAULTS
from src.adstock import CHANNELS

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Playground · MarketLytics", page_icon="🎛️", layout="wide",
                   initial_sidebar_state="collapsed")

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:#F7F6F2; --surface:#FFFFFF; --border:#E2E0D9; --border-strong:#C8C5BC;
  --ink:#141414; --ink-mid:#4A4A4A; --ink-muted:#8C8C8C;
  --green:#15803D; --green-bg:#F0FDF4; --green-border:#BBF7D0;
  --red:#DC2626; --red-bg:#FEF2F2; --amber:#B45309; --amber-bg:#FFFBEB;
  --radius:10px; --font-sans:'DM Sans',sans-serif;
  --font-display:'Syne',sans-serif; --font-mono:'DM Mono',monospace;
  color-scheme: light !important;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main {
  font-family: var(--font-sans);
  background: var(--bg) !important;
  color: var(--ink);
}

.block-container { background: transparent !important; }
.stMarkdown, .stMarkdown p, .stMarkdown div { color: var(--ink) !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.block-container {
  padding: 0 !important;
  max-width: 100% !important;
  margin-top: 0 !important;
}
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── Navbar ── */
.navbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #FFFFFF;
  border-bottom: 1px solid #E2E0D9;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  width: 100%;
}
.navbar-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 3rem;
  height: 56px;
  max-width: 1400px;
  margin: 0 auto;
}
.navbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.navbar-logo {
  width: 30px; height: 30px;
  background: #141414;
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  color: white;
  font-family: var(--font-display);
  font-size: 12px; font-weight: 700; letter-spacing: -0.3px;
}
.navbar-name {
  font-family: var(--font-display);
  font-size: 15px; font-weight: 700;
  color: #141414; letter-spacing: -0.3px;
}
.navbar-links {
  display: flex;
  align-items: center;
  gap: 4px;
}
.nav-link {
  font-size: 13px;
  font-weight: 500;
  color: #4A4A4A;
  padding: 6px 14px;
  border-radius: 6px;
  text-decoration: none !important;
  transition: background 0.15s, color 0.15s;
  display: inline-block;
  cursor: pointer;
}
.nav-link:hover { background: #F7F6F2; color: #141414; text-decoration: none !important; }
.nav-link.active { background: #141414; color: #FFFFFF !important; text-decoration: none !important; }

/* ── Page layout ── */
.page-wrap { max-width: 1100px; margin: 0 auto; padding: 2.5rem 3rem 5rem; }
.page-header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
.page-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 8px; }
.page-title { font-family: var(--font-display); font-size: 36px; font-weight: 800; color: var(--ink); letter-spacing: -0.8px; margin: 0 0 8px 0; }
.page-desc { font-size: 14px; color: var(--ink-mid); margin-top: 4px; line-height: 1.6; max-width: 600px; }
.section-title { font-family: var(--font-display); font-size: 14px; font-weight: 700; color: var(--ink); letter-spacing: -0.2px; margin: 0 0 0.3rem 0; }
.section-desc { font-size: 12px; color: var(--ink-muted); margin-bottom: 1rem; }

/* ── Components ── */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.25rem; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: var(--ink-muted) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; font-size: 22px !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -0.4px !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }
.stButton > button { background: var(--ink) !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: var(--font-sans) !important; font-size: 13px !important; font-weight: 500 !important; padding: 0.5rem 1.25rem !important; }
.stButton > button:hover { opacity: 0.8 !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--ink) !important; }
[data-testid="stNumberInput"] input { font-family: var(--font-mono) !important; font-size: 14px !important; }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.stCaption { font-size: 11px !important; color: var(--ink-muted) !important; }
.stAlert { border-radius: 8px !important; font-size: 13px !important; }

/* ── Slider labels & values ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stSlider"] .st-emotion-cache-label,
div[class*="stSlider"] > label,
div[class*="stSlider"] > label > div,
div[class*="stSlider"] > label > div > p {
  color: var(--ink) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
/* Slider min/max tick labels */
[data-testid="stSlider"] [data-testid="stTickBar"] span,
[data-testid="stSlider"] span {
  color: var(--ink-muted) !important;
}
/* Slider current value bubble */
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] output {
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* ── Number input label ── */
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label p {
  color: var(--ink) !important;
  font-weight: 600 !important;
}

/* ── Warning / info / error / success alert boxes ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-size: 13px !important;
}
/* Warning (yellow) */
[data-testid="stAlert"][data-baseweb="notification"] div[class*="body"],
.stAlert > div,
[data-testid="stNotification"] p,
[data-testid="stNotification"] div {
  color: var(--ink) !important;
}
/* Force all alert/notification text dark */
div[data-baseweb="notification"] *,
div[role="alert"] *,
.stAlert * {
  color: var(--ink) !important;
}
/* Warning box specifically */
.stWarning, .stWarning * { color: var(--amber) !important; }
.stError,   .stError *   { color: var(--red)   !important; }
.stSuccess, .stSuccess * { color: var(--green)  !important; }
.stInfo,    .stInfo *    { color: #1e40af       !important; }

/* ── Section titles rendered via st.markdown ── */
.section-title { color: var(--ink) !important; }
.section-desc  { color: var(--ink-muted) !important; }
.page-title    { color: var(--ink) !important; }
.page-desc     { color: var(--ink-mid) !important; }
.page-eyebrow  { color: var(--ink-muted) !important; }

/* ── Expander label ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
  color: var(--ink) !important;
  font-weight: 600 !important;
}

/* ── Revenue banner base ───────────────────────────── */
.rev-banner {
  background: #fff7f7;
  border-radius: var(--radius);
  padding: 2rem 2.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;

  color: white !important; /* force base text color */
}

/* Ensure all children are white by default */
.rev-banner * {
  color: white !important;
}

/* ── Left (main) ───────────────────────────────────── */
.rev-banner-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.55;
  margin-bottom: 6px;
}

.rev-banner-value {
  font-family: var(--font-display);
  font-size: 44px;
  font-weight: 800;
  letter-spacing: -1.5px;
  line-height: 1;
  margin-bottom: 8px;
}

.rev-banner-delta {
  font-size: 14px;
  font-weight: 500;
}

/* ── Right (side) ─────────────────────────────────── */
.rev-banner-side {
  text-align: right;
}

.rev-banner-side-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.55;
  margin-bottom: 6px;
}

.rev-banner-side-value {
  font-family: var(--font-display);
  font-size: 32px;
  font-weight: 700;
  letter-spacing: -1px;
  line-height: 1;
}

/* ── Status colors (must override white) ───────────── */
.rev-banner-delta.pos,
.rev-banner-side-value.pos {
  color: #4ADE80 !important;
}

.rev-banner-delta.neg,
.rev-banner-side-value.neg {
  color: #F87171 !important;
}
/* ── Optimizer result banner ── */
.opt-banner {
  background: var(--green-bg); border: 1px solid var(--green-border);
  border-radius: var(--radius); padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.opt-banner-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--green) !important; margin-bottom: 6px;
}
.opt-banner-value {
  font-family: var(--font-display); font-size: 32px; font-weight: 800;
  color: var(--ink) !important; letter-spacing: -0.8px; margin-bottom: 4px;
}
.opt-banner-sub { font-size: 13px; color: var(--green) !important; font-weight: 500; }

/* ── Scoped dark text: Streamlit native widgets only, not custom HTML ── */
[data-testid="stMain"] .stMarkdown > div > p,
[data-testid="stMain"] .stText p {
  color: var(--ink) !important;
}
            
</style>
""", unsafe_allow_html=True)

# ── Navbar (pure HTML — no st.page_link) ──────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-inner">
    <div class="navbar-brand">
      <div class="navbar-logo">ML</div>
      <div class="navbar-name">MarketLytics</div>
    </div>
    <div class="navbar-links">
      <a class="nav-link" href="/main">Home</a>
      <a class="nav-link" href="/overview">Overview</a>
      <a class="nav-link active" href="/playground">Playground</a>
      <a class="nav-link" href="/diagnostics">Diagnostics</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
CH_LABELS = {
    "tv_S": "TV", "ooh_S": "Out-of-Home", "print_S": "Print",
    "facebook_S": "Facebook", "search_S": "Paid Search", "newsletter": "Newsletter"
}
CH_ICONS = {
    "tv_S": "📺", "ooh_S": "🏙️", "print_S": "📰",
    "facebook_S": "📘", "search_S": "🔍", "newsletter": "✉️"
}

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["DATE"])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not trained. Run `python -m src.model` first.")
    st.stop()

df        = load_data()
artifacts = load_model()

channel_defaults = {}
for ch in CHANNELS:
    if ch in df.columns:
        channel_defaults[ch] = int(pd.to_numeric(df[ch], errors="coerce").mean())
    else:
        channel_defaults[ch] = 0

total_default = sum(channel_defaults.values())

# ── Page body ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Step 02 · Forward-looking</div>
  <div class="page-title">Budget Playground</div>
  <div class="page-desc">Adjust weekly spend across channels and instantly see the predicted impact on revenue. Use the optimiser to find the best allocation for your budget.</div>
</div>
""", unsafe_allow_html=True)

# ── Total budget ──────────────────────────────────────────────────────────────
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Total Weekly Budget</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Set the total envelope, then distribute across channels below</div>', unsafe_allow_html=True)
    total_budget = st.number_input(
        "", min_value=10_000, max_value=5_000_000,
        value=total_default, step=5_000,
        label_visibility="collapsed",
    )

with right:
    budget_change = total_budget - total_default
    pct_change    = budget_change / total_default * 100 if total_default > 0 else 0
    st.metric(
        "vs Historical Weekly Avg",
        f"${total_budget:,.0f}",
        delta=f"{pct_change:+.1f}%",
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Channel allocation ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Channel Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Drag each slider to set weekly spend per channel. Numbers show allocation vs historical average.</div>', unsafe_allow_html=True)

allocs = {}
cols_top = st.columns(3)
cols_bot = st.columns(3)
all_cols = cols_top + cols_bot

for i, ch in enumerate(CHANNELS):
    label   = CH_LABELS.get(ch, ch)
    icon    = CH_ICONS.get(ch, "•")
    default = min(channel_defaults[ch], total_budget)
    hist    = channel_defaults[ch]

    with all_cols[i]:
        allocs[ch] = st.slider(
            f"{icon} {label}",
            min_value=0, max_value=int(total_budget),
            value=default, step=500,
            key=f"sl_{ch}",
            help=f"Historical weekly avg: ${hist:,}",
        )
        pct_of_total = allocs[ch] / total_budget * 100 if total_budget > 0 else 0
        vs_hist      = allocs[ch] - hist
        sign         = "▲" if vs_hist > 0 else ("▼" if vs_hist < 0 else "—")
        color        = "var(--green)" if vs_hist > 0 else ("var(--red)" if vs_hist < 0 else "var(--ink-muted)")
        st.markdown(
            f'<div style="font-size:11px;color:var(--ink-muted);">'
            f'{pct_of_total:.1f}% of budget &nbsp;·&nbsp; '
            f'<span style="color:{color}">{sign} ${abs(vs_hist):,} vs avg</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Budget status ─────────────────────────────────────────────────────────────
allocated = sum(allocs.values())
remaining = total_budget - allocated

st.markdown("<hr>", unsafe_allow_html=True)
sb1, sb2, sb3 = st.columns(3)
sb1.metric("Allocated", f"${allocated:,.0f}")
sb2.metric("Remaining",  f"${remaining:,.0f}",
           delta=f"{remaining/total_budget*100:.1f}% unallocated" if total_budget > 0 else None)
sb3.metric("Channels Active", f"{sum(1 for v in allocs.values() if v > 0)} / {len(CHANNELS)}")

if remaining < 0:
    st.error(f"Over budget by ${abs(remaining):,}. Reduce one or more channels.")
elif remaining > total_budget * 0.05:
    st.warning(f"${remaining:,} unallocated. Distribute remaining budget for a more accurate prediction.")

# ── Revenue prediction ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Revenue Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Model output based on your current allocation — updates live as you move sliders</div>', unsafe_allow_html=True)

baseline_spend = {ch: channel_defaults[ch] for ch in CHANNELS}
baseline_rev   = predict_revenue_from_spend(baseline_spend, artifacts, SATURATION_DEFAULTS)
current_rev    = predict_revenue_from_spend(allocs, artifacts, SATURATION_DEFAULTS)
delta          = current_rev - baseline_rev
delta_pct      = delta / baseline_rev * 100 if baseline_rev > 0 else 0
sign_cls       = "pos" if delta >= 0 else "neg"
sign_sym       = "▲" if delta >= 0 else "▼"
delta_sign     = "+" if delta >= 0 else ""

st.markdown(f"""
<div class="rev-banner">
  <div class="rev-banner-main">
    <div class="rev-banner-label">Predicted Weekly Revenue</div>
    <div class="rev-banner-value">${current_rev:,.0f}</div>
    <div class="rev-banner-delta {sign_cls}">
      {sign_sym} {abs(delta_pct):.1f}% vs historical baseline (${baseline_rev:,.0f})
    </div>
  </div>
  <div class="rev-banner-side">
    <div class="rev-banner-side-label">Revenue Delta</div>
    <div class="rev-banner-side-value {sign_cls}">{delta_sign}${delta:,.0f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Comparison chart ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
labels    = [CH_LABELS.get(ch, ch) for ch in CHANNELS]
hist_vals = [channel_defaults[ch] for ch in CHANNELS]
curr_vals = [allocs[ch] for ch in CHANNELS]

fig = go.Figure()
fig.add_trace(go.Bar(name="Historical Avg", x=labels, y=hist_vals,
                     marker_color="#E2E0D9",
                     hovertemplate="<b>%{x}</b><br>Hist: $%{y:,.0f}<extra></extra>"))
fig.add_trace(go.Bar(name="Your Allocation", x=labels, y=curr_vals,
                     marker_color="#141414",
                     hovertemplate="<b>%{x}</b><br>Yours: $%{y:,.0f}<extra></extra>"))
fig.update_layout(
    barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
               tickfont=dict(size=11, color="#8C8C8C")),
    xaxis=dict(tickfont=dict(size=12, color="#4A4A4A")),
    legend=dict(orientation="h", y=1.1, font=dict(size=12, color="#141414")),
    bargap=0.25, bargroupgap=0.08,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Optimizer ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
left2, right2 = st.columns([2, 1], gap="large")

with left2:
    st.markdown('<div class="section-title">Optimal Allocation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Let the model find the spend split that maximises predicted revenue for your budget envelope</div>', unsafe_allow_html=True)

with right2:
    run_opt = st.button("Find Optimal Allocation →", type="primary")

if run_opt:
    with st.spinner("Running optimisation..."):
        result  = optimize_budget(total_budget, artifacts, CHANNELS, SATURATION_DEFAULTS)
    opt_rev = result.pop("_predicted_revenue")
    success = result.pop("_success")
    uplift  = (opt_rev - baseline_rev) / baseline_rev * 100
    opt_delta = opt_rev - current_rev

    st.markdown(f"""
    <div class="opt-banner">
      <div class="opt-banner-label">Optimal Predicted Revenue</div>
      <div class="opt-banner-value">${opt_rev:,.0f}</div>
      <div class="opt-banner-sub">
        +${opt_delta:,.0f} vs your current allocation &nbsp;·&nbsp; {uplift:+.1f}% vs historical baseline
      </div>
    </div>
    """, unsafe_allow_html=True)

    opt_cols = st.columns(len(CHANNELS))
    for i, ch in enumerate(CHANNELS):
        label         = CH_LABELS.get(ch, ch)
        current_alloc = allocs[ch]
        optimal_alloc = result[ch]
        diff          = optimal_alloc - current_alloc
        opt_cols[i].metric(label, f"${optimal_alloc:,.0f}", delta=f"${diff:+,.0f}")

    if not success:
        st.caption("⚠ Optimiser converged with warnings — result is a best estimate.")


st.markdown('</div>', unsafe_allow_html=True)