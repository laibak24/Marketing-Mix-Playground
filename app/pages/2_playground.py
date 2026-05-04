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

st.set_page_config(page_title="Budget Playground · MarketLytics", page_icon="🎛️", layout="wide")

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }
[data-testid="stSidebar"] { background: #FAFAF9; border-right: 1px solid #E8E6E1; }
.sidebar-brand { display:flex;align-items:center;gap:10px;margin-bottom:2rem; }
.sidebar-brand-icon { width:32px;height:32px;background:#1a1a1a;border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:600; }
.sidebar-brand-text { font-size:15px;font-weight:600;color:#1a1a1a;letter-spacing:-0.3px; }
.sidebar-sub { font-size:11px;color:#9B9B9B;margin-top:1px; }
.nav-label { font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#9B9B9B;margin:1.5rem 0 0.5rem 0; }
.page-header { border-bottom:1px solid #E8E6E1;padding-bottom:1.5rem;margin-bottom:2rem; }
.page-eyebrow { font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#9B9B9B;margin-bottom:6px; }
.page-title { font-size:28px;font-weight:600;color:#1a1a1a;letter-spacing:-0.5px;margin:0; }
.page-desc { font-size:14px;color:#6B6B6B;margin-top:6px;line-height:1.5; }
.section-title { font-size:13px;font-weight:600;color:#1a1a1a;letter-spacing:-0.2px;margin:0 0 0.35rem 0; }
.section-desc { font-size:12px;color:#9B9B9B;margin-bottom:1rem; }
.panel { background:#fff;border:1px solid #E8E6E1;border-radius:10px;padding:1.5rem; }
[data-testid="stMetric"] { background:#FAFAF9;border:1px solid #E8E6E1;border-radius:10px;padding:1rem 1.25rem; }
[data-testid="stMetricLabel"] { font-size:11px !important;color:#9B9B9B !important;font-weight:500 !important;text-transform:uppercase;letter-spacing:0.04em; }
[data-testid="stMetricValue"] { font-size:22px !important;font-weight:600 !important;color:#1a1a1a !important;letter-spacing:-0.4px !important; }
[data-testid="stMetricDelta"] { font-size:12px !important; }
.stButton > button { background:#1a1a1a !important;color:#fff !important;border:none !important;border-radius:8px !important;font-family:'DM Sans',sans-serif !important;font-size:13px !important;font-weight:500 !important;padding:0.5rem 1.25rem !important; }
.stButton > button:hover { opacity:0.8 !important; }
[data-testid="stSlider"] > div > div > div > div { background:#1a1a1a !important; }
hr { border:none;border-top:1px solid #E8E6E1;margin:2rem 0; }
.stCaption { font-size:11px !important;color:#9B9B9B !important; }
[data-testid="stSidebarNav"] a { font-size:13px !important;color:#3D3D3D !important;font-weight:400 !important;padding:0.4rem 0.75rem !important;border-radius:6px !important; }
[data-testid="stSidebarNav"] [aria-current="page"] { background:#ECEAE4 !important;font-weight:500 !important;color:#1a1a1a !important; }
.channel-card { background:#FAFAF9;border:1px solid #E8E6E1;border-radius:10px;padding:1.25rem;margin-bottom:0.75rem; }
.channel-name { font-size:12px;font-weight:600;color:#1a1a1a;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.75rem; }
.budget-bar-bg { background:#E8E6E1;border-radius:4px;height:4px;margin:0.5rem 0; }
.budget-bar-fill { background:#1a1a1a;border-radius:4px;height:4px; }
.result-banner { background:#F5F4F0;border:1px solid #E8E6E1;border-radius:10px;
  padding:1.25rem 1.5rem;display:flex;align-items:center;justify-content:space-between; }
.result-banner-label { font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:#9B9B9B; }
.result-banner-value { font-size:26px;font-weight:600;color:#1a1a1a;letter-spacing:-0.5px; }
.result-banner-delta.pos { color:#16A34A;font-size:13px;font-weight:500; }
.result-banner-delta.neg { color:#DC2626;font-size:13px;font-weight:500; }
[data-testid="stNumberInput"] input { font-family:'DM Mono',monospace !important;font-size:14px !important; }
</style>
"""
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">ML</div>
  <div><div class="sidebar-brand-text">MarketLytics</div><div class="sidebar-sub">MMM Platform</div></div>
</div>
<div class="nav-label">Analytics</div>
""", unsafe_allow_html=True)

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

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Scenario Planning</div>
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
st.markdown('<div class="section-desc">Drag each slider to set weekly spend per channel. The bar shows your allocation vs historical average.</div>', unsafe_allow_html=True)

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
        color        = "#16A34A" if vs_hist > 0 else ("#DC2626" if vs_hist < 0 else "#9B9B9B")
        st.markdown(
            f'<div style="font-size:11px;color:#9B9B9B;">'
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
sb2.metric("Remaining",  f"${remaining:,.0f}", delta=f"{remaining/total_budget*100:.1f}% unallocated" if total_budget > 0 else None)
sb3.metric("Channels Active", f"{sum(1 for v in allocs.values() if v > 0)} / {len(CHANNELS)}")

if remaining < 0:
    st.error(f"Over budget by ${abs(remaining):,}. Reduce one or more channels.")
elif remaining > total_budget * 0.05:
    st.warning(f"${remaining:,} unallocated. Distribute remaining budget for a more accurate prediction.")

# ── Live prediction ───────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Revenue Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Model output based on your current allocation — updates as you move sliders</div>', unsafe_allow_html=True)

baseline_spend = {ch: channel_defaults[ch] for ch in CHANNELS}
baseline_rev   = predict_revenue_from_spend(baseline_spend, artifacts, SATURATION_DEFAULTS)
current_rev    = predict_revenue_from_spend(allocs, artifacts, SATURATION_DEFAULTS)
delta          = current_rev - baseline_rev
delta_pct      = delta / baseline_rev * 100 if baseline_rev > 0 else 0
sign_cls       = "pos" if delta >= 0 else "neg"
sign_sym       = "▲" if delta >= 0 else "▼"

st.markdown(f"""
<div class="result-banner">
  <div>
    <div class="result-banner-label">Predicted Weekly Revenue</div>
    <div class="result-banner-value">${current_rev:,.0f}</div>
    <div class="result-banner-delta {sign_cls}">{sign_sym} {abs(delta_pct):.1f}% vs historical baseline (${baseline_rev:,.0f})</div>
  </div>
  <div style="text-align:right;">
    <div class="result-banner-label">Revenue Delta</div>
    <div class="result-banner-value" style="font-size:20px;color:{'#16A34A' if delta>=0 else '#DC2626'}">${delta:+,.0f}</div>
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
                     marker_color="#E8E6E1",
                     hovertemplate="<b>%{x}</b><br>Hist: $%{y:,.0f}<extra></extra>"))
fig.add_trace(go.Bar(name="Your Allocation", x=labels, y=curr_vals,
                     marker_color="#1a1a1a",
                     hovertemplate="<b>%{x}</b><br>Yours: $%{y:,.0f}<extra></extra>"))
fig.update_layout(
    barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
               tickfont=dict(size=11, color="#9B9B9B")),
    xaxis=dict(tickfont=dict(size=12, color="#3D3D3D")),
    legend=dict(orientation="h", y=1.1, font=dict(size=12)),
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
        result   = optimize_budget(total_budget, artifacts, CHANNELS, SATURATION_DEFAULTS)
    opt_rev  = result.pop("_predicted_revenue")
    success  = result.pop("_success")
    uplift   = (opt_rev - baseline_rev) / baseline_rev * 100

    opt_delta = opt_rev - current_rev
    st.markdown(f"""
    <div class="result-banner" style="margin-bottom:1.5rem;">
      <div>
        <div class="result-banner-label">Optimal Predicted Revenue</div>
        <div class="result-banner-value">${opt_rev:,.0f}</div>
        <div class="result-banner-delta pos">▲ ${opt_delta:+,.0f} vs your current allocation &nbsp;·&nbsp; {uplift:+.1f}% vs baseline</div>
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
        st.session_state[f"sl_{ch}_opt"] = optimal_alloc

    if not success:
        st.caption("⚠ Optimiser converged with warnings — result is a best estimate.")