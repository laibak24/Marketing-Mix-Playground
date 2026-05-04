"""
2_playground.py — Budget reallocation playground with live revenue prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.optimizer import predict_revenue_from_spend, optimize_budget
from src.saturation import SATURATION_DEFAULTS
from src.adstock import CHANNELS

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Budget Playground | MMM", page_icon="🎛️", layout="wide")
st.title("🎛️ Budget Playground")
st.caption("Drag the sliders to reallocate media spend. Predicted revenue updates live.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["DATE"])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.model` first.")
    st.stop()

df        = load_data()
artifacts = load_model()

# Weekly averages as defaults
channel_defaults = {}
for ch in CHANNELS:
    col = ch
    if col in df.columns:
        channel_defaults[ch] = int(pd.to_numeric(df[col], errors="coerce").mean())
    else:
        channel_defaults[ch] = 0

total_default = sum(channel_defaults.values())

# ── Budget input ──────────────────────────────────────────────────────────────
st.subheader("Total Weekly Budget")
total_budget = st.number_input(
    "Budget ($)", min_value=10_000, max_value=2_000_000,
    value=total_default, step=5_000,
    help="Set your total weekly media budget, then allocate across channels below."
)

# ── Channel sliders ───────────────────────────────────────────────────────────
st.subheader("Allocate by Channel")
cols = st.columns(len(CHANNELS))
allocs = {}
for i, ch in enumerate(CHANNELS):
    label = ch.replace("_S", "").replace("_", " ").title()
    default_val = min(channel_defaults[ch], total_budget)
    allocs[ch] = cols[i].slider(
        label, min_value=0, max_value=total_budget,
        value=default_val, step=1_000,
        key=f"slider_{ch}",
    )

allocated = sum(allocs.values())
remaining = total_budget - allocated

if remaining < 0:
    st.error(f"⚠️ Over budget by ${abs(remaining):,}. Reduce channel spend.")
elif remaining > 0:
    st.warning(f"${remaining:,} unallocated — consider distributing all budget for best prediction.")
else:
    st.success("✅ Budget fully allocated.")

# ── Prediction ────────────────────────────────────────────────────────────────
baseline_spend = {ch: channel_defaults[ch] for ch in CHANNELS}
baseline_rev   = predict_revenue_from_spend(baseline_spend, artifacts, SATURATION_DEFAULTS)
current_rev    = predict_revenue_from_spend(allocs, artifacts, SATURATION_DEFAULTS)
delta          = current_rev - baseline_rev
delta_pct      = delta / baseline_rev * 100 if baseline_rev > 0 else 0

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Weekly Revenue", f"${current_rev:,.0f}")
m2.metric("vs Historical Baseline",   f"{delta_pct:+.1f}%", delta=f"${delta:+,.0f}")
m3.metric("Historical Baseline",      f"${baseline_rev:,.0f}")
m4.metric("Allocated Budget",         f"${allocated:,.0f}")

# ── Waterfall: current vs baseline per channel ────────────────────────────────
st.subheader("Spend Comparison: Your Allocation vs Historical Average")
labels    = [ch.replace("_S", "").title() for ch in CHANNELS]
hist_vals = [channel_defaults[ch] for ch in CHANNELS]
curr_vals = [allocs[ch] for ch in CHANNELS]

fig = go.Figure(data=[
    go.Bar(name="Historical Avg", x=labels, y=hist_vals, marker_color="#94a3b8"),
    go.Bar(name="Your Allocation", x=labels, y=curr_vals, marker_color="#2563eb"),
])
fig.update_layout(barmode="group", height=360,
                  yaxis_title="Weekly Spend ($)",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# ── Optimizer ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Find Optimal Allocation")
st.caption("Scipy optimizer finds the spend split that maximises predicted revenue for your budget.")

if st.button("Run Optimizer", type="primary"):
    with st.spinner("Optimising..."):
        result = optimize_budget(total_budget, artifacts, CHANNELS, SATURATION_DEFAULTS)

    opt_rev = result.pop("_predicted_revenue")
    success  = result.pop("_success")
    uplift   = (opt_rev - baseline_rev) / baseline_rev * 100

    st.success(f"Optimal predicted revenue: **${opt_rev:,.0f}** ({uplift:+.1f}% vs baseline)")

    opt_cols = st.columns(len(CHANNELS))
    for i, ch in enumerate(CHANNELS):
        label = ch.replace("_S", "").replace("_", " ").title()
        current_alloc = allocs[ch]
        optimal_alloc = result[ch]
        diff = optimal_alloc - current_alloc
        opt_cols[i].metric(
            label,
            f"${optimal_alloc:,.0f}",
            delta=f"${diff:+,.0f} vs yours",
        )