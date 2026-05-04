"""
1_overview.py — Channel ROI and revenue attribution overview page.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.attribution import channel_contributions, channel_roi
from src.features import build_features

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Overview | MMM", page_icon="📈", layout="wide")
st.title("📈 Channel Overview")

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

# ── KPI row ──────────────────────────────────────────────────────────────────
contribs = channel_contributions(artifacts, df)
roi_df   = channel_roi(df, contribs)

total_rev   = df["revenue"].sum()
total_spend = roi_df["total_spend"].sum()
avg_roi     = total_rev / total_spend if total_spend > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",    f"${total_rev/1e6:.1f}M")
c2.metric("Total Media Spend", f"${total_spend/1e6:.1f}M")
c3.metric("Blended ROI",      f"{avg_roi:.2f}x")
c4.metric("Weeks of Data",    len(df))

st.divider()

# ── ROI bar chart ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROI by Channel")
    fig_roi = px.bar(
        roi_df.sort_values("roi"),
        x="roi", y="channel", orientation="h",
        color="roi",
        color_continuous_scale="Blues",
        labels={"roi": "ROI (revenue / spend)", "channel": ""},
        text=roi_df.sort_values("roi")["roi"].apply(lambda x: f"{x:.2f}x"),
    )
    fig_roi.update_traces(textposition="outside")
    fig_roi.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig_roi, use_container_width=True)

with col2:
    st.subheader("Revenue Contribution %")
    fig_pie = px.pie(
        contribs, values="contribution_pct", names="channel",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Spend vs Revenue over time ────────────────────────────────────────────────
st.subheader("Revenue vs Total Media Spend Over Time")
CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"]
df["total_spend_week"] = df[CHANNELS].apply(pd.to_numeric, errors="coerce").sum(axis=1)

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df["DATE"], y=df["revenue"],
    name="Revenue", line=dict(color="#2563eb", width=2)
))
fig_ts.add_trace(go.Bar(
    x=df["DATE"], y=df["total_spend_week"],
    name="Total Spend", marker_color="rgba(148,163,184,0.5)", yaxis="y2"
))
fig_ts.update_layout(
    yaxis=dict(title="Revenue ($)"),
    yaxis2=dict(title="Spend ($)", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.1),
    height=380,
)
st.plotly_chart(fig_ts, use_container_width=True)

# ── Attribution table ─────────────────────────────────────────────────────────
st.subheader("Channel Attribution Summary")
display = roi_df.copy()
display["total_spend"]        = display["total_spend"].apply(lambda x: f"${x:,.0f}")
display["attributed_revenue"] = display["attributed_revenue"].apply(lambda x: f"${x:,.0f}")
display["roi"]                = display["roi"].apply(lambda x: f"{x:.2f}x")
display["contribution_pct"]   = display["contribution_pct"].apply(lambda x: f"{x:.1f}%")
st.dataframe(display, use_container_width=True, hide_index=True)