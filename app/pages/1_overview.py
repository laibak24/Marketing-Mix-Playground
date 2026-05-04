"""
1_overview.py — Channel ROI and revenue attribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.attribution import channel_contributions, channel_roi
from src.features import build_features

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Overview · MarketLytics", page_icon="📈", layout="wide")

# ── Shared styles (identical across all pages) ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }
[data-testid="stSidebar"] { background: #FAFAF9; border-right: 1px solid #E8E6E1; }
.sidebar-brand { display:flex;align-items:center;gap:10px;margin-bottom:2rem; }
.sidebar-brand-icon { width:32px;height:32px;background:#1a1a1a;border-radius:8px;
  display:flex;align-items:center;justify-content:center;color:white;font-size:14px;font-weight:600; }
.sidebar-brand-text { font-size:15px;font-weight:600;color:#1a1a1a;letter-spacing:-0.3px; }
.sidebar-sub { font-size:11px;color:#9B9B9B;margin-top:1px; }
.nav-label { font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
  color:#9B9B9B;margin:1.5rem 0 0.5rem 0; }
.page-header { border-bottom:1px solid #E8E6E1;padding-bottom:1.5rem;margin-bottom:2rem; }
.page-eyebrow { font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
  color:#9B9B9B;margin-bottom:6px; }
.page-title { font-size:28px;font-weight:600;color:#1a1a1a;letter-spacing:-0.5px;margin:0; }
.page-desc { font-size:14px;color:#6B6B6B;margin-top:6px;line-height:1.5; }
.section-title { font-size:13px;font-weight:600;color:#1a1a1a;letter-spacing:-0.2px;margin:0 0 0.35rem 0; }
.section-desc { font-size:12px;color:#9B9B9B;margin-bottom:1rem; }
.panel { background:#fff;border:1px solid #E8E6E1;border-radius:10px;padding:1.5rem; }
[data-testid="stMetric"] { background:#FAFAF9;border:1px solid #E8E6E1;border-radius:10px;padding:1rem 1.25rem; }
[data-testid="stMetricLabel"] { font-size:11px !important;color:#9B9B9B !important;font-weight:500 !important;text-transform:uppercase;letter-spacing:0.04em; }
[data-testid="stMetricValue"] { font-size:22px !important;font-weight:600 !important;color:#1a1a1a !important;letter-spacing:-0.4px !important; }
.stButton > button { background:#1a1a1a !important;color:#fff !important;border:none !important;
  border-radius:8px !important;font-family:'DM Sans',sans-serif !important;font-size:13px !important;
  font-weight:500 !important;padding:0.5rem 1.25rem !important; }
hr { border:none;border-top:1px solid #E8E6E1;margin:2rem 0; }
[data-testid="stDataFrame"] { border:1px solid #E8E6E1;border-radius:10px;overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:0;border-bottom:1px solid #E8E6E1;background:transparent; }
.stTabs [data-baseweb="tab"] { font-size:13px;font-weight:500;color:#6B6B6B;
  padding:0.5rem 1rem;border-bottom:2px solid transparent;background:transparent; }
.stTabs [aria-selected="true"] { color:#1a1a1a !important;border-bottom:2px solid #1a1a1a !important; }
.stCaption { font-size:11px !important;color:#9B9B9B !important; }
[data-testid="stSidebarNav"] a { font-size:13px !important;color:#3D3D3D !important;
  font-weight:400 !important;padding:0.4rem 0.75rem !important;border-radius:6px !important; }
[data-testid="stSidebarNav"] [aria-current="page"] { background:#ECEAE4 !important;
  font-weight:500 !important;color:#1a1a1a !important; }
.insight-pill { display:inline-block;background:#F5F4F0;border-radius:6px;
  padding:0.3rem 0.75rem;font-size:12px;color:#3D3D3D;margin:0.25rem 0.25rem 0.25rem 0; }
.insight-pill.green { background:#F0FDF4;color:#16A34A; }
.insight-pill.amber { background:#FFFBEB;color:#D97706; }
.insight-pill.red { background:#FEF2F2;color:#DC2626; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">ML</div>
  <div><div class="sidebar-brand-text">MarketLytics</div><div class="sidebar-sub">MMM Platform</div></div>
</div>
<div class="nav-label">Analytics</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["DATE"])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not trained yet. Run `python -m src.model` in your terminal.")
    st.stop()

df        = load_data()
artifacts = load_model()
contribs  = channel_contributions(artifacts, df)
roi_df    = channel_roi(df, contribs)

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"]
CH_LABELS = {"tv_S": "TV", "ooh_S": "Out-of-Home", "print_S": "Print",
             "facebook_S": "Facebook", "search_S": "Search", "newsletter": "Newsletter"}

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Channel Performance</div>
  <div class="page-title">Channel Overview</div>
  <div class="page-desc">Revenue attribution and ROI across all media channels — based on 208 weeks of spend data.</div>
</div>
""", unsafe_allow_html=True)

# ── Date filter ───────────────────────────────────────────────────────────────
with st.expander("Filter date range", expanded=False):
    min_d, max_d = df["DATE"].min().date(), df["DATE"].max().date()
    d1, d2 = st.date_input("Select range", [min_d, max_d], min_value=min_d, max_value=max_d)
    df = df[(df["DATE"].dt.date >= d1) & (df["DATE"].dt.date <= d2)].copy()
    contribs = channel_contributions(artifacts, df)
    roi_df   = channel_roi(df, contribs)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_rev   = df["revenue"].sum()
total_spend = roi_df["total_spend"].sum()
avg_roi     = total_rev / total_spend if total_spend > 0 else 0
best_ch     = roi_df.loc[roi_df["roi"].idxmax(), "channel"] if not roi_df.empty else "—"
best_ch_label = CH_LABELS.get(best_ch, best_ch)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",     f"${total_rev/1e6:.2f}M")
c2.metric("Total Media Spend", f"${total_spend/1e6:.2f}M")
c3.metric("Blended ROI",       f"{avg_roi:.2f}×")
c4.metric("Top Channel",       best_ch_label)

st.markdown("<hr>", unsafe_allow_html=True)

# ── ROI + Contribution ────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<div class="section-title">Return on Investment by Channel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Revenue generated per dollar of media spend</div>', unsafe_allow_html=True)

    roi_plot = roi_df.copy()
    roi_plot["label"] = roi_plot["channel"].map(CH_LABELS).fillna(roi_plot["channel"])
    roi_plot = roi_plot.sort_values("roi", ascending=True)

    fig_roi = go.Figure()
    colors  = ["#E8E6E1"] * len(roi_plot)
    colors[-1] = "#1a1a1a"   # highlight top channel
    fig_roi.add_trace(go.Bar(
        x=roi_plot["roi"], y=roi_plot["label"], orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}×" for v in roi_plot["roi"]],
        textposition="outside",
        textfont=dict(size=12, color="#1a1a1a", family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>ROI: %{x:.2f}×<extra></extra>",
    ))
    fig_roi.update_layout(
        height=280, margin=dict(l=0, r=60, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0EEE9", zeroline=False,
                   tickfont=dict(size=11, color="#9B9B9B"), showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#3D3D3D")),
        bargap=0.35,
    )
    st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar": False})

with col2:
    st.markdown('<div class="section-title">Revenue Contribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Share of attributed revenue per channel</div>', unsafe_allow_html=True)

    contribs_plot = contribs.copy()
    contribs_plot["label"] = contribs_plot["channel"].map(CH_LABELS).fillna(contribs_plot["channel"])

    fig_pie = go.Figure(go.Pie(
        labels=contribs_plot["label"],
        values=contribs_plot["contribution_pct"],
        hole=0.6,
        marker=dict(colors=["#1a1a1a","#3D3D3D","#6B6B6B","#9B9B9B","#C4C2BC","#E8E6E1"],
                    line=dict(color="white", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig_pie.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white",
        legend=dict(orientation="v", font=dict(size=11, color="#3D3D3D"),
                    itemsizing="constant", x=0.75, y=0.5),
        annotations=[dict(text=f"<b>{contribs_plot.loc[contribs_plot['contribution_pct'].idxmax(),'label']}</b><br>leads",
                         x=0.38, y=0.5, font_size=11, showarrow=False, font_color="#6B6B6B")]
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

# ── Time series ───────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Revenue & Spend Over Time</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Weekly revenue trend alongside total media investment</div>', unsafe_allow_html=True)

df["total_spend_week"] = df[CHANNELS].apply(pd.to_numeric, errors="coerce").sum(axis=1)

tab1, tab2 = st.tabs(["Revenue vs Spend", "Channel Spend Breakdown"])

with tab1:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df["DATE"], y=df["revenue"], name="Revenue",
        line=dict(color="#1a1a1a", width=2),
        hovertemplate="<b>Revenue</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_ts.add_trace(go.Bar(
        x=df["DATE"], y=df["total_spend_week"], name="Media Spend",
        marker_color="#E8E6E1", yaxis="y2",
        hovertemplate="<b>Spend</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_ts.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Revenue ($)", gridcolor="#F0EEE9", tickformat="$,.0f",
                   tickfont=dict(size=11, color="#9B9B9B")),
        yaxis2=dict(title="Spend ($)", overlaying="y", side="right",
                    tickfont=dict(size=11, color="#9B9B9B"), tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.08, font=dict(size=12)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

with tab2:
    melted = df[["DATE"] + CHANNELS].melt(id_vars="DATE", var_name="channel", value_name="spend")
    melted["channel"] = melted["channel"].map(CH_LABELS).fillna(melted["channel"])
    melted["spend"]   = pd.to_numeric(melted["spend"], errors="coerce").fillna(0)

    fig_area = go.Figure()
    ch_colors = ["#1a1a1a","#3D3D3D","#6B6B6B","#9B9B9B","#C4C2BC","#E8E6E1"]
    for i, (ch_raw, ch_label) in enumerate(CH_LABELS.items()):
        ch_data = melted[melted["channel"] == ch_label]
        fig_area.add_trace(go.Bar(
            x=ch_data["DATE"], y=ch_data["spend"],
            name=ch_label, marker_color=ch_colors[i % len(ch_colors)],
        ))
    fig_area.update_layout(
        barmode="stack", height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                   tickfont=dict(size=11, color="#9B9B9B")),
        legend=dict(orientation="h", y=1.08, font=dict(size=12)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})

# ── Attribution table ─────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Attribution Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Full breakdown of spend, attributed revenue, and efficiency per channel</div>', unsafe_allow_html=True)

display = roi_df.copy()
display["channel"] = display["channel"].map(CH_LABELS).fillna(display["channel"])
display = display.rename(columns={
    "channel": "Channel", "total_spend": "Total Spend",
    "attributed_revenue": "Attributed Revenue",
    "roi": "ROI", "contribution_pct": "Contribution %",
})
display["Total Spend"]        = display["Total Spend"].apply(lambda x: f"${x:,.0f}")
display["Attributed Revenue"] = display["Attributed Revenue"].apply(lambda x: f"${x:,.0f}")
display["ROI"]                = display["ROI"].apply(lambda x: f"{x:.2f}×")
display["Contribution %"]     = display["Contribution %"].apply(lambda x: f"{x:.1f}%")
st.dataframe(display, use_container_width=True, hide_index=True)

# ── Key insights ──────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

top_roi   = roi_df.loc[roi_df["roi"].idxmax()]
low_roi   = roi_df.loc[roi_df["roi"].idxmin()]
top_label = CH_LABELS.get(top_roi["channel"], top_roi["channel"])
low_label = CH_LABELS.get(low_roi["channel"], low_roi["channel"])

st.markdown(f"""
<span class="insight-pill green">✓ {top_label} delivers the highest ROI at {top_roi['roi']:.2f}×</span>
<span class="insight-pill amber">△ {low_label} has the lowest ROI at {low_roi['roi']:.2f}× — consider reallocation</span>
<span class="insight-pill">◈ Blended ROI of {avg_roi:.2f}× across ${total_spend/1e6:.1f}M total spend</span>
""", unsafe_allow_html=True)