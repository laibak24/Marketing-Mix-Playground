"""
1_overview.py — Channel ROI and revenue attribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.attribution import channel_contributions, channel_roi
from src.features import build_features

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Overview · MarketLytics", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ── Design system (identical across all pages) ────────────────────────────────
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
}
html,body,[class*="css"]{font-family:var(--font-sans);background:var(--bg)!important;color:var(--ink);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="collapsedControl"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]>div{padding:0!important;}
.topnav{position:sticky;top:0;z-index:1000;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 3rem;height:56px;box-shadow:0 1px 3px rgba(0,0,0,0.04);}
.topnav-brand{display:flex;align-items:center;gap:10px;}
.topnav-logo{width:30px;height:30px;background:var(--ink);border-radius:7px;display:flex;align-items:center;justify-content:center;color:white;font-family:var(--font-display);font-size:12px;font-weight:700;letter-spacing:-0.3px;}
.topnav-name{font-family:var(--font-display);font-size:15px;font-weight:700;color:var(--ink);letter-spacing:-0.3px;}
.topnav-links{display:flex;align-items:center;gap:4px;}
.topnav-link{font-size:13px;font-weight:500;color:var(--ink-mid);padding:6px 14px;border-radius:6px;text-decoration:none;transition:all 0.15s;display:inline-block;}
.topnav-link:hover{background:var(--bg);color:var(--ink);}
.topnav-link.active{background:var(--ink);color:white!important;}
.page-wrap{max-width:1100px;margin:0 auto;padding:2.5rem 3rem 5rem;}
.page-header{border-bottom:1px solid var(--border);padding-bottom:1.5rem;margin-bottom:2rem;}
.page-eyebrow{font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:8px;}
.page-title{font-family:var(--font-display);font-size:36px;font-weight:800;color:var(--ink);letter-spacing:-0.8px;margin:0 0 8px 0;}
.page-desc{font-size:14px;color:var(--ink-mid);margin-top:4px;line-height:1.6;max-width:600px;}
.section-title{font-family:var(--font-display);font-size:14px;font-weight:700;color:var(--ink);letter-spacing:-0.2px;margin:0 0 0.3rem 0;}
.section-desc{font-size:12px;color:var(--ink-muted);margin-bottom:1rem;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.25rem;}
[data-testid="stMetricLabel"]{font-size:10px!important;color:var(--ink-muted)!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.05em!important;}
[data-testid="stMetricValue"]{font-family:var(--font-display)!important;font-size:22px!important;font-weight:700!important;color:var(--ink)!important;letter-spacing:-0.4px!important;}
[data-testid="stMetricDelta"]{font-size:12px!important;}
.stButton>button{background:var(--ink)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:var(--font-sans)!important;font-size:13px!important;font-weight:500!important;padding:0.5rem 1.25rem!important;}
.stButton>button:hover{opacity:0.8!important;}
hr{border:none;border-top:1px solid var(--border);margin:2rem 0;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);background:transparent;}
.stTabs [data-baseweb="tab"]{font-size:13px;font-weight:500;color:var(--ink-mid);padding:0.5rem 1rem;border-bottom:2px solid transparent;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--ink)!important;}
.stCaption{font-size:11px!important;color:var(--ink-muted)!important;}
.insight-pill{display:inline-flex;align-items:center;gap:6px;background:var(--bg);border:1px solid var(--border);border-radius:20px;padding:6px 14px;font-size:12px;color:var(--ink-mid);margin:0.25rem 0.25rem 0.25rem 0;font-weight:500;}
.insight-pill.green{background:var(--green-bg);border-color:var(--green-border);color:var(--green);}
.insight-pill.amber{background:var(--amber-bg);border-color:#FDE68A;color:var(--amber);}
.next-footer{border-top:1px solid var(--border);margin-top:3rem;padding-top:1.5rem;display:flex;align-items:center;justify-content:space-between;}
.prev-link,.next-link{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;padding:9px 18px;border-radius:8px;text-decoration:none;transition:opacity 0.15s;}
.prev-link{background:var(--bg);border:1px solid var(--border);color:var(--ink)!important;}
.next-link{background:var(--ink);color:white!important;}
.prev-link:hover,.next-link:hover{opacity:0.8;}
</style>
""", unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">
    <div class="topnav-logo">ML</div>
    <div class="topnav-name">MarketLytics</div>
  </div>
  <div class="topnav-links">
    <a class="topnav-link" href="/">Home</a>
    <a class="topnav-link active" href="/1_overview">Overview</a>
    <a class="topnav-link" href="/2_playground">Playground</a>
    <a class="topnav-link" href="/3_diagnostics">Diagnostics</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Chart colours ─────────────────────────────────────────────────────────────
PALETTE = ["#141414", "#3D3D3D", "#6B6B6B", "#9B9B9B", "#C4C2BC", "#E8E6E1"]

# ── Page body ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Step 01 · Backward-looking</div>
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

# ── KPI row ───────────────────────────────────────────────────────────────────
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

    colors = [PALETTE[0] if i == len(roi_plot) - 1 else "#E2E0D9" for i in range(len(roi_plot))]
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(
        x=roi_plot["roi"], y=roi_plot["label"], orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}×" for v in roi_plot["roi"]],
        textposition="outside",
        textfont=dict(size=12, color="#141414", family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>ROI: %{x:.2f}×<extra></extra>",
    ))
    fig_roi.update_layout(
        height=280, margin=dict(l=0, r=60, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0EEE9", zeroline=False,
                   tickfont=dict(size=11, color="#8C8C8C"), showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#4A4A4A")),
        bargap=0.35,
    )
    st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar": False})

with col2:
    st.markdown('<div class="section-title">Revenue Contribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Share of attributed revenue per channel</div>', unsafe_allow_html=True)

    contribs_plot = contribs.copy()
    contribs_plot["label"] = contribs_plot["channel"].map(CH_LABELS).fillna(contribs_plot["channel"])
    top_ch = contribs_plot.loc[contribs_plot["contribution_pct"].idxmax(), "label"]

    fig_pie = go.Figure(go.Pie(
        labels=contribs_plot["label"],
        values=contribs_plot["contribution_pct"],
        hole=0.6,
        marker=dict(colors=PALETTE, line=dict(color="white", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig_pie.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="white",
        legend=dict(orientation="v", font=dict(size=11, color="#4A4A4A"),
                    itemsizing="constant", x=0.72, y=0.5),
        annotations=[dict(
            text=f"<b>{top_ch}</b><br><span style='color:#8C8C8C'>leads</span>",
            x=0.35, y=0.5, font_size=11, showarrow=False, font_color="#141414"
        )]
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
        line=dict(color="#141414", width=2),
        hovertemplate="<b>Revenue</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_ts.add_trace(go.Bar(
        x=df["DATE"], y=df["total_spend_week"], name="Media Spend",
        marker_color="#E2E0D9", yaxis="y2",
        hovertemplate="<b>Spend</b><br>%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_ts.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Revenue ($)", gridcolor="#F0EEE9", tickformat="$,.0f",
                   tickfont=dict(size=11, color="#8C8C8C")),
        yaxis2=dict(title="Spend ($)", overlaying="y", side="right",
                    tickfont=dict(size=11, color="#8C8C8C"), tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.08, font=dict(size=12)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

with tab2:
    melted = df[["DATE"] + CHANNELS].melt(id_vars="DATE", var_name="channel", value_name="spend")
    melted["channel"] = melted["channel"].map(CH_LABELS).fillna(melted["channel"])
    melted["spend"]   = pd.to_numeric(melted["spend"], errors="coerce").fillna(0)

    fig_area = go.Figure()
    for i, (ch_raw, ch_label) in enumerate(CH_LABELS.items()):
        ch_data = melted[melted["channel"] == ch_label]
        fig_area.add_trace(go.Bar(
            x=ch_data["DATE"], y=ch_data["spend"],
            name=ch_label, marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig_area.update_layout(
        barmode="stack", height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                   tickfont=dict(size=11, color="#8C8C8C")),
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

# ── Page nav ──────────────────────────────────────────────────────────────────
_pn_l, _pn_r = st.columns([1,1])
with _pn_l:
    st.page_link("main.py", label="← Home")
with _pn_r:
    st.page_link("pages/2_playground.py", label="Budget Playground →")
 
st.markdown('</div>', unsafe_allow_html=True)
 
