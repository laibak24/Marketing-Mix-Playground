"""
main.py — MarketLytics MMM Dashboard entry point.
Run: streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title="MarketLytics · MMM",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 2rem 3rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FAFAF9;
    border-right: 1px solid #E8E6E1;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 2rem;
}
.sidebar-brand-icon {
    width: 32px; height: 32px; background: #1a1a1a;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; color: white; font-size: 14px; font-weight: 600;
}
.sidebar-brand-text { font-size: 15px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.3px; }
.sidebar-sub { font-size: 11px; color: #9B9B9B; margin-top: 1px; }

.nav-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9B9B9B; margin: 1.5rem 0 0.5rem 0;
}

/* Stale link removal */
a { text-decoration: none !important; }

/* Page content */
.page-header {
    border-bottom: 1px solid #E8E6E1;
    padding-bottom: 1.5rem; margin-bottom: 2rem;
}
.page-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9B9B9B; margin-bottom: 6px;
}
.page-title {
    font-size: 28px; font-weight: 600; color: #1a1a1a;
    letter-spacing: -0.5px; margin: 0;
}
.page-desc { font-size: 14px; color: #6B6B6B; margin-top: 6px; line-height: 1.5; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
            background: #E8E6E1; border: 1px solid #E8E6E1; border-radius: 10px;
            overflow: hidden; margin-bottom: 2rem; }
.kpi-card { background: #fff; padding: 1.25rem 1.5rem; }
.kpi-label { font-size: 11px; color: #9B9B9B; font-weight: 500;
             letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 24px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.5px; }
.kpi-delta { font-size: 12px; color: #6B6B6B; margin-top: 3px; }
.kpi-delta.pos { color: #16A34A; }
.kpi-delta.neg { color: #DC2626; }

/* Section headers */
.section-title {
    font-size: 13px; font-weight: 600; color: #1a1a1a;
    letter-spacing: -0.2px; margin: 0 0 1rem 0;
}
.section-desc { font-size: 12px; color: #9B9B9B; margin-top: -0.75rem; margin-bottom: 1rem; }

/* Cards / panels */
.panel {
    background: #fff; border: 1px solid #E8E6E1;
    border-radius: 10px; padding: 1.5rem;
}

/* Metric overrides */
[data-testid="stMetric"] {
    background: #FAFAF9; border: 1px solid #E8E6E1;
    border-radius: 10px; padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #9B9B9B !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 600 !important;
    color: #1a1a1a !important; letter-spacing: -0.4px !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Buttons */
.stButton > button {
    background: #1a1a1a !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important; letter-spacing: -0.1px !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.8 !important; }

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #1a1a1a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #E8E6E1; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px; font-weight: 500; color: #6B6B6B;
    padding: 0.5rem 1rem; border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #1a1a1a !important; border-bottom: 2px solid #1a1a1a !important;
}

/* Divider */
hr { border: none; border-top: 1px solid #E8E6E1; margin: 2rem 0; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid #E8E6E1; border-radius: 10px; overflow: hidden; }

/* Info / warning boxes */
.stAlert { border-radius: 8px !important; border: 1px solid #E8E6E1 !important;
           font-size: 13px !important; }

/* Number input */
[data-testid="stNumberInput"] input {
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
}

/* Plotly charts */
.js-plotly-plot { border-radius: 8px; }

/* Caption */
.stCaption { font-size: 11px !important; color: #9B9B9B !important; }

/* Sidebar nav items */
[data-testid="stSidebarNav"] a {
    font-size: 13px !important; color: #3D3D3D !important;
    font-weight: 400 !important; padding: 0.4rem 0.75rem !important;
    border-radius: 6px !important;
}
[data-testid="stSidebarNav"] a:hover { background: #F0EEE9 !important; }
[data-testid="stSidebarNav"] [aria-current="page"] {
    background: #ECEAE4 !important; font-weight: 500 !important; color: #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">ML</div>
  <div>
    <div class="sidebar-brand-text">MarketLytics</div>
    <div class="sidebar-sub">MMM Platform</div>
  </div>
</div>
<div class="nav-label">Analytics</div>
""", unsafe_allow_html=True)

# Home page content
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">MarketLytics · MMM Platform</div>
  <div class="page-title">Marketing Mix Model</div>
  <div class="page-desc">Understand how media spend across channels drives revenue — and find the optimal budget allocation.</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="panel">
      <div style="font-size:20px;margin-bottom:0.75rem;">📈</div>
      <div class="section-title">Channel Overview</div>
      <div style="font-size:13px;color:#6B6B6B;line-height:1.6;">
        ROI by channel, revenue contribution breakdown, and spend vs revenue trends over time.
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="panel">
      <div style="font-size:20px;margin-bottom:0.75rem;">🎛️</div>
      <div class="section-title">Budget Playground</div>
      <div style="font-size:13px;color:#6B6B6B;line-height:1.6;">
        Reallocate spend across channels and see predicted revenue change in real time.
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="panel">
      <div style="font-size:20px;margin-bottom:0.75rem;">🔬</div>
      <div class="section-title">Model Diagnostics</div>
      <div style="font-size:13px;color:#6B6B6B;line-height:1.6;">
        Validate model performance with actual vs predicted, residual analysis, and CV metrics.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size:12px;color:#9B9B9B;">
Dataset · Robyn Open-Source MMM · 208 weekly observations · 6 media channels
&nbsp;·&nbsp; Model · Ridge Regression with adstock + Hill saturation transforms
&nbsp;·&nbsp; Validation · 5-fold TimeSeriesSplit CV
</div>
""", unsafe_allow_html=True)