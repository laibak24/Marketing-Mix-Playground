"""
main.py — MarketLytics MMM Dashboard entry point.
Run: streamlit run app/main.py
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import joblib
import streamlit as st

DATA_PATH  = Path("data/raw/weekly_media_data.csv")
MODEL_PATH = Path("models/mmm_model.joblib")

st.set_page_config(
    page_title="MarketLytics · MMM",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root {
  color-scheme: light !important;
}
html, body, .stApp {
  background-color: #F7F6F2 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Shared design system (imported on every page) ─────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

/* FORCE LIGHT MODE OVERRIDES */
.stApp {
  background: var(--bg) !important;
}

[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
}

[data-testid="stMain"] {
  background: var(--bg) !important;
}

section.main {
  background: var(--bg) !important;
}

.block-container {
  background: transparent !important;
}

/* Fix markdown text rendering */
.stMarkdown, .stMarkdown p, .stMarkdown div {
  color: var(--ink) !important;
}
:root {
  --bg:            #F7F6F2;
  --surface:       #FFFFFF;
  --border:        #D6D3CC;
  --border-strong: #BFBBAF;

  --ink:           #111111;
  --ink-mid:       #2F2F2F;     /* darker */
  --ink-muted:     #6B6B6B;     /* more readable */

  --green:         #15803D;
  --red:           #DC2626;
  --amber:         #B45309;

  --radius:        10px;
  --green-border:  #BBF7D0;
  --red-bg:        #FEF2F2;
  --amber-bg:      #FFFBEB;
  --radius:        10px;
  --font-sans:     'DM Sans', sans-serif;
  --font-display:  'Syne', sans-serif;
  --font-mono:     'DM Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font-sans);
  background: var(--bg) !important;
  color: var(--ink);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── Sticky top nav ── */
.topnav {
  position: sticky; top: 0; z-index: 1000;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 3rem; height: 56px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.topnav-brand { display: flex; align-items: center; gap: 10px; }
.topnav-logo {
  width: 30px; height: 30px; background: var(--ink);
  border-radius: 7px; display: flex; align-items: center;
  justify-content: center; color: white;
  font-family: var(--font-display); font-size: 12px; font-weight: 700;
  letter-spacing: -0.3px;
}
.topnav-name {
  font-family: var(--font-display); font-size: 15px; font-weight: 700;
  color: var(--ink); letter-spacing: -0.3px;
}
.topnav-links { display: flex; align-items: center; gap: 4px; }
.topnav-link {
  font-size: 13px; font-weight: 500; color: var(--ink-mid);
  padding: 6px 14px; border-radius: 6px;
  text-decoration: none; transition: all 0.15s; display: inline-block;
}
.topnav-link:hover { background: var(--bg); color: var(--ink); }
.topnav-link.active { background: var(--ink); color: white !important; }

/* ── Page wrapper ── */
.page-wrap { max-width: 1100px; margin: 0 auto; padding: 3rem 3rem 5rem; }

/* ── Hero ── */
.hero {
  padding: 3.5rem 0 3rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2.5rem;

  background: var(--bg);   /* ADD THIS */
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 20px; padding: 5px 14px;
  font-size: 11px; color: var(--ink-muted); font-weight: 500;
  margin-bottom: 18px;
}
.hero-title {
  font-family: var(--font-display);
  font-size: 56px;
  font-weight: 800;
  color: #0A0A0A;   /* stronger than var(--ink) */
  letter-spacing: -1.5px;
  line-height: 1.05;
}
.hero-desc {
  font-size: 15px; color: var(--ink-mid); line-height: 1.7;
  max-width: 540px; margin-bottom: 0;
}

/* ── KPI strip ── */
.kpi-strip {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; margin-bottom: 2.5rem;
}
.kpi-cell { background: var(--surface); padding: 1.25rem 1.5rem; }
.kpi-label {
  font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 6px;
}
.kpi-value {
  font-family: var(--font-display); font-size: 26px; font-weight: 700;
  color: var(--ink); letter-spacing: -0.5px; line-height: 1.1;
}
.kpi-sub { font-size: 12px; color: var(--ink-muted); margin-top: 4px; }

/* ── Section header ── */
.section-header { margin-bottom: 1.25rem; }
.section-title {
  font-family: var(--font-display); font-size: 20px; font-weight: 700;
  color: var(--ink); letter-spacing: -0.4px; margin: 0 0 4px 0;
}
.section-desc { font-size: 13px; color: var(--ink-muted); }

/* ── How it works ── */
.hiw-steps {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; margin-bottom: 2.5rem;
}
.hiw-step { background: var(--surface); padding: 2rem; }
.hiw-num {
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 0.75rem;
}
.hiw-icon { font-size: 22px; margin-bottom: 0.5rem; }
.hiw-step-title {
  font-family: var(--font-display); font-size: 17px; font-weight: 700;
  color: var(--ink); letter-spacing: -0.3px; margin-bottom: 0.5rem;
}
.hiw-step-desc { font-size: 13px; color: var(--ink-mid); line-height: 1.65; }
.hiw-arrow {
  display: flex; align-items: center; justify-content: center;
  background: var(--bg); color: var(--ink-muted); font-size: 18px;
  align-self: stretch; padding: 0 0.5rem;
}

/* ── Feature cards ── */
.feature-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 16px; margin-bottom: 2.5rem;
}
.feature-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.5rem;
  transition: border-color 0.15s, box-shadow 0.15s; cursor: default;
}
.feature-card:hover {
  border-color: var(--border-strong);
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.feature-icon { font-size: 20px; margin-bottom: 0.6rem; }
.feature-dir {
  font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 4px;
}
.feature-title {
  font-family: var(--font-display); font-size: 15px; font-weight: 700;
  color: var(--ink); margin-bottom: 0.4rem;
}
.feature-desc { font-size: 13px; color: var(--ink-mid); line-height: 1.6; }

/* ── Tech stack strip ── */
.tech-strip {
  border-top: 1px solid var(--border); padding-top: 1.5rem;
  font-size: 12px; color: var(--ink-muted); line-height: 1.8;
}

/* ── Page footer (next link) ── */
.next-footer {
  border-top: 1px solid var(--border); margin-top: 3rem;
  padding-top: 1.5rem;
  display: flex; align-items: center; justify-content: flex-end;
}
.next-link {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--ink); color: white !important;
  font-size: 13px; font-weight: 600;
  padding: 9px 18px; border-radius: 8px;
  text-decoration: none; transition: opacity 0.15s;
}
.next-link:hover { opacity: 0.8; }

/* ── Streamlit widget overrides ── */
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important; color: var(--ink-muted) !important;
  font-weight: 600 !important; text-transform: uppercase;
  letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 22px !important; font-weight: 700 !important;
  color: var(--ink) !important; letter-spacing: -0.4px !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }
.stButton > button {
  background: var(--ink) !important; color: white !important;
  border: none !important; border-radius: 8px !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 0.5rem 1.25rem !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.8 !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--ink) !important; }
[data-testid="stNumberInput"] input {
  font-family: var(--font-mono) !important; font-size: 14px !important;
}
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.stCaption { font-size: 11px !important; color: var(--ink-muted) !important; }
[data-testid="stDataFrame"] {
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 1px solid var(--border); background: transparent;
}
.stTabs [data-baseweb="tab"] {
  font-size: 13px; font-weight: 500; color: var(--ink-mid);
  padding: 0.5rem 1rem; border-bottom: 2px solid transparent;
  background: transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important; border-bottom: 2px solid var(--ink) !important;
}
.stAlert { border-radius: 8px !important; font-size: 13px !important; }

/* ── st.page_link nav styling ── */
[data-testid="stPageLink"] {
  margin-top: 20px;
}

[data-testid="stPageLink"] a {
  background: var(--ink) !important;
  color: white !important;
  border-radius: 8px !important;
  padding: 10px 18px !important;
  font-weight: 600 !important;
  display: inline-block;
}
[data-testid="stPageLink-active"] a { background:var(--ink)!important; color:white!important; }
[data-testid="stSidebarNav"] { display:none!important; }
[data-testid="stPageLink"] a {
  background: var(--ink) !important;
  color: white !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  font-weight: 600 !important;
}
</style>
"""
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
with st.container():
    st.page_link("pages/1_overview.py", label="Start with Channel Overview →")

st.markdown('</div>', unsafe_allow_html=True)

# ── Load data for live KPIs ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["DATE"])
    return None

@st.cache_resource
def load_artifacts():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df        = load_data()
artifacts = load_artifacts()
CHANNELS  = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"]

# ── Page body ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-badge">◈ Internship Project &nbsp;·&nbsp; Marketing Mix Modelling</div>
  <div class="hero-title">Understand what your<br>marketing is really doing.</div>
  <div class="hero-desc">
    MarketLytics uses a Ridge Regression Marketing Mix Model with adstock decay and Hill saturation
    transforms to decompose revenue across 6 media channels — and find the mathematically optimal
    budget allocation.
  </div>
</div>
""", unsafe_allow_html=True)

# KPI strip
if df is not None and artifacts is not None:
    CHANNELS_present = [ch for ch in CHANNELS if ch in df.columns]
    total_rev   = df["revenue"].sum() if "revenue" in df.columns else 0
    total_spend = df[CHANNELS_present].apply(pd.to_numeric, errors="coerce").sum().sum()
    avg_roi     = total_rev / total_spend if total_spend > 0 else 0
    cv_mape     = artifacts.get("cv_mape", 0)
    n_weeks     = len(df)
    kpi_html = f"""
    <div class="kpi-strip">
      <div class="kpi-cell">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">${total_rev/1e6:.1f}M</div>
        <div class="kpi-sub">{n_weeks} weekly observations</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Total Media Spend</div>
        <div class="kpi-value">${total_spend/1e6:.1f}M</div>
        <div class="kpi-sub">Across 6 channels</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Blended ROI</div>
        <div class="kpi-value">{avg_roi:.2f}×</div>
        <div class="kpi-sub">Revenue per $ of spend</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Model CV MAPE</div>
        <div class="kpi-value">{cv_mape:.1%}</div>
        <div class="kpi-sub">5-fold TimeSeriesSplit</div>
      </div>
    </div>
    """
else:
    kpi_html = """
    <div class="kpi-strip">
      <div class="kpi-cell"><div class="kpi-label">Total Revenue</div><div class="kpi-value">—</div><div class="kpi-sub">Run python -m src.model first</div></div>
      <div class="kpi-cell"><div class="kpi-label">Media Spend</div><div class="kpi-value">—</div></div>
      <div class="kpi-cell"><div class="kpi-label">Blended ROI</div><div class="kpi-value">—</div></div>
      <div class="kpi-cell"><div class="kpi-label">CV MAPE</div><div class="kpi-value">—</div></div>
    </div>
    """
st.markdown(kpi_html, unsafe_allow_html=True)

# How it works
st.markdown("""
<div class="section-header">
  <div class="section-title">How it works</div>
  <div class="section-desc">Three pages, one story — from measurement to optimisation to validation.</div>
</div>
<div class="hiw-steps">
  <div class="hiw-step">
    <div class="hiw-icon">📈</div>
    <div class="hiw-num">Step 01 &nbsp;·&nbsp; Backward-looking</div>
    <div class="hiw-step-title">Understand the past</div>
    <div class="hiw-step-desc">See which channels drove revenue, compare ROI across all 6 channels, and review spend vs revenue trends over 208 weeks.</div>
  </div>
  <div class="hiw-step">
    <div class="hiw-icon">🎛️</div>
    <div class="hiw-num">Step 02 &nbsp;·&nbsp; Forward-looking</div>
    <div class="hiw-step-title">Simulate the future</div>
    <div class="hiw-step-desc">Drag sliders to reallocate budget and watch predicted revenue update live. Hit "Find Optimal Allocation" to let the model do the maths.</div>
  </div>
  <div class="hiw-step">
    <div class="hiw-icon">🔬</div>
    <div class="hiw-num">Step 03 &nbsp;·&nbsp; Credibility layer</div>
    <div class="hiw-step-title">Trust the numbers</div>
    <div class="hiw-step-desc">Validate fit with actual vs predicted, residual analysis, coefficient breakdown, saturation curves, and cross-validated error metrics.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
st.markdown("""
<div class="section-header">
  <div class="section-title">Explore the platform</div>
</div>
<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">📈</div>
    <div class="feature-dir">Backward-looking</div>
    <div class="feature-title">Channel Overview</div>
    <div class="feature-desc">ROI by channel, revenue contribution donut chart, and stacked spend breakdown — with a date-range filter.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🎛️</div>
    <div class="feature-dir">Forward-looking</div>
    <div class="feature-title">Budget Playground</div>
    <div class="feature-desc">Set a total budget, distribute across channels, and see predicted revenue delta vs historical baseline in real time.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔬</div>
    <div class="feature-dir">Credibility layer</div>
    <div class="feature-title">Model Diagnostics</div>
    <div class="feature-desc">Actual vs predicted, residuals, coefficient waterfall, saturation curves, and CV score cards — all in one view.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Tech footnote + next CTA
st.markdown("""
<div class="tech-strip">
  Dataset · Robyn Open-Source MMM · 208 weekly observations · 6 media channels &nbsp;·&nbsp;
  Model · Ridge Regression + adstock + Hill saturation &nbsp;·&nbsp;
  Validation · 5-fold TimeSeriesSplit CV
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)