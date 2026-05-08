"""
3_diagnostics.py — Model validation and diagnostics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import get_X_y
from src.saturation import hill_saturation, SATURATION_DEFAULTS
from src.adstock import CHANNELS

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Diagnostics · MarketLytics", page_icon="🔬", layout="wide",
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
}
html,body,[class*="css"]{font-family:var(--font-sans);background:var(--bg)!important;color:var(--ink);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="collapsedControl"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]>div{padding:0!important;}
.topnav{position:sticky;top:0;z-index:1000;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 3rem;height:56px;box-shadow:0 1px 3px rgba(0,0,0,0.04);}
.topnav-brand{display:flex;align-items:center;gap:10px;}
.topnav-logo{width:30px;height:30px;background:var(--ink);border-radius:7px;display:flex;align-items:center;justify-content:center;color:white;font-family:var(--font-display);font-size:12px;font-weight:700;}
.topnav-name{font-family:var(--font-display);font-size:15px;font-weight:700;color:var(--ink);letter-spacing:-0.3px;}
.topnav-links{display:flex;align-items:center;gap:4px;}
.topnav-link{font-size:13px;font-weight:500;color:var(--ink-mid);padding:6px 14px;border-radius:6px;text-decoration:none;transition:all 0.15s;display:inline-block;}
.topnav-link:hover{background:var(--bg);color:var(--ink);}
.topnav-link.active{background:var(--ink);color:white!important;}
.page-wrap{max-width:1100px;margin:0 auto;padding:2.5rem 3rem 5rem;}
.page-header{border-bottom:1px solid var(--border);padding-bottom:1.5rem;margin-bottom:2rem;}
.page-eyebrow{font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:8px;}
.page-title{font-family:var(--font-display);font-size:36px;font-weight:800;color:var(--ink);letter-spacing:-0.8px;margin:0 0 8px 0;}
.page-desc{font-size:14px;color:var(--ink-mid);margin-top:4px;line-height:1.6;max-width:640px;}
.section-title{font-family:var(--font-display);font-size:14px;font-weight:700;color:var(--ink);letter-spacing:-0.2px;margin:0 0 0.3rem 0;}
.section-desc{font-size:12px;color:var(--ink-muted);margin-bottom:1rem;}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.25rem;}
[data-testid="stMetricLabel"]{font-size:10px!important;color:var(--ink-muted)!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.05em!important;}
[data-testid="stMetricValue"]{font-family:var(--font-display)!important;font-size:22px!important;font-weight:700!important;color:var(--ink)!important;letter-spacing:-0.4px!important;}
hr{border:none;border-top:1px solid var(--border);margin:2rem 0;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);background:transparent;}
.stTabs [data-baseweb="tab"]{font-size:13px;font-weight:500;color:var(--ink-mid);padding:0.5rem 1rem;border-bottom:2px solid transparent;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--ink)!important;}
.stCaption{font-size:11px!important;color:var(--ink-muted)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;}

/* ── Score cards ── */
.score-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;text-align:center;}
.score-value{font-family:var(--font-display);font-size:32px;font-weight:800;color:var(--ink);letter-spacing:-0.8px;}
.score-label{font-size:10px;color:var(--ink-muted);font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;}
.score-sub{font-size:12px;color:var(--ink-muted);margin-top:6px;line-height:1.4;}
.verdict{display:inline-flex;align-items:center;gap:5px;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600;margin-top:10px;}
.verdict.good{background:var(--green-bg);border:1px solid var(--green-border);color:var(--green);}
.verdict.warn{background:var(--amber-bg);border:1px solid #FDE68A;color:var(--amber);}
.verdict.bad{background:var(--red-bg);border:1px solid #FECACA;color:var(--red);}

/* ── Context banner ── */
.context-banner{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem 1.5rem;margin-bottom:1.5rem;display:flex;align-items:flex-start;gap:12px;}
.context-icon{font-size:18px;flex-shrink:0;margin-top:2px;}
.context-body{}
.context-title{font-family:var(--font-display);font-size:14px;font-weight:700;color:var(--ink);margin-bottom:4px;}
.context-text{font-size:13px;color:var(--ink-mid);line-height:1.6;}

/* ── Page nav ── */
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
    <a class="topnav-link" href="/1_overview">Overview</a>
    <a class="topnav-link" href="/2_playground">Playground</a>
    <a class="topnav-link active" href="/3_diagnostics">Diagnostics</a>
  </div>
</div>
""", unsafe_allow_html=True)

CH_LABELS = {
    "tv_S": "TV", "ooh_S": "Out-of-Home", "print_S": "Print",
    "facebook_S": "Facebook", "search_S": "Paid Search", "newsletter": "Newsletter"
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
model        = artifacts["model"]
scaler       = artifacts["scaler"]
feature_cols = artifacts["feature_cols"]

X, y, _ = get_X_y(df)
X_scaled  = scaler.transform(X)
y_pred    = model.predict(X_scaled)
residuals = y - y_pred

# ── Page body ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Step 03 · Credibility layer</div>
  <div class="page-title">Model Diagnostics</div>
  <div class="page-desc">
    The Budget Playground's recommendations are only as good as the model driving them.
    This page proves the model is trustworthy — with cross-validated error, residual analysis,
    and saturation curves. Good MMM targets: MAPE &lt; 15%, R² &gt; 0.80.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Why diagnostics matter ────────────────────────────────────────────────────
st.markdown("""
<div class="context-banner">
  <div class="context-icon">💡</div>
  <div class="context-body">
    <div class="context-title">Why this page matters</div>
    <div class="context-text">
      Actual vs Predicted, residuals, and CV MAPE show you the difference between a dashboard
      and a <em>validated</em> model. These diagnostics confirm the model generalises to unseen
      data — not just the training set — which is what makes the budget recommendations credible.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Score cards ───────────────────────────────────────────────────────────────
cv_mape    = artifacts["cv_mape"]
cv_r2      = artifacts["cv_r2"]
train_mape = artifacts["train_mape"]
train_r2   = artifacts["train_r2"]

mape_verdict = ("✓ Excellent", "good") if cv_mape < 0.10 else \
               ("✓ Good", "good")      if cv_mape < 0.15 else \
               ("△ Acceptable", "warn") if cv_mape < 0.20 else ("✕ Needs work", "bad")
r2_verdict   = ("✓ Excellent", "good") if cv_r2 > 0.85 else \
               ("✓ Good", "good")      if cv_r2 > 0.75 else \
               ("△ Acceptable", "warn") if cv_r2 > 0.60 else ("✕ Needs work", "bad")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{cv_mape:.1%}</div>
      <div class="score-label">CV MAPE</div>
      <div class="score-sub">Cross-validated mean<br>absolute percentage error</div>
      <div class="verdict {mape_verdict[1]}">{mape_verdict[0]}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{cv_r2:.3f}</div>
      <div class="score-label">CV R²</div>
      <div class="score-sub">Variance explained<br>on holdout folds</div>
      <div class="verdict {r2_verdict[1]}">{r2_verdict[0]}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{train_mape:.1%}</div>
      <div class="score-label">Train MAPE</div>
      <div class="score-sub">In-sample fit<br>(always lower than CV)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{artifacts['n_obs']}</div>
      <div class="score-label">Observations</div>
      <div class="score-sub">Weekly data points<br>used in training</div>
    </div>""", unsafe_allow_html=True)

st.caption("CV metrics use 5-fold TimeSeriesSplit — they reflect real-world holdout performance, not in-sample fit.")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Diagnostic tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Fit Quality", "Coefficients", "Saturation Curves"])

with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-title">Actual vs Predicted Revenue</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Dashed line (predicted) should closely track the solid line (actual)</div>', unsafe_allow_html=True)
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=df["DATE"], y=y, name="Actual",
            line=dict(color="#141414", width=2),
            hovertemplate="%{x|%b %Y}<br>Actual: $%{y:,.0f}<extra></extra>",
        ))
        fig_fit.add_trace(go.Scatter(
            x=df["DATE"], y=y_pred, name="Predicted",
            line=dict(color="#9B9B9B", width=1.5, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>Predicted: $%{y:,.0f}<extra></extra>",
        ))
        fig_fit.update_layout(
            height=310, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                       tickfont=dict(size=11, color="#8C8C8C")),
            xaxis=dict(tickfont=dict(size=11, color="#8C8C8C")),
            legend=dict(orientation="h", y=1.1, font=dict(size=12)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fit, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="section-title">Residuals Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Should be random noise around zero — any visible pattern signals model issues</div>', unsafe_allow_html=True)
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=df["DATE"], y=residuals,
            marker_color=["#DC2626" if r < 0 else "#86EFAC" for r in residuals],
            hovertemplate="%{x|%b %Y}<br>Residual: $%{y:,.0f}<extra></extra>",
        ))
        fig_res.add_hline(y=0, line_dash="dot", line_color="#8C8C8C", line_width=1)
        fig_res.update_layout(
            height=310, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                       tickfont=dict(size=11, color="#8C8C8C")),
            xaxis=dict(tickfont=dict(size=11, color="#8C8C8C")),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">Residual Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Should be approximately bell-shaped and centred near zero</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals, nbinsx=30,
        marker_color="#141414", opacity=0.75,
    ))
    fig_hist.add_vline(x=0, line_dash="dot", line_color="#DC2626", line_width=1.5)
    fig_hist.add_vline(x=np.mean(residuals), line_dash="dash", line_color="#8C8C8C",
                       annotation_text=f"Mean: ${np.mean(residuals):,.0f}",
                       annotation_font_size=11)
    fig_hist.update_layout(
        height=240, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat="$,.0f", tickfont=dict(size=11, color="#8C8C8C")),
        yaxis=dict(gridcolor="#F0EEE9", tickfont=dict(size=11, color="#8C8C8C"), title="Weeks"),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with tab2:
    st.markdown('<div class="section-title">Feature Coefficients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Larger absolute value = stronger influence on revenue. Negative = suppressor effect. Coefficients are in scaled feature space — compare relative magnitudes.</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": model.coef_,
    })
    coef_df["Display"] = coef_df["Feature"].apply(
        lambda x: CH_LABELS.get(x.replace("_saturated",""), x)
                  .replace("_saturated","").replace("_"," ").title()
    )
    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=True)

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Display"],
        orientation="h",
        marker_color=["#DC2626" if v < 0 else "#141414" for v in coef_df["Coefficient"]],
        text=[f"{v:+,.0f}" for v in coef_df["Coefficient"]],
        textposition="outside",
        textfont=dict(size=11, color="#4A4A4A"),
        hovertemplate="<b>%{y}</b><br>Coef: %{x:+,.2f}<extra></extra>",
    ))
    fig_coef.add_vline(x=0, line_color="#E2E0D9", line_width=1)
    fig_coef.update_layout(
        height=420, margin=dict(l=0,r=80,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0EEE9", zeroline=False,
                   tickfont=dict(size=11, color="#8C8C8C"), showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color="#4A4A4A")),
    )
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View raw coefficient values"):
        raw = coef_df[["Display", "Feature", "Coefficient"]].copy()
        raw["Coefficient"] = raw["Coefficient"].apply(lambda x: f"{x:+,.4f}")
        st.dataframe(raw.rename(columns={"Display":"Name"}), use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-title">Saturation Curves by Channel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Each channel has diminishing returns — the curve flattens as spend increases. The dot marks current average weekly spend. The dashed line is the 50% saturation point.</div>', unsafe_allow_html=True)

    sat_cols = st.columns(3)
    ch_list  = [ch for ch in CHANNELS if ch in df.columns]

    for i, ch in enumerate(ch_list):
        label      = CH_LABELS.get(ch, ch)
        params     = SATURATION_DEFAULTS.get(ch, {"alpha": 1.5, "gamma": 10000})
        spend_vals = pd.to_numeric(df[ch], errors="coerce").dropna()
        max_spend  = spend_vals.max() * 1.5 if len(spend_vals) > 0 else 100_000
        avg_spend  = spend_vals.mean() if len(spend_vals) > 0 else 0
        x_range    = np.linspace(0, max_spend, 300)
        y_sat      = hill_saturation(x_range, params["alpha"], params["gamma"])
        y_at_avg   = hill_saturation(np.array([avg_spend]), params["alpha"], params["gamma"])[0]

        fig_sat = go.Figure()
        fig_sat.add_trace(go.Scatter(
            x=x_range, y=y_sat, mode="lines",
            line=dict(color="#141414", width=2), showlegend=False,
            hovertemplate="Spend: $%{x:,.0f}<br>Response: %{y:.3f}<extra></extra>",
        ))
        fig_sat.add_trace(go.Scatter(
            x=[avg_spend], y=[y_at_avg], mode="markers",
            marker=dict(color="#141414", size=8), showlegend=False,
            hovertemplate=f"Avg: ${avg_spend:,.0f}<br>Response: {y_at_avg:.3f}<extra></extra>",
        ))
        fig_sat.add_hline(y=0.5, line_dash="dot", line_color="#E2E0D9", line_width=1)
        fig_sat.update_layout(
            title=dict(text=label, font=dict(size=12, color="#4A4A4A", family="Syne"), x=0),
            height=200, margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickformat="$,.0f", tickfont=dict(size=9, color="#8C8C8C")),
            yaxis=dict(range=[0,1.05], gridcolor="#F0EEE9",
                       tickfont=dict(size=9, color="#8C8C8C")),
        )
        with sat_cols[i % 3]:
            st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"α={params['alpha']} · γ=${params['gamma']:,} · avg ${avg_spend:,.0f}/wk")

# ── Page nav ──────────────────────────────────────────────────────────────────
_pn_l, _pn_r = st.columns([1,1])
with _pn_l:
    st.page_link("pages/2_playground.py", label="← Budget Playground")
with _pn_r:
    st.page_link("main.py", label="Back to Home ↩")
 
st.markdown('</div>', unsafe_allow_html=True)
 
