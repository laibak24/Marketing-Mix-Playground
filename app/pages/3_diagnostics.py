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
  color-scheme: light !important;
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main {
  font-family: var(--font-sans);
  background: var(--bg) !important;
  color: var(--ink);
}

.block-container { background: transparent !important; }

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
.navbar-brand { display: flex; align-items: center; gap: 10px; }
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
.navbar-links { display: flex; align-items: center; gap: 4px; }
.nav-link {
  font-size: 13px; font-weight: 500; color: #4A4A4A;
  padding: 6px 14px; border-radius: 6px;
  text-decoration: none !important;
  transition: background 0.15s, color 0.15s;
  display: inline-block; cursor: pointer;
}
.nav-link:hover { background: #F7F6F2; color: #141414; text-decoration: none !important; }
.nav-link.active { background: #141414; color: #FFFFFF !important; text-decoration: none !important; }

/* ── Page layout ── */
.page-wrap { max-width: 1100px; margin: 0 auto; padding: 2.5rem 3rem 5rem; }

/* ── Streamlit components ── */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.25rem; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: var(--ink-muted) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; font-size: 22px !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -0.4px !important; }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); background: transparent; }
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: var(--ink-mid); padding: 0.5rem 1rem; border-bottom: 2px solid transparent; background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom: 2px solid var(--ink) !important; }
.stCaption { font-size: 11px !important; color: var(--ink-muted) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }

/* ── Expander ── */
[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
[data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600 !important; }
[data-testid="stExpander"] summary span, [data-testid="stExpander"] summary p { color: var(--ink) !important; }
[data-testid="stExpander"] svg { fill: var(--ink) !important; }

/* ── Tab labels ── */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

# ── Navbar (pure HTML) ────────────────────────────────────────────────────────
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
      <a class="nav-link" href="/playground">Playground</a>
      <a class="nav-link active" href="/diagnostics">Diagnostics</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
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

# Page header — fully inline styled
st.markdown("""
<div style="border-bottom:1px solid #E2E0D9;padding-bottom:1.5rem;margin-bottom:2rem;">
  <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8C8C8C;margin-bottom:8px;">Step 03 · Credibility layer</div>
  <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:#141414;letter-spacing:-0.8px;margin:0 0 8px 0;">Model Diagnostics</div>
  <div style="font-size:14px;color:#4A4A4A;margin-top:4px;line-height:1.6;max-width:640px;">
    The Budget Playground's recommendations are only as good as the model driving them.
    This page proves the model is trustworthy — with cross-validated error, residual analysis,
    and saturation curves. Good MMM targets: MAPE &lt; 15%, R² &gt; 0.80.
  </div>
</div>
""", unsafe_allow_html=True)

# Context banner — fully inline styled
st.markdown("""
<div style="background:#FFFFFF;border:1px solid #E2E0D9;border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;display:flex;align-items:flex-start;gap:12px;">
  <div style="font-size:18px;flex-shrink:0;margin-top:2px;">💡</div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#141414;margin-bottom:4px;">Why this page matters</div>
    <div style="font-size:13px;color:#4A4A4A;line-height:1.6;">
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

def verdict_html(label, cls):
    colors = {
        "good":  ("background:#F0FDF4;border:1px solid #BBF7D0;color:#15803D;", "✓"),
        "warn":  ("background:#FFFBEB;border:1px solid #FDE68A;color:#B45309;", "△"),
        "bad":   ("background:#FEF2F2;border:1px solid #FECACA;color:#DC2626;", "✕"),
    }
    style, sym = colors.get(cls, colors["warn"])
    return f'<div style="display:inline-flex;align-items:center;gap:5px;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600;margin-top:10px;{style}">{sym} {label}</div>'

mape_verdict = ("Excellent", "good") if cv_mape < 0.10 else \
               ("Good", "good")      if cv_mape < 0.15 else \
               ("Acceptable", "warn") if cv_mape < 0.20 else ("Needs work", "bad")
r2_verdict   = ("Excellent", "good") if cv_r2 > 0.85 else \
               ("Good", "good")      if cv_r2 > 0.75 else \
               ("Acceptable", "warn") if cv_r2 > 0.60 else ("Needs work", "bad")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E2E0D9;border-radius:10px;padding:1.5rem;text-align:center;">
      <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#141414;letter-spacing:-0.8px;">{cv_mape:.1%}</div>
      <div style="font-size:10px;color:#8C8C8C;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;">CV MAPE</div>
      <div style="font-size:12px;color:#8C8C8C;margin-top:6px;line-height:1.4;">Cross-validated mean<br>absolute percentage error</div>
      {verdict_html(*mape_verdict)}
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E2E0D9;border-radius:10px;padding:1.5rem;text-align:center;">
      <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#141414;letter-spacing:-0.8px;">{cv_r2:.3f}</div>
      <div style="font-size:10px;color:#8C8C8C;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;">CV R²</div>
      <div style="font-size:12px;color:#8C8C8C;margin-top:6px;line-height:1.4;">Variance explained<br>on holdout folds</div>
      {verdict_html(*r2_verdict)}
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E2E0D9;border-radius:10px;padding:1.5rem;text-align:center;">
      <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#141414;letter-spacing:-0.8px;">{train_mape:.1%}</div>
      <div style="font-size:10px;color:#8C8C8C;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;">Train MAPE</div>
      <div style="font-size:12px;color:#8C8C8C;margin-top:6px;line-height:1.4;">In-sample fit<br>(always lower than CV)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E2E0D9;border-radius:10px;padding:1.5rem;text-align:center;">
      <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#141414;letter-spacing:-0.8px;">{artifacts['n_obs']}</div>
      <div style="font-size:10px;color:#8C8C8C;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-top:4px;">Observations</div>
      <div style="font-size:12px;color:#8C8C8C;margin-top:6px;line-height:1.4;">Weekly data points<br>used in training</div>
    </div>""", unsafe_allow_html=True)

st.caption("CV metrics use 5-fold TimeSeriesSplit — they reflect real-world holdout performance, not in-sample fit.")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Diagnostic tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Fit Quality", "Coefficients", "Saturation Curves"])

with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:14px;font-weight:700;color:#141414;letter-spacing:-0.2px;margin:0 0 0.3rem 0;">Actual vs Predicted Revenue</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#8C8C8C;margin-bottom:1rem;">Dashed line (predicted) should closely track the solid line (actual)</div>', unsafe_allow_html=True)
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
            height=310, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                       tickfont=dict(size=11, color="#8C8C8C")),
            xaxis=dict(tickfont=dict(size=11, color="#8C8C8C")),
            legend=dict(orientation="h", y=1.1, font=dict(size=12, color="#141414")),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fit, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:14px;font-weight:700;color:#141414;letter-spacing:-0.2px;margin:0 0 0.3rem 0;">Residuals Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#8C8C8C;margin-bottom:1rem;">Should be random noise around zero — any visible pattern signals model issues</div>', unsafe_allow_html=True)
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=df["DATE"], y=residuals,
            marker_color=["#DC2626" if r < 0 else "#86EFAC" for r in residuals],
            hovertemplate="%{x|%b %Y}<br>Residual: $%{y:,.0f}<extra></extra>",
        ))
        fig_res.add_hline(y=0, line_dash="dot", line_color="#8C8C8C", line_width=1)
        fig_res.update_layout(
            height=310, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                       tickfont=dict(size=11, color="#8C8C8C")),
            xaxis=dict(tickfont=dict(size=11, color="#8C8C8C")),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:14px;font-weight:700;color:#141414;letter-spacing:-0.2px;margin:0 0 0.3rem 0;">Residual Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8C8C8C;margin-bottom:1rem;">Should be approximately bell-shaped and centred near zero</div>', unsafe_allow_html=True)
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
        height=240, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat="$,.0f", tickfont=dict(size=11, color="#8C8C8C")),
        yaxis=dict(gridcolor="#F0EEE9", tickfont=dict(size=11, color="#8C8C8C"), title="Weeks"),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with tab2:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:14px;font-weight:700;color:#141414;letter-spacing:-0.2px;margin:0 0 0.3rem 0;">Feature Coefficients</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8C8C8C;margin-bottom:1rem;">Larger absolute value = stronger influence on revenue. Negative = suppressor effect. Coefficients are in scaled feature space — compare relative magnitudes.</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": model.coef_,
    })
    coef_df["Display"] = coef_df["Feature"].apply(
        lambda x: CH_LABELS.get(x.replace("_saturated", ""), x)
                  .replace("_saturated", "").replace("_", " ").title()
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
        height=420, margin=dict(l=0, r=80, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0EEE9", zeroline=False,
                   tickfont=dict(size=11, color="#8C8C8C"), showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color="#4A4A4A")),
    )
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View raw coefficient values"):
        raw = coef_df[["Display", "Feature", "Coefficient"]].copy()
        raw["Coefficient"] = raw["Coefficient"].apply(lambda x: f"{x:+,.4f}")
        st.dataframe(raw.rename(columns={"Display": "Name"}), use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:14px;font-weight:700;color:#141414;letter-spacing:-0.2px;margin:0 0 0.3rem 0;">Saturation Curves by Channel</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#8C8C8C;margin-bottom:1rem;">Each channel has diminishing returns — the curve flattens as spend increases. The dot marks current average weekly spend. The dashed line is the 50% saturation point.</div>', unsafe_allow_html=True)

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
            height=200, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickformat="$,.0f", tickfont=dict(size=9, color="#8C8C8C")),
            yaxis=dict(range=[0, 1.05], gridcolor="#F0EEE9",
                       tickfont=dict(size=9, color="#8C8C8C")),
        )
        with sat_cols[i % 3]:
            st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"α={params['alpha']} · γ=${params['gamma']:,} · avg ${avg_spend:,.0f}/wk")

st.markdown('</div>', unsafe_allow_html=True)