"""
3_diagnostics.py — Model validation and diagnostics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import get_X_y
from src.saturation import hill_saturation, SATURATION_DEFAULTS
from src.adstock import CHANNELS

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Diagnostics · MarketLytics", page_icon="🔬", layout="wide")

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
.page-header { border-bottom:1px solid #E8E6E1;padding-bottom:1.5rem;margin-bottom:2rem; }
.page-eyebrow { font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#9B9B9B;margin-bottom:6px; }
.page-title { font-size:28px;font-weight:600;color:#1a1a1a;letter-spacing:-0.5px;margin:0; }
.page-desc { font-size:14px;color:#6B6B6B;margin-top:6px;line-height:1.5; }
.section-title { font-size:13px;font-weight:600;color:#1a1a1a;letter-spacing:-0.2px;margin:0 0 0.35rem 0; }
.section-desc { font-size:12px;color:#9B9B9B;margin-bottom:1rem; }
[data-testid="stMetric"] { background:#FAFAF9;border:1px solid #E8E6E1;border-radius:10px;padding:1rem 1.25rem; }
[data-testid="stMetricLabel"] { font-size:11px !important;color:#9B9B9B !important;font-weight:500 !important;text-transform:uppercase;letter-spacing:0.04em; }
[data-testid="stMetricValue"] { font-size:22px !important;font-weight:600 !important;color:#1a1a1a !important;letter-spacing:-0.4px !important; }
hr { border:none;border-top:1px solid #E8E6E1;margin:2rem 0; }
.stTabs [data-baseweb="tab-list"] { gap:0;border-bottom:1px solid #E8E6E1;background:transparent; }
.stTabs [data-baseweb="tab"] { font-size:13px;font-weight:500;color:#6B6B6B;padding:0.5rem 1rem;border-bottom:2px solid transparent;background:transparent; }
.stTabs [aria-selected="true"] { color:#1a1a1a !important;border-bottom:2px solid #1a1a1a !important; }
.stCaption { font-size:11px !important;color:#9B9B9B !important; }
[data-testid="stDataFrame"] { border:1px solid #E8E6E1;border-radius:10px;overflow:hidden; }
[data-testid="stSidebarNav"] a { font-size:13px !important;color:#3D3D3D !important;font-weight:400 !important;padding:0.4rem 0.75rem !important;border-radius:6px !important; }
[data-testid="stSidebarNav"] [aria-current="page"] { background:#ECEAE4 !important;font-weight:500 !important;color:#1a1a1a !important; }
.score-card { background:#FAFAF9;border:1px solid #E8E6E1;border-radius:10px;padding:1.25rem;text-align:center; }
.score-value { font-size:28px;font-weight:600;color:#1a1a1a;letter-spacing:-0.5px; }
.score-label { font-size:11px;color:#9B9B9B;font-weight:500;text-transform:uppercase;letter-spacing:0.04em;margin-top:4px; }
.score-sub { font-size:11px;color:#6B6B6B;margin-top:6px;line-height:1.4; }
.verdict { display:inline-flex;align-items:center;gap:6px;background:#F0FDF4;
  border:1px solid #BBF7D0;border-radius:6px;padding:0.35rem 0.75rem;
  font-size:12px;color:#16A34A;font-weight:500;margin-top:0.5rem; }
.verdict.warn { background:#FFFBEB;border-color:#FDE68A;color:#D97706; }
.verdict.bad { background:#FEF2F2;border-color:#FECACA;color:#DC2626; }
</style>
"""
st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">ML</div>
  <div><div class="sidebar-brand-text">MarketLytics</div><div class="sidebar-sub">MMM Platform</div></div>
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

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-eyebrow">Model Validation</div>
  <div class="page-title">Model Diagnostics</div>
  <div class="page-desc">Validate model reliability before trusting its budget recommendations. Good MMM targets: MAPE &lt; 15%, R² &gt; 0.80.</div>
</div>
""", unsafe_allow_html=True)

# ── Score cards ───────────────────────────────────────────────────────────────
cv_mape    = artifacts["cv_mape"]
cv_r2      = artifacts["cv_r2"]
train_mape = artifacts["train_mape"]
train_r2   = artifacts["train_r2"]

mape_verdict = ("✓ Excellent", "good") if cv_mape < 0.10 else \
               ("✓ Good", "good") if cv_mape < 0.15 else \
               ("△ Acceptable", "warn") if cv_mape < 0.20 else ("✕ Needs work", "bad")
r2_verdict   = ("✓ Excellent", "good") if cv_r2 > 0.85 else \
               ("✓ Good", "good") if cv_r2 > 0.75 else \
               ("△ Acceptable", "warn") if cv_r2 > 0.60 else ("✕ Needs work", "bad")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{cv_mape:.1%}</div>
      <div class="score-label">CV MAPE</div>
      <div class="score-sub">Cross-validated error</div>
      <div class="verdict {mape_verdict[1]}">{mape_verdict[0]}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{cv_r2:.3f}</div>
      <div class="score-label">CV R²</div>
      <div class="score-sub">Variance explained</div>
      <div class="verdict {r2_verdict[1]}">{r2_verdict[0]}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{train_mape:.1%}</div>
      <div class="score-label">Train MAPE</div>
      <div class="score-sub">In-sample fit</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="score-card">
      <div class="score-value">{artifacts['n_obs']}</div>
      <div class="score-label">Observations</div>
      <div class="score-sub">Weekly data points</div>
    </div>""", unsafe_allow_html=True)

st.caption("CV metrics use 5-fold TimeSeriesSplit — they reflect real-world holdout performance, not in-sample fit.")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main diagnostic tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Fit Quality", "Coefficients", "Saturation Curves"])

with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-title">Actual vs Predicted Revenue</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Dashed line should track the solid line closely</div>', unsafe_allow_html=True)
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=df["DATE"], y=y, name="Actual",
            line=dict(color="#1a1a1a", width=2),
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
                       tickfont=dict(size=11, color="#9B9B9B")),
            xaxis=dict(tickfont=dict(size=11, color="#9B9B9B")),
            legend=dict(orientation="h", y=1.1, font=dict(size=12)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fit, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div class="section-title">Residuals Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Should be random noise around zero — no visible pattern</div>', unsafe_allow_html=True)
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=df["DATE"], y=residuals,
            marker_color=["#DC2626" if r < 0 else "#D1FAE5" for r in residuals],
            hovertemplate="%{x|%b %Y}<br>Residual: $%{y:,.0f}<extra></extra>",
        ))
        fig_res.add_hline(y=0, line_dash="dot", line_color="#9B9B9B", line_width=1)
        fig_res.update_layout(
            height=310, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0EEE9", tickformat="$,.0f",
                       tickfont=dict(size=11, color="#9B9B9B")),
            xaxis=dict(tickfont=dict(size=11, color="#9B9B9B")),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

    # Residual distribution
    st.markdown('<div class="section-title">Residual Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Should be approximately bell-shaped and centred near zero</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals, nbinsx=30, name="Residuals",
        marker_color="#1a1a1a", opacity=0.75,
    ))
    fig_hist.add_vline(x=0, line_dash="dot", line_color="#DC2626", line_width=1.5)
    fig_hist.add_vline(x=np.mean(residuals), line_dash="dash", line_color="#9B9B9B",
                       annotation_text=f"Mean: ${np.mean(residuals):,.0f}",
                       annotation_font_size=11)
    fig_hist.update_layout(
        height=240, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat="$,.0f", tickfont=dict(size=11, color="#9B9B9B")),
        yaxis=dict(gridcolor="#F0EEE9", tickfont=dict(size=11, color="#9B9B9B"),
                   title="Weeks"),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with tab2:
    st.markdown('<div class="section-title">Feature Coefficients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Larger absolute value = stronger influence on revenue prediction. Negative = suppressor effect.</div>', unsafe_allow_html=True)

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
        marker_color=["#DC2626" if v < 0 else "#1a1a1a" for v in coef_df["Coefficient"]],
        text=[f"{v:+,.0f}" for v in coef_df["Coefficient"]],
        textposition="outside",
        textfont=dict(size=11, color="#3D3D3D"),
        hovertemplate="<b>%{y}</b><br>Coef: %{x:+,.2f}<extra></extra>",
    ))
    fig_coef.add_vline(x=0, line_color="#E8E6E1", line_width=1)
    fig_coef.update_layout(
        height=420, margin=dict(l=0,r=80,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F0EEE9", zeroline=False,
                   tickfont=dict(size=11, color="#9B9B9B"), showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color="#3D3D3D")),
    )
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    st.caption("Coefficients are in scaled feature space (post StandardScaler). Compare relative magnitudes, not absolute values.")

    # Raw coefficient table
    with st.expander("View raw coefficient values"):
        raw = coef_df[["Display", "Feature", "Coefficient"]].copy()
        raw["Coefficient"] = raw["Coefficient"].apply(lambda x: f"{x:+,.4f}")
        st.dataframe(raw.rename(columns={"Display":"Name"}), use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-title">Saturation Curves by Channel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Each channel has diminishing returns — the curve flattens as spend increases. The dot marks current average weekly spend.</div>', unsafe_allow_html=True)

    sat_cols = st.columns(3)
    ch_list  = [ch for ch in CHANNELS if ch in df.columns]

    for i, ch in enumerate(ch_list):
        label       = CH_LABELS.get(ch, ch)
        params      = SATURATION_DEFAULTS.get(ch, {"alpha": 1.5, "gamma": 10000})
        spend_vals  = pd.to_numeric(df[ch], errors="coerce").dropna()
        max_spend   = spend_vals.max() * 1.5 if len(spend_vals) > 0 else 100_000
        avg_spend   = spend_vals.mean() if len(spend_vals) > 0 else 0
        x_range     = np.linspace(0, max_spend, 300)
        y_sat       = hill_saturation(x_range, params["alpha"], params["gamma"])
        y_at_avg    = hill_saturation(np.array([avg_spend]), params["alpha"], params["gamma"])[0]

        fig_sat = go.Figure()
        fig_sat.add_trace(go.Scatter(
            x=x_range, y=y_sat,
            mode="lines", line=dict(color="#1a1a1a", width=2),
            name="Saturation", showlegend=False,
            hovertemplate="Spend: $%{x:,.0f}<br>Response: %{y:.3f}<extra></extra>",
        ))
        fig_sat.add_trace(go.Scatter(
            x=[avg_spend], y=[y_at_avg],
            mode="markers", marker=dict(color="#1a1a1a", size=8),
            name="Current avg", showlegend=False,
            hovertemplate=f"Avg spend: ${avg_spend:,.0f}<br>Response: {y_at_avg:.3f}<extra></extra>",
        ))
        fig_sat.add_hline(y=0.5, line_dash="dot", line_color="#E8E6E1", line_width=1)
        fig_sat.update_layout(
            title=dict(text=label, font=dict(size=12, color="#3D3D3D"), x=0),
            height=200, margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickformat="$,.0f", tickfont=dict(size=9, color="#9B9B9B")),
            yaxis=dict(range=[0,1.05], gridcolor="#F0EEE9",
                       tickfont=dict(size=9, color="#9B9B9B")),
        )
        with sat_cols[i % 3]:
            st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"α={params['alpha']} · γ=${params['gamma']:,} · avg spend ${avg_spend:,.0f}")