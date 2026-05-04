"""
3_diagnostics.py — Model fit diagnostics and validation metrics.
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

from src.features import build_features, get_X_y, FEATURE_COLS

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")

st.set_page_config(page_title="Diagnostics | MMM", page_icon="🔬", layout="wide")
st.title("🔬 Model Diagnostics")

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

model        = artifacts["model"]
scaler       = artifacts["scaler"]
feature_cols = artifacts["feature_cols"]

X, y, _ = get_X_y(df)
X_scaled = scaler.transform(X)
y_pred   = model.predict(X_scaled)
residuals = y - y_pred

# ── Model metrics ──────────────────────────────────────────────────────────────
st.subheader("Model Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Train MAPE",   f"{artifacts['train_mape']:.1%}")
m2.metric("CV MAPE",      f"{artifacts['cv_mape']:.1%}",
          help="5-fold TimeSeriesSplit — closer to real-world performance")
m3.metric("Train R²",     f"{artifacts['train_r2']:.3f}")
m4.metric("CV R²",        f"{artifacts['cv_r2']:.3f}")

st.caption(
    "**Good MMM targets:** MAPE < 15%, R² > 0.80.  "
    "CV metrics (TimeSeriesSplit) matter more than train metrics."
)

st.divider()

# ── Actual vs Predicted ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual vs Predicted Revenue")
    df_plot = df[["DATE"]].copy()
    df_plot["Actual"]    = y
    df_plot["Predicted"] = y_pred

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_plot["DATE"], y=df_plot["Actual"],
        name="Actual", line=dict(color="#1e40af", width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_plot["DATE"], y=df_plot["Predicted"],
        name="Predicted", line=dict(color="#f59e0b", width=2, dash="dash")
    ))
    fig_ts.update_layout(height=340, yaxis_title="Revenue ($)",
                         legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("Residuals Over Time")
    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(
        x=df["DATE"], y=residuals,
        marker_color=["#ef4444" if r < 0 else "#22c55e" for r in residuals],
        name="Residual",
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_res.update_layout(height=340, yaxis_title="Residual ($)",
                          showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)

# ── Feature coefficients ───────────────────────────────────────────────────────
st.subheader("Model Coefficients (Scaled Feature Space)")
coef_df = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": model.coef_,
}).sort_values("coefficient", key=abs, ascending=True)

fig_coef = px.bar(
    coef_df, x="coefficient", y="feature", orientation="h",
    color="coefficient",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels={"coefficient": "Coefficient", "feature": ""},
)
fig_coef.update_layout(height=400, coloraxis_showscale=False)
st.plotly_chart(fig_coef, use_container_width=True)

# ── Residual distribution ──────────────────────────────────────────────────────
st.subheader("Residual Distribution")
fig_hist = px.histogram(
    x=residuals, nbins=30,
    labels={"x": "Residual ($)", "count": "Weeks"},
    color_discrete_sequence=["#6366f1"],
)
fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
fig_hist.update_layout(height=300)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption("Residuals should be roughly centred at 0 and normally distributed.")