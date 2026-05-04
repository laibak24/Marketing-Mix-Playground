"""
main.py — Streamlit entry point.
Run: streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title="Marketing Mix Model | MarketLytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📊 MMM Dashboard")
st.sidebar.caption("Marketing Mix Model — Robyn Dataset")

st.title("Marketing Mix Model")
st.markdown("""
Welcome to the **MarketLytics MMM Dashboard**.  
Use the sidebar to navigate between pages.

| Page | What it shows |
|------|--------------|
| 📈 Overview | Channel ROI, revenue waterfall, spend vs revenue |
| 🎛️ Budget Playground | Drag sliders to reallocate spend, see predicted revenue |
| 🔬 Diagnostics | Model fit, residuals, cross-validation scores |
""")
st.info("👈 Select a page from the sidebar to get started.")