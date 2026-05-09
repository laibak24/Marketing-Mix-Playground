# Marketing Mix Model Dashboard

A channel attribution + budget optimiser built on the Robyn open-source dataset.
Shows ROI per media channel and lets users drag sliders to reallocate budget and see predicted revenue change.

## Channels modelled
`tv_S` · `ooh_S` · `print_S` · `facebook_S` · `search_S` · `newsletter`

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/laibak24/Marketing-Mix-Playground
cd mmm-dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your data
cp /path/to/weekly_media_data.csv data/raw/weekly_media_data.csv

# 3. Train the model
python -m src.model

# 4. Run the dashboard
streamlit run app/main.py
```

## Project structure
```
mmm-dashboard/
├── data/raw/               ← Robyn weekly CSV (not committed)
├── src/
│   ├── adstock.py          ← Geometric decay transform
│   ├── saturation.py       ← Hill saturation curves
│   ├── features.py         ← Full feature pipeline
│   ├── model.py            ← Ridge regression + CV training
│   ├── attribution.py      ← Channel ROI + contribution %
│   └── optimizer.py        ← Scipy budget optimizer
├── app/
│   ├── main.py             ← Streamlit entry point
│   └── pages/
│       ├── 1_overview.py   ← ROI charts, waterfall
│       ├── 2_playground.py ← Budget sliders + live prediction
│       └── 3_diagnostics.py ← Model fit, residuals
├── models/                 ← Saved .joblib (not committed)
└── requirements.txt
```

## Key modelling decisions
- **Adstock** — Geometric decay per channel (TV: 0.7, digital: 0.2–0.3)
- **Saturation** — Hill function captures diminishing returns
- **Validation** — TimeSeriesSplit (5-fold), never random shuffle
- **Model** — Ridge regression (L2 regularisation prevents coefficient blowup)