"""
model.py
Train Ridge regression MMM, save artifacts to models/mmm_model.joblib.

Run: python -m src.model
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.features import get_X_y, FEATURE_COLS, build_features

MODEL_PATH = Path("models/mmm_model.joblib")
DATA_PATH  = Path("data/raw/weekly_media_data.csv")


def train(df: pd.DataFrame, target: str = "revenue", alpha: float = 10.0) -> dict:
    X, y, feature_cols = get_X_y(df, target)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series cross-validation (never shuffle time series!)
    tscv  = TimeSeriesSplit(n_splits=5)
    mapes = []
    r2s   = []
    for train_idx, val_idx in tscv.split(X_scaled):
        m = Ridge(alpha=alpha)
        m.fit(X_scaled[train_idx], y[train_idx])
        preds = m.predict(X_scaled[val_idx])
        mapes.append(mean_absolute_percentage_error(y[val_idx], preds))
        r2s.append(r2_score(y[val_idx], preds))

    # Final model on all data
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    full_preds = model.predict(X_scaled)

    artifacts = {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "cv_mape":      float(np.mean(mapes)),
        "cv_r2":        float(np.mean(r2s)),
        "train_mape":   float(mean_absolute_percentage_error(y, full_preds)),
        "train_r2":     float(r2_score(y, full_preds)),
        "coefficients": dict(zip(feature_cols, model.coef_)),
        "intercept":    float(model.intercept_),
        "n_obs":        len(y),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)

    print(f"\nModel trained on {len(y)} observations")
    print(f"  CV  MAPE : {artifacts['cv_mape']:.1%}  |  CV  R2 : {artifacts['cv_r2']:.3f}")
    print(f"  Train MAPE: {artifacts['train_mape']:.1%}  |  Train R2: {artifacts['train_r2']:.3f}")
    print(f"\nCoefficients:")
    for col, coef in artifacts["coefficients"].items():
        print(f"  {col:<30} {coef:+.4f}")
    print(f"\nSaved → {MODEL_PATH}")
    return artifacts


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train(df)