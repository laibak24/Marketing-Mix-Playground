"""
features.py — Feature engineering pipeline for Marketing Mix Model
Transforms raw Robyn weekly spend data into a model-ready feature matrix.

Pipeline order per channel:
  raw spend → geometric adstock → Hill saturation → scaled feature

Also adds:
  - Linear trend
  - Sine/cosine seasonality (annual + semi-annual)
  - Control variables (competitor sales, events, newsletter) passed through as-is

Outputs:
  data/processed/X_train.csv         ← feature matrix (no target column)
  data/processed/feature_metadata.json ← column names, params, target col name
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── adjust this import path if you run from project root ──────────────────────
from src.adstock import apply_adstock, DECAY_DEFAULTS
from src.saturation import hill_saturation, SATURATION_DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# 1.  COLUMN CONFIGURATION
#     Edit these if your CSV uses different column names.
# ─────────────────────────────────────────────────────────────────────────────

# Maps the internal channel key (must match adstock.py + saturation.py dicts)
# to the actual CSV column name for that channel's spend.
CHANNEL_COLUMN_MAP: dict[str, str] = {
    "tv":       "tv_S",
    "facebook": "facebook_S",
    "google":   "search_S",   # Robyn calls it search_S; we treat it as google
    "email":    "newsletter",
}

# Columns passed through to the model without any transformation.
# Set to [] if your dataset has none of these.
CONTROL_COLUMNS: list[str] = [
    "competitor_sales_B",
    "events",
]

DATE_COLUMN:   str = "DATE"
TARGET_COLUMN: str = "revenue"       # y — not included in X_train.csv


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SATURATION WRAPPER
#     apply_adstock() is already defined in adstock.py.
#     We need a matching apply_saturation() that mirrors its interface.
# ─────────────────────────────────────────────────────────────────────────────

def apply_saturation(
    df: pd.DataFrame,
    channels: list[str],
    saturation_params: dict | None = None,
) -> pd.DataFrame:
    """
    Apply Hill saturation to each channel's adstocked column.

    Reads  <ch>_adstocked  →  writes  <ch>_saturated.
    Returns a copy of df with the new columns appended.
    """
    params = saturation_params or SATURATION_DEFAULTS
    result = df.copy()
    for ch in channels:
        src_col = f"{ch}_adstocked"
        if src_col not in result.columns:
            raise KeyError(
                f"Column '{src_col}' not found. "
                "Run apply_adstock() before apply_saturation()."
            )
        p = params[ch]
        result[f"{ch}_saturated"] = hill_saturation(
            result[src_col].values,
            alpha=p["alpha"],
            gamma=p["gamma"],
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised linear trend: 0 → 1 over the full date range."""
    result = df.copy()
    n = len(result)
    result["trend"] = np.linspace(0, 1, n)
    return result


def add_seasonality(df: pd.DataFrame, date_col: str = DATE_COLUMN) -> pd.DataFrame:
    """
    Fourier-basis seasonality using week-of-year.
    Two harmonics: annual (period=52) and semi-annual (period=26).
    Gives the model a smooth, learnable seasonal curve.
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])
    week  = dates.dt.isocalendar().week.astype(float)

    result["sin_annual"]       = np.sin(2 * np.pi * week / 52)
    result["cos_annual"]       = np.cos(2 * np.pi * week / 52)
    result["sin_semi_annual"]  = np.sin(2 * np.pi * week / 26)
    result["cos_semi_annual"]  = np.cos(2 * np.pi * week / 26)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    decay_params:      dict | None = None,
    saturation_params: dict | None = None,
    scale_features:    bool = True,
    date_col:  str = DATE_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df                : Raw weekly data DataFrame (must contain date, target,
                        spend columns, and any control columns).
    channels          : List of channel keys. Defaults to CHANNEL_COLUMN_MAP keys.
    decay_params      : Override adstock decay per channel.
    saturation_params : Override Hill saturation params per channel.
    scale_features    : If True, StandardScaler is applied to spend features
                        (but NOT to trend/seasonality, which are already in [–1,1]).
    date_col          : Name of the date column.
    target_col        : Name of the revenue/KPI column (excluded from X).

    Returns
    -------
    X           : Feature DataFrame (ready for Ridge regression).
    y           : Target Series.
    metadata    : Dict with column lists, params, scaler stats — for attribution
                  and the Streamlit playground.
    """
    channels = channels or list(CHANNEL_COLUMN_MAP.keys())

    # ── Rename spend columns to the internal channel key ─────────────────────
    rename_map = {v: k for k, v in CHANNEL_COLUMN_MAP.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    # ── Validate required columns are present ────────────────────────────────
    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        raise ValueError(
            f"Missing spend columns for channels: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Check CHANNEL_COLUMN_MAP in features.py."
        )

    # ── Step 1: Geometric adstock ─────────────────────────────────────────────
    df = apply_adstock(df, channels, decay_params)

    # ── Step 2: Hill saturation ───────────────────────────────────────────────
    df = apply_saturation(df, channels, saturation_params)

    # ── Step 3: Temporal features ─────────────────────────────────────────────
    df = add_trend(df)
    df = add_seasonality(df, date_col)

    # ── Step 4: Assemble feature matrix ───────────────────────────────────────
    spend_features     = [f"{ch}_saturated" for ch in channels]
    temporal_features  = ["trend", "sin_annual", "cos_annual",
                          "sin_semi_annual", "cos_semi_annual"]
    control_features   = [c for c in CONTROL_COLUMNS if c in df.columns]

    feature_cols = spend_features + temporal_features + control_features

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ── Step 5: Scale spend + control features ────────────────────────────────
    scaler_stats: dict = {}
    if scale_features:
        cols_to_scale = spend_features + control_features
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        scaler_stats = {
            "scaled_columns": cols_to_scale,
            "mean_":  dict(zip(cols_to_scale, scaler.mean_.tolist())),
            "scale_": dict(zip(cols_to_scale, scaler.scale_.tolist())),
        }

    # ── Step 6: Build metadata dict ───────────────────────────────────────────
    metadata = {
        "channels":            channels,
        "channel_column_map":  CHANNEL_COLUMN_MAP,
        "spend_features":      spend_features,
        "temporal_features":   temporal_features,
        "control_features":    control_features,
        "all_feature_columns": feature_cols,
        "target_column":       target_col,
        "date_column":         date_col,
        "n_rows":              int(len(X)),
        "decay_params":        decay_params or DECAY_DEFAULTS,
        "saturation_params":   saturation_params or SATURATION_DEFAULTS,
        "scaled":              scale_features,
        "scaler_stats":        scaler_stats,
    }

    return X, y, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: dict,
    out_dir: str = "data/processed",
) -> None:
    """
    Write X_train.csv, y_train.csv, and feature_metadata.json to out_dir.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(out_dir, exist_ok=True)

    X.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "y_train.csv"), index=False, header=True)

    with open(os.path.join(out_dir, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[features.py] Saved {X.shape[0]} rows × {X.shape[1]} features → {out_dir}/")
    print(f"[features.py] Feature columns: {X.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENTRY POINT  (python -m src.features  OR  python src/features.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    raw_path = "data/raw/weekly_media_data.csv"
    if not os.path.exists(raw_path):
        print(f"[features.py] ERROR: could not find {raw_path}")
        print("  Place the Robyn CSV there and re-run.")
        sys.exit(1)

    print(f"[features.py] Loading {raw_path} …")
    raw_df = pd.read_csv(raw_path, parse_dates=[DATE_COLUMN])
    print(f"[features.py] Raw shape: {raw_df.shape}")
    print(f"[features.py] Columns:   {raw_df.columns.tolist()}")

    X, y, meta = build_features(raw_df)
    save_processed(X, y, meta)

    print("\n[features.py] Sample of X_train (first 3 rows):")
    print(X.head(3).to_string())
    print(f"\n[features.py] y stats: min={y.min():.0f}  max={y.max():.0f}  mean={y.mean():.0f}")