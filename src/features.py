"""
features.py
Full feature pipeline for the Robyn dataset.

Paid channels (adstocked + saturated):
    tv_S, ooh_S, print_S, facebook_S, search_S, newsletter

Control variables (used as-is):
    facebook_I        — Facebook impressions (reach signal)
    search_clicks_P   — Paid search clicks
    competitor_sales_B — Competitor sales index
    events            — Event flag (na -> 0, numeric kept)

Trend + seasonality:
    trend, sin_52, cos_52
"""
import numpy as np
import pandas as pd

from src.adstock import apply_adstock, CHANNELS, DECAY_DEFAULTS
from src.saturation import apply_saturation, SATURATION_DEFAULTS

CONTROL_COLS = ["facebook_I", "search_clicks_P", "competitor_sales_B"]

FEATURE_COLS = (
    [f"{ch}_saturated" for ch in CHANNELS]
    + CONTROL_COLS
    + ["trend", "sin_52", "cos_52"]
)


def _clean_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the 'na' strings and missing values in control columns."""
    df = df.copy()

    # events column: 'na' -> 0, anything else -> 1 (binary flag)
    if "events" in df.columns:
        df["events_flag"] = df["events"].apply(
            lambda x: 0 if str(x).strip().lower() in ("na", "nan", "") else 1
        )

    # Numeric controls: coerce and fill NaN with column median
    for col in CONTROL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0  # column absent -- zero-fill

    return df


def build_features(
    df: pd.DataFrame,
    channels=None,
    decay_params=None,
    sat_params=None,
    date_col: str = "DATE",
) -> pd.DataFrame:
    channels = channels or CHANNELS
    df       = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df       = df.sort_values(date_col).reset_index(drop=True)

    # Fill any NaN spend with 0
    for ch in channels:
        if ch in df.columns:
            df[ch] = pd.to_numeric(df[ch], errors="coerce").fillna(0)

    df = _clean_controls(df)
    df = apply_adstock(df, channels, decay_params or DECAY_DEFAULTS)
    df = apply_saturation(df, channels, sat_params or SATURATION_DEFAULTS)

    df["trend"]  = np.arange(len(df))
    df["sin_52"] = np.sin(2 * np.pi * df["trend"] / 52)
    df["cos_52"] = np.cos(2 * np.pi * df["trend"] / 52)

    return df


def get_X_y(df: pd.DataFrame, target: str = "revenue"):
    df_feat = build_features(df)
    X = df_feat[FEATURE_COLS].values.astype(float)
    y = df_feat[target].values.astype(float)
    return X, y, FEATURE_COLS