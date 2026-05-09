"""
attribution.py
Channel-level revenue attribution and ROI calculation.
"""
import numpy as np
import pandas as pd

from src.features import build_features, FEATURE_COLS
from src.adstock import CHANNELS


def channel_contributions(artifacts: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose predicted revenue into per-channel contributions.

    Uses RAW (unscaled) saturated values × model coefficients to avoid the
    StandardScaler mean-centering problem that collapses contributions to zero.
    The raw shares are then rescaled to sum to total attributed media revenue.
    """
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    df_feat = build_features(df)
    X_raw   = df_feat[feature_cols].values.astype(float)   # ← unscaled

    # Recover coefficient magnitudes in original feature units.
    # model.coef_ lives in scaled space; dividing by scaler.scale_ converts
    # each coefficient back to "revenue per unit of raw feature".
    coefs_raw = model.coef_ / scaler.scale_                # shape (n_features,)

    # Raw contributions: always ≥ 0 for saturated channels (Hill outputs [0,1])
    contribs_raw = X_raw * coefs_raw                       # (n_weeks, n_features)

    # --- channel-level aggregation ---
    sat_cols  = [f"{ch}_saturated" for ch in CHANNELS]
    records   = []
    for i, col in enumerate(feature_cols):
        if col in sat_cols:
            records.append({
                "feature":      col,
                "contribution": float(contribs_raw[:, i].sum()),
            })

    ch_result = pd.DataFrame(records)
    ch_result["channel"] = ch_result["feature"].str.replace(
        "_saturated", "", regex=False
    )

    # Clip negatives (safety net; shouldn't trigger with Hill saturation)
    ch_result["contribution"] = ch_result["contribution"].clip(lower=0)

    # Scale raw shares to match total predicted (attributed) revenue.
    # This converts arbitrary coefficient-unit sums into real revenue dollars.
    X_scaled       = scaler.transform(X_raw)
    y_pred         = model.predict(X_scaled)
    intercept_rev  = model.intercept_                      # baseline / non-media
    media_rev      = max(float(y_pred.sum()) - intercept_rev, 0.0)

    raw_total = ch_result["contribution"].sum()
    scale_factor = (media_rev / raw_total) if raw_total > 0 else 1.0
    ch_result["contribution"] = ch_result["contribution"] * scale_factor

    total = ch_result["contribution"].sum()
    ch_result["contribution_pct"] = (
        ch_result["contribution"] / total * 100 if total > 0 else 0.0
    )

    return ch_result[["channel", "contribution", "contribution_pct"]].reset_index(
        drop=True
    )


def channel_roi(df: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """
    ROI = attributed revenue / total spend per channel.
    """
    records = []
    for _, row in contributions.iterrows():
        ch        = row["channel"]
        spend_col = ch + "_S" if not ch.endswith("_S") else ch
        if spend_col not in df.columns:
            spend_col = ch
        if spend_col not in df.columns:
            continue

        total_spend = pd.to_numeric(df[spend_col], errors="coerce").sum()
        attr_rev    = row["contribution"]
        roi         = attr_rev / total_spend if total_spend > 0 else 0.0

        records.append({
            "channel":            ch,
            "total_spend":        round(total_spend, 0),
            "attributed_revenue": round(attr_rev, 0),
            "roi":                round(roi, 3),
            "contribution_pct":   round(row["contribution_pct"], 1),
        })

    return pd.DataFrame(records).sort_values("roi", ascending=False)