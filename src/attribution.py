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
    Method: marginal contribution — each channel's saturated spend
    times its model coefficient (in original revenue scale).
    """
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    df_feat  = build_features(df)
    X        = df_feat[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)

    # Contribution of each feature = X_scaled * coef (mean across weeks)
    coefs   = model.coef_
    contribs = X_scaled * coefs          # shape (n_weeks, n_features)

    # Scale back to revenue units
    # StandardScaler mean/std are on the feature side; coef is in scaled space.
    # Predicted revenue = intercept + sum(X_scaled * coef)
    # So each column of X_scaled * coef IS already in revenue-unit contribution.
    records = []
    for i, col in enumerate(feature_cols):
        records.append({
            "feature":     col,
            "contribution": float(contribs[:, i].sum()),   # total over all weeks
        })

    result = pd.DataFrame(records)
    # Keep only channel (saturated) features for the UI
    sat_cols = [f"{ch}_saturated" for ch in CHANNELS]
    ch_result = result[result["feature"].isin(sat_cols)].copy()
    ch_result["channel"] = ch_result["feature"].str.replace("_saturated", "", regex=False)
    ch_result["contribution"] = ch_result["contribution"].clip(lower=0)
    total = ch_result["contribution"].sum()
    ch_result["contribution_pct"] = (
        ch_result["contribution"] / total * 100 if total > 0 else 0
    )
    return ch_result[["channel", "contribution", "contribution_pct"]].reset_index(drop=True)


def channel_roi(df: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ROI = attributed revenue / total spend per channel.
    """
    records = []
    for _, row in contributions.iterrows():
        ch = row["channel"]
        spend_col = ch + "_S" if not ch.endswith("_S") else ch
        # Handle newsletter (no _S suffix)
        if spend_col not in df.columns:
            spend_col = ch
        if spend_col not in df.columns:
            continue
        total_spend = pd.to_numeric(df[spend_col], errors="coerce").sum()
        attr_rev    = row["contribution"]
        roi         = attr_rev / total_spend if total_spend > 0 else 0
        records.append({
            "channel":            ch,
            "total_spend":        round(total_spend, 0),
            "attributed_revenue": round(attr_rev, 0),
            "roi":                round(roi, 3),
            "contribution_pct":   round(row["contribution_pct"], 1),
        })
    return pd.DataFrame(records).sort_values("roi", ascending=False)