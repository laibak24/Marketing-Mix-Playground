"""
optimizer.py
Budget reallocation optimizer using scipy.
Given a total budget, finds the spend split across channels
that maximises predicted revenue.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.adstock import CHANNELS, DECAY_DEFAULTS
from src.saturation import SATURATION_DEFAULTS, hill_saturation
from src.features import FEATURE_COLS


def predict_revenue_from_spend(
    spend_dict: dict,
    artifacts: dict,
    sat_params: dict | None = None,
) -> float:
    """
    Predict weekly revenue for a given spend allocation dict.
    spend_dict: {channel: weekly_spend_$}
    """
    sat_params   = sat_params or SATURATION_DEFAULTS
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    x = np.zeros((1, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        for ch in spend_dict:
            if col == f"{ch}_saturated":
                p      = sat_params.get(ch, {"alpha": 1.5, "gamma": 10_000})
                sat    = hill_saturation(
                    np.array([spend_dict[ch]]), p["alpha"], p["gamma"]
                )[0]
                x[0, i] = sat
    return float(model.predict(scaler.transform(x))[0])


def optimize_budget(
    total_budget: float,
    artifacts: dict,
    channels: list | None = None,
    sat_params: dict | None = None,
    min_pct: float = 0.02,      # each channel gets at least 2% of budget
) -> dict:
    """
    Find the spend allocation that maximises predicted revenue.

    Args:
        total_budget: Total weekly budget in $.
        artifacts:    Loaded model artifacts dict.
        channels:     List of channel names to optimise over.
        sat_params:   Saturation parameters per channel.
        min_pct:      Minimum budget fraction per channel (avoids zeroing out channels).

    Returns:
        Dict of {channel: optimal_spend}.
    """
    channels   = channels or CHANNELS
    sat_params = sat_params or SATURATION_DEFAULTS
    n          = len(channels)
    min_spend  = total_budget * min_pct

    def neg_revenue(allocs):
        spend = dict(zip(channels, allocs))
        return -predict_revenue_from_spend(spend, artifacts, sat_params)

    x0          = [total_budget / n] * n
    bounds      = [(min_spend, total_budget)] * n
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - total_budget}]

    result = minimize(
        neg_revenue, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    optimal = dict(zip(channels, result.x))
    optimal["_predicted_revenue"] = -result.fun
    optimal["_success"]           = result.success
    return optimal