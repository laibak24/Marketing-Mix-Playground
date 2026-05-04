"""
adstock.py
Geometric adstock (carryover) transformation.

Adstock captures the lagged effect of advertising —
TV spend today still drives revenue 3 weeks from now.
Higher decay = longer carryover. Typical values:
  TV:       0.6–0.8
  Radio:    0.4–0.6
  Digital:  0.2–0.4
  Email:    0.1–0.2
"""
import numpy as np
import pandas as pd

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"]

# Decay rates: TV/OOH/print carry longer; digital decays fast
DECAY_DEFAULTS = {
    "tv_S":       0.70,
    "ooh_S":      0.50,
    "print_S":    0.60,
    "facebook_S": 0.25,
    "search_S":   0.20,
    "newsletter": 0.10,
}


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """
    Apply geometric adstock to a 1-D spend array.

    Args:
        x:       Weekly spend values (array of floats).
        decay:   Carryover rate per period [0, 1).
        max_lag: Number of weeks to carry effect forward.

    Returns:
        Transformed array, same length as x.
    """
    if not 0 <= decay < 1:
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    weights = np.array([decay ** i for i in range(max_lag)])
    weights /= weights.sum()                     # normalise so total effect = 1
    return np.convolve(x, weights, mode="full")[: len(x)]


def apply_adstock(
    df: pd.DataFrame,
    channels: list | None = None,
    decay_params: dict | None = None,
) -> pd.DataFrame:
    """
    Apply adstock to each channel column, adding *_adstocked columns.

    Args:
        df:           DataFrame with channel spend columns.
        channels:     List of channel names (defaults to CHANNELS).
        decay_params: Dict of {channel: decay}. Defaults to DECAY_DEFAULTS.

    Returns:
        Copy of df with extra *_adstocked columns.
    """
    channels = channels or CHANNELS
    decay    = decay_params or DECAY_DEFAULTS
    result   = df.copy()
    for ch in channels:
        if ch not in df.columns:
            raise KeyError(f"Column '{ch}' not found in DataFrame.")
        result[f"{ch}_adstocked"] = geometric_adstock(df[ch].values, decay[ch])
    return result