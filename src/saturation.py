"""
saturation.py
Saturation (diminishing returns) curves for ad spend.

Captures the idea that the 1st $1,000 on Google drives more revenue
than the 1,001st $1,000. Two options:
  - Hill function: flexible S-curve (preferred for MMM)
  - Log saturation: simpler, good default
"""
import numpy as np

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"]

# gamma = ~median weekly spend per channel (tune after EDA)
SATURATION_DEFAULTS = {
    "tv_S":       {"alpha": 2.5, "gamma": 50_000},
    "ooh_S":      {"alpha": 2.0, "gamma": 80_000},
    "print_S":    {"alpha": 1.8, "gamma": 30_000},
    "facebook_S": {"alpha": 1.5, "gamma":  5_000},
    "search_S":   {"alpha": 2.0, "gamma": 10_000},
    "newsletter": {"alpha": 1.2, "gamma":  8_000},
}


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill (sigmoid) saturation function.

    f(x) = x^alpha / (x^alpha + gamma^alpha)

    Args:
        x:     Adstocked spend values (non-negative).
        alpha: Shape parameter — higher = steeper S-curve. Typical: 1.2–3.0.
        gamma: Half-saturation point — spend level at which response = 0.5.
               Set near the median spend for that channel.

    Returns:
        Values in [0, 1]. Multiply by channel revenue coefficient to scale.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    x = np.asarray(x, dtype=float)
    return x ** alpha / (x ** alpha + gamma ** alpha)


def log_saturation(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Log-based saturation — simpler alternative to Hill.

    Returns values in [0, 1], normalised to max of x.
    """
    x = np.asarray(x, dtype=float)
    if x.max() == 0:
        return np.zeros_like(x)
    return np.log1p(x * scale) / np.log1p(x.max() * scale)


def apply_saturation(
    df,
    channels: list | None = None,
    sat_params: dict | None = None,
    method: str = "hill",
):
    """
    Apply saturation to *_adstocked columns, adding *_saturated columns.

    Args:
        df:         DataFrame with {channel}_adstocked columns.
        channels:   List of channel names.
        sat_params: Dict of {channel: {alpha, gamma}}. Defaults to SATURATION_DEFAULTS.
        method:     "hill" or "log".

    Returns:
        Copy of df with extra *_saturated columns.
    """
    channels  = channels or CHANNELS
    sat_params = sat_params or SATURATION_DEFAULTS
    result    = df.copy()

    for ch in channels:
        col = f"{ch}_adstocked"
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found — run apply_adstock first.")
        x = df[col].values
        if method == "hill":
            p = sat_params[ch]
            result[f"{ch}_saturated"] = hill_saturation(x, p["alpha"], p["gamma"])
        elif method == "log":
            result[f"{ch}_saturated"] = log_saturation(x)
        else:
            raise ValueError(f"method must be 'hill' or 'log', got '{method}'")

    return result