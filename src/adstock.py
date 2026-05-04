import numpy as np
import pandas as pd

def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """
    Apply geometric adstock transformation.
    decay: 0 = no carryover, 0.9 = heavy carryover (TV ~0.7, digital ~0.3)
    """
    weights = np.array([decay ** i for i in range(max_lag)])
    weights /= weights.sum()
    return np.convolve(x, weights, mode='full')[:len(x)]

DECAY_DEFAULTS = {
    "google":   0.3,
    "facebook": 0.25,
    "email":    0.1,
    "tv":       0.7,
}

def apply_adstock(df: pd.DataFrame, channels: list, 
                  decay_params: dict = None) -> pd.DataFrame:
    decay = decay_params or DECAY_DEFAULTS
    result = df.copy()
    for ch in channels:
        result[f"{ch}_adstocked"] = geometric_adstock(df[ch].values, decay[ch])
    return result