import numpy as np

def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill function saturation.
    alpha: shape (steepness), gamma: half-saturation point
    Returns values in [0, 1] range — multiply by channel coefficient.
    """
    return x ** alpha / (x ** alpha + gamma ** alpha)

def log_saturation(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Simpler log curve — faster to tune, less expressive."""
    return np.log1p(x * scale) / np.log1p(x.max() * scale)

SATURATION_DEFAULTS = {
    "google":   {"alpha": 2.0, "gamma": 20000},
    "facebook": {"alpha": 1.5, "gamma": 15000},
    "email":    {"alpha": 1.2, "gamma": 5000},
    "tv":       {"alpha": 2.5, "gamma": 80000},
}