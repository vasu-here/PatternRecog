"""Temperature scaling fitted on a calibration holdout, never on test data."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _temperature_nll(log_temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Numerically stable negative log likelihood at one temperature."""
    temperature = float(np.exp(log_temperature))
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))


def fit_temperature(log_probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Choose one positive temperature by minimizing calibration-set log loss."""
    # Log probabilities preserve each model's relative class scores while
    # allowing the calibration set to soften (T > 1) or sharpen (T < 1) them.
    result = minimize_scalar(
        _temperature_nll,
        bounds=(np.log(0.05), np.log(20.0)),
        args=(log_probabilities, labels),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(np.exp(result.x))


def apply_temperature(log_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Convert log scores into normalized, temperature-scaled probabilities."""
    scaled = log_probabilities / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
