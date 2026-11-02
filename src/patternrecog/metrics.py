"""Classification and probability-quality metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """Mean squared error between class probabilities and one-hot labels."""
    one_hot = np.eye(n_classes, dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    """Top-label ECE: confidence/accuracy gap across equal-width bins."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        # Include confidence 1.0 in the final bin.
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return ece


def score_predictions(y_true, predicted, probabilities, n_classes: int) -> dict[str, float]:
    """Return comparable metrics; lower probability-loss/calibration is better."""
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
        "brier_score": multiclass_brier(y_true, probabilities, n_classes),
        "ece_15_bins": expected_calibration_error(y_true, probabilities),
    }
