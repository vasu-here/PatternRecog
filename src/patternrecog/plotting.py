"""Small publication-ready figures created from saved experiment predictions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_reliability_plot(path: Path, y_true, raw_probabilities, calibrated_probabilities, title: str):
    """Draw raw and calibrated top-label reliability curves."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="Perfect calibration")
    for probabilities, label in ((raw_probabilities, "Raw"), (calibrated_probabilities, "Temperature scaled")):
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == y_true
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < 9 else confidence <= 1)
            if mask.any():
                xs.append(confidence[mask].mean())
                ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Observed accuracy", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_plot(path: Path, y_true, predicted, class_names, title: str):
    """Save row-normalized confusion matrix so class recall is easy to compare."""
    matrix = confusion_matrix(y_true, predicted, labels=np.arange(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted newsgroup")
    ax.set_ylabel("Actual newsgroup")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
