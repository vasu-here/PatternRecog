"""Dataset loading and leakage-safe calibration split helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TextData:
    """The official train/test messages and their integer class labels."""

    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    class_names: list[str]


def load_dataset(remove_metadata: bool = True) -> TextData:
    """Fetch the canonical 20 Newsgroups split and optionally remove metadata.

    Headers, quoted replies, and signatures can reveal the newsgroup without
    teaching a model about message content. By default all three are removed.
    """
    remove = ("headers", "footers", "quotes") if remove_metadata else ()
    # Keep the dataset cache within the project. This makes first-run downloads
    # predictable on restricted machines and keeps downloaded files Git-ignored.
    data_home = Path(__file__).resolve().parents[2] / "data" / "scikit-learn"
    train = fetch_20newsgroups(
        data_home=data_home, subset="train", remove=remove, shuffle=True, random_state=42
    )
    test = fetch_20newsgroups(
        data_home=data_home, subset="test", remove=remove, shuffle=True, random_state=42
    )
    return TextData(
        train_texts=list(train.data),
        train_labels=np.asarray(train.target, dtype=np.int64),
        test_texts=list(test.data),
        test_labels=np.asarray(test.target, dtype=np.int64),
        class_names=list(train.target_names),
    )


def split_training_data(
    texts: list[str], labels: np.ndarray, test_size: float, random_state: int
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Reserve a stratified slice of training data for probability calibration."""
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
