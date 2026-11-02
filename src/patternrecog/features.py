"""Bag-of-words feature recipes used in the controlled comparison."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(frozen=True)
class FeatureRecipe:
    """Human-readable feature choice and its vectorizer settings."""

    name: str
    description: str
    kind: str
    ngram_range: tuple[int, int] = (1, 1)
    max_features: int = 1500


RECIPES = (
    FeatureRecipe("count_unigram", "Unigram term counts", "count"),
    FeatureRecipe("binary_unigram", "Unigram presence/absence", "binary"),
    FeatureRecipe("tfidf_unigram", "Unigram TF-IDF weights", "tfidf"),
    FeatureRecipe("count_1_2gram", "Unigram and bigram term counts", "count", (1, 2)),
)


def make_vectorizer(recipe: FeatureRecipe):
    """Create a fresh vectorizer; it must be fitted on training text only."""
    common = dict(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=recipe.ngram_range,
        max_features=recipe.max_features,
        min_df=2,
        dtype=float,
    )
    if recipe.kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, norm="l2", **common)
    return CountVectorizer(binary=(recipe.kind == "binary"), **common)
