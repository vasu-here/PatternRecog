"""End-to-end, reproducible feature/model/calibration experiment."""

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB

from .calibration import apply_temperature, fit_temperature
from .data import load_dataset, split_training_data
from .features import RECIPES, make_vectorizer
from .metrics import score_predictions
from .plotting import save_confusion_plot, save_reliability_plot


def _models():
    """Construct the assignment's two model families with documented settings."""
    return {
        "fisher_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "multinomial_nb": MultinomialNB(alpha=0.5),
    }


def run_experiments(
    output_dir: Path,
    remove_metadata: bool = True,
    calibration_fraction: float = 0.20,
    random_state: int = 42,
    selected_recipes: set[str] | None = None,
) -> list[dict]:
    """Train all requested combinations, calibrate on held-out train data, score test.

    The vocabulary and feature weights are learned only from the model-training
    partition. The calibration slice is held apart from fitting; the canonical
    test partition is used only for final measurement.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    data = load_dataset(remove_metadata=remove_metadata)
    fit_texts, calibration_texts, fit_labels, calibration_labels = split_training_data(
        data.train_texts, data.train_labels, calibration_fraction, random_state
    )
    recipes = [r for r in RECIPES if selected_recipes is None or r.name in selected_recipes]
    if not recipes:
        raise ValueError("No feature recipes selected. Choose one or more known recipe names.")

    rows = []
    n_classes = len(data.class_names)
    for recipe in recipes:
        print(f"\nFeatures: {recipe.name} — {recipe.description}", flush=True)
        vectorizer = make_vectorizer(recipe)
        x_fit_sparse = vectorizer.fit_transform(fit_texts)
        x_cal_sparse = vectorizer.transform(calibration_texts)
        x_test_sparse = vectorizer.transform(data.test_texts)
        vocabulary_size = len(vectorizer.vocabulary_)
        print(f"  vocabulary={vocabulary_size:,}; train rows={x_fit_sparse.shape[0]:,}", flush=True)

        # Fisher LDA estimates class means and a shared covariance matrix. It
        # currently needs dense arrays, so the feature vocabulary is capped.
        lda_data = {
            "fit": x_fit_sparse.toarray(),
            "cal": x_cal_sparse.toarray(),
            "test": x_test_sparse.toarray(),
        }
        for model_name, model in _models().items():
            started = time.perf_counter()
            if model_name == "fisher_lda":
                x_fit, x_cal, x_test = lda_data["fit"], lda_data["cal"], lda_data["test"]
            else:
                x_fit, x_cal, x_test = x_fit_sparse, x_cal_sparse, x_test_sparse
            model.fit(x_fit, fit_labels)
            fit_seconds = time.perf_counter() - started

            cal_log = model.predict_log_proba(x_cal)
            temperature = fit_temperature(cal_log, calibration_labels)
            raw_probabilities = model.predict_proba(x_test)
            test_log = model.predict_log_proba(x_test)
            calibrated_probabilities = apply_temperature(test_log, temperature)
            predicted = model.classes_[np.argmax(raw_probabilities, axis=1)]
            calibrated_predicted = model.classes_[np.argmax(calibrated_probabilities, axis=1)]
            raw_metrics = score_predictions(data.test_labels, predicted, raw_probabilities, n_classes)
            calibrated_metrics = score_predictions(
                data.test_labels, calibrated_predicted, calibrated_probabilities, n_classes
            )
            run_id = f"{recipe.name}__{model_name}"
            rows.append({
                "run_id": run_id,
                "feature_recipe": recipe.name,
                "feature_description": recipe.description,
                "model": model_name,
                "vocabulary_size": vocabulary_size,
                "train_documents": len(fit_texts),
                "calibration_documents": len(calibration_texts),
                "test_documents": len(data.test_texts),
                "fit_seconds": round(fit_seconds, 4),
                "temperature": round(temperature, 6),
                "raw": raw_metrics,
                "calibrated": calibrated_metrics,
            })
            np.savez_compressed(
                prediction_dir / f"{run_id}.npz",
                y_true=data.test_labels,
                y_pred=predicted,
                y_pred_calibrated=calibrated_predicted,
                probabilities=raw_probabilities,
                probabilities_calibrated=calibrated_probabilities,
                class_names=np.asarray(data.class_names),
            )
            save_reliability_plot(
                output_dir / f"{run_id}_reliability.png",
                data.test_labels,
                raw_probabilities,
                calibrated_probabilities,
                f"Reliability: {recipe.name} / {model_name}",
            )
            save_confusion_plot(
                output_dir / f"{run_id}_confusion.png",
                data.test_labels,
                calibrated_predicted,
                data.class_names,
                f"Test confusion matrix: {recipe.name} / {model_name}",
            )
            print(
                f"  {model_name}: macro-F1={calibrated_metrics['macro_f1']:.3f}, "
                f"accuracy={calibrated_metrics['accuracy']:.3f}, "
                f"test ECE={calibrated_metrics['ece_15_bins']:.3f}, T={temperature:.3f}",
                flush=True,
            )

    experiment_record = {
        "configuration": {
            "dataset": "20 Newsgroups via scikit-learn fetch_20newsgroups",
            "remove_headers_footers_quotes": remove_metadata,
            "calibration_fraction_of_official_training_split": calibration_fraction,
            "random_state": random_state,
            "feature_recipes": [recipe.name for recipe in recipes],
            "models": {"fisher_lda": "LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')",
                       "multinomial_nb": "MultinomialNB(alpha=0.5)"},
        },
        "runs": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(experiment_record, indent=2), encoding="utf-8")
    _write_csv(output_dir / "results.csv", rows)
    _write_class_report(output_dir / "class_report.csv", data, prediction_dir, rows)
    _write_summary(output_dir / "RESULTS_SUMMARY.md", rows, data, remove_metadata, random_state)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a flat table that opens cleanly in spreadsheets."""
    columns = ["run_id", "feature_recipe", "feature_description", "model", "vocabulary_size",
               "train_documents", "calibration_documents", "test_documents", "fit_seconds", "temperature"]
    columns += [f"{kind}_{metric}" for kind in ("raw", "calibrated") for metric in rows[0][kind]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key in columns}
            for kind in ("raw", "calibrated"):
                flat.update({f"{kind}_{key}": value for key, value in row[kind].items()})
            writer.writerow(flat)


def _write_class_report(path: Path, data, prediction_dir: Path, rows: list[dict]) -> None:
    """Save per-newsgroup precision, recall and F1 for each calibrated run."""
    from sklearn.metrics import classification_report

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "newsgroup", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for row in rows:
            saved = np.load(prediction_dir / f"{row['run_id']}.npz")
            report = classification_report(
                saved["y_true"], saved["y_pred_calibrated"], labels=np.arange(len(data.class_names)),
                target_names=data.class_names, output_dict=True, zero_division=0,
            )
            for name in data.class_names:
                writer.writerow({
                    "run_id": row["run_id"],
                    "newsgroup": name,
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1": report[name]["f1-score"],
                    "support": report[name]["support"],
                })


def _write_summary(path: Path, rows: list[dict], data, remove_metadata: bool, random_state: int) -> None:
    """Create a concise evidence-based report from actual run metrics."""
    best = max(rows, key=lambda row: row["calibrated"]["macro_f1"])
    lines = [
        "# Results summary (generated)", "",
        "This file is generated by `patternrecog`. Values come from the canonical held-out test set.",
        "Feature recipes and model settings were fixed before this run; do not tune against this test set.", "",
        f"- Classes: {len(data.class_names)}", f"- Test documents: {len(data.test_texts)}",
        f"- Header/footer/quote removal: {remove_metadata}", f"- Random seed: {random_state}",
        f"- Best macro-F1 in this feature comparison: `{best['run_id']}` ({best['calibrated']['macro_f1']:.4f})", "",
        "| Feature recipe | Model | Accuracy | Macro-F1 | Raw log loss | Cal. log loss | Raw Brier | Cal. Brier | Raw ECE | Cal. ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw, calibrated = row["raw"], row["calibrated"]
        lines.append(
            f"| {row['feature_recipe']} | {row['model']} | {calibrated['accuracy']:.4f} | "
            f"{calibrated['macro_f1']:.4f} | {raw['log_loss']:.4f} | {calibrated['log_loss']:.4f} | "
            f"{raw['brier_score']:.4f} | {calibrated['brier_score']:.4f} | "
            f"{raw['ece_15_bins']:.4f} | {calibrated['ece_15_bins']:.4f} |"
        )
    lines += ["", "Raw and calibrated metrics, per-class results, predictions, and plots are saved alongside this report.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
