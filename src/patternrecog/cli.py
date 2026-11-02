"""Command-line entry point for the course project."""

import argparse
from pathlib import Path

from .experiment import run_experiments
from .features import RECIPES


def build_parser() -> argparse.ArgumentParser:
    """Declare command-line options with discoverable help text."""
    parser = argparse.ArgumentParser(
        description="Compare Fisher LDA and Multinomial Naive Bayes on 20 Newsgroups bag-of-words features."
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for metrics, predictions and plots.")
    parser.add_argument("--keep-metadata", action="store_true", help="Keep headers, signatures and quoted text (can cause topic leakage).")
    parser.add_argument("--calibration-fraction", type=float, default=0.20, help="Fraction of official training data reserved to calibrate probabilities.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the stratified calibration split.")
    parser.add_argument("--features", nargs="+", choices=[recipe.name for recipe in RECIPES], help="Run only these feature recipes; by default run all.")
    return parser


def main() -> None:
    """Validate options, run experiments and print the summary path."""
    args = build_parser().parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.40:
        raise SystemExit("--calibration-fraction must be between 0.05 and 0.40")
    selected = set(args.features) if args.features else None
    rows = run_experiments(
        output_dir=args.output,
        remove_metadata=not args.keep_metadata,
        calibration_fraction=args.calibration_fraction,
        random_state=args.seed,
        selected_recipes=selected,
    )
    print(f"\nCompleted {len(rows)} runs. Open: {args.output / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
