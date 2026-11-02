# Bag-of-Words Text Classification: 20 Newsgroups

An end-to-end, reproducible mini-project for the Pattern Recognition assignment. It compares two required multiclass model families on the same text features:

1. **Fisher Linear Discriminant Analysis (LDA)**, a linear discriminant classifier using a shared, shrinkage-regularized covariance estimate.
2. **Multinomial Naive Bayes (MNB)**, a probabilistic generative classifier with additive smoothing.

Four bag-of-words feature recipes are evaluated. The program learns a vocabulary from training documents, holds out part of the official training split to calibrate probabilities, and reserves the official test split for final evaluation. It writes numeric metrics, per-class scores, predictions, reliability plots, and confusion matrices.

## 1. Assignment requirements covered

| Requirement | Where it is handled |
|---|---|
| End-to-end multiclass text pipeline | `src/patternrecog/experiment.py` |
| Bag-of-words representation | `src/patternrecog/features.py` |
| Fisher's discriminant for multiple classes | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` |
| Probabilistic generative model | `MultinomialNB(alpha=0.5)` |
| Class predictions and probabilities | Raw and calibrated prediction arrays saved for every run |
| Probability calibration | Temperature scaling fitted on a training-only calibration holdout |
| Feature-choice study | Counts, binary presence, TF-IDF, and unigram-plus-bigram counts |
| Rigorous evaluation | Accuracy, macro/weighted F1, log loss, multiclass Brier score, ECE, per-class report, and confusion matrices |

## 2. Quick start (Windows PowerShell)

Python 3.11 or 3.12 is recommended. From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiments.py
```

The first run downloads the 20 Newsgroups data through scikit-learn and can take several minutes. Internet access is needed for that first download. Results are written to `results/`.

Run a smaller comparison while learning the workflow:

```powershell
python run_experiments.py --features count_unigram tfidf_unigram
```

Run with metadata retained (included only as a leakage-prone comparison):

```powershell
python run_experiments.py --keep-metadata
```

Change the seed, calibration split, or output directory:

```powershell
python run_experiments.py --seed 17 --calibration-fraction 0.20 --output results_seed17
```

Equivalent invocation without installing the console command:

```powershell
python -m pip install -e .
patternrecog --help
```

## 3. Output files

| File | Contents |
|---|---|
| `results/RESULTS_SUMMARY.md` | Generated comparison table for the current run |
| `results/results.csv` | Flat raw and calibrated metrics, easy to open in Excel |
| `results/results.json` | Full structured metrics and run metadata |
| `results/class_report.csv` | Per-class precision, recall, F1, and support for each run |
| `results/predictions/*.npz` | Ground truth, predictions, probabilities, and class names |
| `results/*_reliability.png` | Raw versus calibrated reliability curves |
| `results/*_confusion.png` | Row-normalized test confusion matrices using calibrated predictions |

Generated data and results are ignored by Git so the repository stays small. Do not report numbers from an example or another machine: run the program and cite the metrics generated on your own machine.

## 4. Data and experimental protocol

The assignment links the [UCI Twenty Newsgroups page](https://archive.ics.uci.edu/dataset/113/twenty%2Bnewsgroups), which describes 20,000 messages. The program uses scikit-learn's canonical `fetch_20newsgroups` train/test partition so there is a stable official test split. That packaged version contains 11,314 training messages and 7,532 test messages (some original documents are omitted by the packaged collection). It has 20 topic labels.

By default, message headers, footers, and quoted text are removed because they can contain source or reply clues that make topic recognition artificially easy. The `--keep-metadata` option lets you inspect this effect. Class names and the standard train/test split are unchanged.

The official training partition is stratified into a model-fitting portion (80% by default) and a calibration portion (20%). A vocabulary and IDF weights, when used, are learned from only the fitting portion. The calibration slice chooses a single temperature per model/feature run. The official test partition is held out until scoring. Do not adjust settings repeatedly based on test scores; choose settings using training-only cross-validation if you need model selection.

For Fisher LDA, bag-of-words vectors are dense. The vocabulary is limited to the top 1,500 training terms to bound the covariance matrix's memory. Multinomial Naive Bayes remains sparse. Each feature recipe uses the same maximum vocabulary size and document preprocessing for a fair comparison.

## 5. Feature choices

The vectorizer lowercases text, normalizes accented characters, removes common English stop words, ignores terms appearing in fewer than two fitting documents, and caps the vocabulary at 1,500 features. These are conventional token word features; this is not a sequence or embedding model.

| Recipe | Feature value | What it tests |
|---|---|---|
| `count_unigram` | Number of occurrences of each word | Repetition can provide evidence |
| `binary_unigram` | 1 if a word occurs, otherwise 0 | Whether presence is enough without repetition |
| `tfidf_unigram` | Sublinear term frequency times inverse document frequency, L2 normalized | Downweights words common across many documents |
| `count_1_2gram` | Counts of one-word and adjacent two-word tokens | Whether short phrases add topic information |

Feature choices can change both accuracy and probability quality. A model can classify well but be overconfident, so the probability metrics and reliability plots matter too.

## 6. Models, in understandable terms

### Fisher multiclass LDA

LDA represents each document by a feature vector. For each class it estimates a mean feature vector, then estimates one pooled within-class covariance shared across classes. Its linear score for class `k` is proportional to:

`xᵀ Σ⁻¹ μₖ − ½ μₖᵀ Σ⁻¹ μₖ + log P(k)`

Here `x` is a document vector, `μₖ` is the class mean, `Σ` is the shared covariance, and `P(k)` is the class prior. Selecting the largest score is the multiclass Fisher decision rule. Shrinkage (`auto`) stabilizes covariance estimates when features are correlated or the sample count is limited. Scikit-learn converts class scores to posterior-like probabilities with `predict_proba`.

### Multinomial Naive Bayes

MNB treats a document as a set of word counts and assumes the word occurrences are conditionally independent given a topic. For class `k` and word `j`, it estimates a smoothed word probability:

`P(word j | class k) = (count(j,k) + α) / (total word counts in k + α × vocabulary size)`

The class score combines log word probabilities with a class prior. `alpha=0.5` is additive smoothing: it prevents a word absent from the fitting messages of one topic from receiving probability zero. Count and binary features match its assumptions most directly; TF-IDF is included as a useful empirical comparison, though TF-IDF values are not literal multinomial counts.

### Temperature scaling

Both models return a probability distribution, but raw probabilities can be too confident or too cautious. Temperature scaling divides log-probability scores by a positive scalar `T`, then normalizes them to sum to one. A value above one softens the distribution; below one sharpens it. We select `T` by minimizing negative log likelihood on the calibration slice, not on the test documents. This simple method preserves each model's top class, so it changes confidence without changing the predicted label except for numerical ties.

## 7. Metrics and how to read them

- **Accuracy:** fraction of all messages assigned to the correct newsgroup.
- **Macro-F1:** computes F1 independently for each class and averages equally. Use it to notice weak performance on smaller topics.
- **Weighted-F1:** averages class F1 weighted by each class's test support.
- **Log loss:** penalizes probability assigned away from the true class; lower is better. A confident wrong answer receives a large penalty.
- **Multiclass Brier score:** mean sum of squared differences between the predicted class-probability vector and the true one-hot vector; lower is better.
- **ECE (15 bins):** top-label expected calibration error. It compares average confidence with observed accuracy in confidence bins; lower is better. ECE depends on binning, so read it alongside log loss, Brier score, and the reliability plot.
- **Per-class precision/recall/F1:** precision asks what fraction of predicted class examples are correct; recall asks what fraction of actual class examples were found.
- **Confusion matrix:** each row is normalized to one; an off-diagonal cell shows the fraction of one true topic predicted as another.

Always state which split was scored and whether results are raw or calibrated. For label-quality comparisons, use accuracy and macro-F1. For confidence quality, use log loss, Brier, ECE, and the reliability curve.

## 8. Source-code map

The code is split into short modules so that data handling, feature creation, fitting, and scoring are not mixed together:

| File | Responsibility |
|---|---|
| `src/patternrecog/cli.py` | Parses command-line options and starts a run |
| `src/patternrecog/data.py` | Downloads the dataset and makes the stratified calibration split |
| `src/patternrecog/features.py` | Defines the four text feature recipes |
| `src/patternrecog/calibration.py` | Fits and applies temperature scaling |
| `src/patternrecog/metrics.py` | Computes classification and calibration metrics |
| `src/patternrecog/plotting.py` | Draws reliability curves and confusion matrices |
| `src/patternrecog/experiment.py` | Coordinates vectorization, model fitting, predictions, and output writing |
| `run_experiments.py` | Convenience launcher for running directly from the project checkout |

The main run loop is: load train/test text → split train for calibration → fit a vectorizer only on fitting text → transform calibration/test text → fit each model → learn its temperature from calibration predictions → score test predictions → save all metrics and figures.

## 9. Writing up your findings

Use `results/RESULTS_SUMMARY.md` as the measured-results table. In the final course report, add a short explanation answering:

1. Which feature recipe gave the strongest macro-F1 for each model?
2. Did repeated counts, binary presence, TF-IDF, or bigrams help? Offer a text-based explanation, not just a number.
3. Which model had better raw probability quality? Did temperature scaling improve test log loss, Brier score, or ECE?
4. Which newsgroups are most often confused? Use `class_report.csv` and confusion plots to identify them.
5. What are the limitations of bag-of-words, the conditional-independence assumption, and a single scalar calibration temperature?

Do not claim calibration improved every measure: use the generated raw-versus-calibrated values. Calibration temperature scaling preserves argmax labels; changes in classification metrics should only occur in tie cases.

## 10. Limitations and reproducibility

- 20 Newsgroups is a historical English discussion dataset. Results may depend on collection-era vocabulary and newsgroup-specific style.
- Removing metadata reduces shortcut learning, but does not eliminate duplicate, author, or topic leakage.
- Bag-of-words ignores word order except in the bigram recipe and cannot understand context or word meaning.
- LDA's shared-covariance assumption is restrictive; shrinkage makes estimation stable, not necessarily correct.
- Naive Bayes independence is knowingly simplistic and often yields overconfident raw probabilities.
- Temperature scaling uses one held-out subset and one scalar, so calibration may vary by class. Test calibration metrics themselves have sampling and binning uncertainty.
- This is a single seeded split evaluation rather than repeated cross-validation. If the course requires uncertainty estimates, run multiple seeds or training-only cross-validation and report variation without tuning on the test set.

Reproducibility controls: fixed official data split, fixed seed (`42` by default), stratified calibration subset, fixed feature/model settings, and saved run configuration metadata in `results.json`.

## References

- Mitchell, T. (1997). [Twenty Newsgroups, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/113/twenty%2Bnewsgroups). DOI: [10.24432/C5C323](https://doi.org/10.24432/C5C323).
- scikit-learn, [Working with text data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).
- scikit-learn, [20 Newsgroups dataset loading](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).
- scikit-learn, [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html) and [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
- Guo, C. et al. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Temperature scaling is also a simple post-hoc calibration technique; this project applies it to linear and generative classifier scores.
