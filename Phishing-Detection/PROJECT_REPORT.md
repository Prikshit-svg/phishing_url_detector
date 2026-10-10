# Phishing URL Detection — Reproducible Evaluation Report

## Run summary

- Dataset rows after cleaning: **120,000**
- Duplicate URLs removed: **0**
- Supplied label counts: **60,000 label 0 / 60,000 label 1**
- Expected label mapping (must be verified against source): `0 = legitimate`, `1 = phishing`
- Features: **30 lexical URL features**, regenerated from each URL
- Split: **GroupShuffleSplit by approximate registrable domain**, with 20% requested for holdout (actual held-out size: 24,757 rows)
- Model selection: highest validation F1
- Selected model: **Random Forest**
- Trained model rows: **45,000 sampled rows from the training partition** for practical runtime; held-out test domains remain separate

## Held-out test results

| Metric | Result |
|---|---:|
| Accuracy | 83.52% |
| Precision (phishing class) | 81.94% |
| Recall (phishing class) | 87.90% |
| F1-score (phishing class) | 84.82% |
| ROC-AUC | 0.9202 |
| Average precision | 0.9263 |
| Brier score | 0.1165 |

Confusion matrix (rows = actual, columns = predicted; class order: legitimate, phishing):

| | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | 9,276 | 2,512 |
| Actual phishing | 1,569 | 11,400 |

## Model selection (validation F1)

| Model | Validation F1 | Validation ROC-AUC |
|---|---:|---:|
| Random Forest | 0.8669 | 0.9424 |
| Extra Trees | 0.8433 | 0.9221 |
| XGBoost | 0.8387 | 0.9232 |
| Gradient Boosting | 0.7936 | 0.8868 |
| Logistic Regression | 0.7356 | 0.8143 |

See `artifacts/model_comparison.csv` for the full validation metrics.

## Generated evidence

- `model_comparison.png` — validation metrics for model selection
- `roc_curves.png` — ROC curves on held-out test data
- `precision_recall_curves.png` — precision/recall trade-off on held-out test data
- `confusion_matrix.png` — selected model's held-out test errors
- `calibration_curve.png` — observed frequency versus predicted probability
- `threshold_analysis.png` and `threshold_analysis.csv` — threshold trade-offs on validation data only
- `permutation_importance.png` and `permutation_importance.csv` — feature permutation effects on held-out data
- `class_distribution.png` — label counts
- `adaptive_learning.png` and `adaptive_learning_experiment.csv` — deterministic group-batch refit simulation

## Limitations and interpretation

1. **Labels are not independently verified.** Counts are balanced, but that does not establish label correctness. Verify source documentation and inspect examples before presenting the result as a definitive benchmark.
2. **Domain grouping is approximate.** The split reduces URL/domain overlap, but the parser is not a full Public Suffix List implementation. Use an exact registrable-domain parser and an external dataset for a stronger study.
3. **No timestamps are present.** The adaptive-learning plot uses deterministic domain batches and is not evidence of temporal drift, online learning, or future performance.
4. **No live URL checks occur.** TLS, DNS, and region panels are string/TLD heuristics. No website is visited and no certificate, DNS record, or server location is verified.
5. **Probability calibration is not guaranteed.** A Brier score and calibration plot are supplied, but calibration should be re-evaluated on representative deployment data. Use validation data—not the test set—to tune a decision threshold.
6. **Metrics are dataset-specific.** Real-world phishing prevalence may differ from the near-balanced input dataset. Precision and false-alarm rates can change substantially under that distribution shift.

## Reproduction

From the project directory, install `requirements.txt`, then run `python train.py`. This regenerates the model artifact, JSON/CSV metrics, and graphs. Run automated checks with `python -m unittest discover -s tests -v`.

## Added URL-level explainability and reliability views

The application now presents a SHAP waterfall plot, normalized URL-feature profile, reference URL-length distributions, a held-out calibration curve marked with the current score, validation threshold trade-offs, a separate heuristic indicator chart, qualitative model disagreement, and SHAP explanation-sensitivity analysis for controlled path/query variants. The model-evaluation page adds subgroup precision/recall/F1 for overlapping lexical URL categories.

These additions are designed to make a research demonstration more transparent. They must not be interpreted as independent proof of maliciousness. The feature-profile radar uses empirical percentiles against the supplied dataset; the reference distributions inherit any label-quality problems in that dataset. The comparison models saved for the current interactive chart are lightweight fits on a sample of the training partition, so their disagreement is qualitative only and must not be reported as a benchmark result or confidence interval. The selected model's reported evaluation remains the basis for benchmark metrics.

The explanation-sensitivity graph changes only path/query variants, but each variant is a different URL and may genuinely have different characteristics. Similarity is descriptive, not a robustness guarantee. Calibration is a population-level property, not a per-URL confidence estimate. SSL, DNS and region indicators remain string-based heuristics rather than live verification.
