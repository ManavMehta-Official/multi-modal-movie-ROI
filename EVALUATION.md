# Measured evaluation results

## Evaluation status

Reused exploratory temporal benchmark: initial results were seen before a bounded train/validation-only follow-on model comparison. This is not a fresh independent test; see artifacts_initial and experiments/validation_candidates.json. No further test-guided tuning.

The original run is preserved in artifacts_initial/ and experiments/INITIAL_EVALUATION.md. A bounded follow-on compared 20 fixed train/validation candidates; final per-modality choices are in experiments/final_regression_specs.json. The revised pipeline needs a fresh later temporal holdout for independent assessment.

## Cohort and splits

Full financially eligible metadata: 4,910 films. Identical poster-complete training/evaluation cohort for all ablations: 2,601 films. Decoded poster coverage is 100% in each split. Image availability and financial reporting introduce selection bias.

| Split | Films | Release dates |
|---|---:|---|
| train | 1560 | 1922-01-11 to 2007-03-23 |
| validation | 390 | 2007-04-05 to 2011-02-04 |
| calibration | 260 | 2011-02-08 to 2013-09-06 |
| test | 391 | 2013-09-07 to 2016-12-25 |

## Validation ablations

Lowest validation MAE selects the regression branch; this does not select an independently optimal risk classifier. Classifiers remain paired by modality. Validation Brier scores are before calibration.

| Features | Regressor | MAE | RMSE | R² | ROC AUC | Brier |
|---|---|---:|---:|---:|---:|---:|
| tabular | extratrees_log | 1.730 | 3.345 | -0.023 | 0.647 | 0.216 |
| tabular_text | extratrees_log | 1.742 | 3.330 | -0.014 | 0.619 | 0.218 |
| tabular_poster | extratrees_log | 1.756 | 3.348 | -0.025 | 0.632 | 0.216 |
| multimodal | hgb_absolute_log | 1.758 | 3.327 | -0.012 | 0.650 | 0.210 |

## Reused temporal benchmark

| Model | MAE | RMSE | R² | ROC AUC | Brier | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Metadata (validation winner) | 2.322 | 4.910 | -0.043 | 0.681 | 0.206 | 0.602 |
| Full multimodal demonstration | 2.386 | 4.990 | -0.077 | 0.561 | 0.223 | 0.502 |
| Metadata + plot (missing-poster fallback) | 2.325 | 4.915 | -0.045 | 0.585 | 0.221 | 0.524 |
| Constant training-median / training-prior baseline | 2.379 | 4.943 | -0.057 | 0.500 | 0.226 | 0.500 |

MAE/RMSE are in ROI-ratio units: selected-model MAE 2.322 is approximately 232.2 percentage points of ROI error. Its MAE improvement over the constant baseline is only 2.4%. The full multimodal model does not beat the constant baseline MAE. Negative R² indicates worse squared-error performance than a test-mean reference (which is not a deployable fitted baseline).

Poster/text additions do not demonstrate a robust predictive gain. The full multimodal downside classifier is only slightly better than chance discrimination. The metadata branch performs better on downside discrimination, but its selection was based on regression validation MAE. This is a working educational research prototype, not a reliable commercial prediction system.

Calibration improves measured Brier scores, but does not establish individual-film certainty. Exact uncalibrated/calibrated scores and reliability-bin counts are in metrics.json. Benchmark downside prevalence is 34.3%.

## Residual intervals

| Model | Target coverage | Measured coverage | Mean width (ROI units) |
|---|---:|---:|---:|
| multimodal | 90.0% | 88.0% | 6.277 |
| tabular_text | 90.0% | 89.0% | 6.637 |
| tabular | 90.0% | 87.7% | 6.280 |

Intervals are wide and below the nominal 90% target on this reused temporal benchmark. They are empirical residual intervals, not guaranteed future coverage or expected-profit confidence intervals.

## Reproduction and checks

Run the final recipe from the project root:

python -m movie_roi train --data data/movies_multimodal.csv --output artifacts --regression-specs experiments/final_regression_specs.json --reused-benchmark

This reproduces the chosen configuration; do not retune against the benchmark. Run python -m pytest for software checks. Synthetic fixtures verify wiring and never establish real predictive quality. See MODEL_METHODS.md and DATA_SOURCES.md for the full design and provenance.

Artifacts include metrics.json, selection_lock.json, split_manifest.csv, test_predictions.csv, model_card.json and trusted model.joblib. Never load an untrusted joblib/pickle file.
