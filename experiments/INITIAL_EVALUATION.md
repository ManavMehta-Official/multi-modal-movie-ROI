# Archived first-run evaluation snapshot

This records the initial model before the follow-on validation experiment. It is retained for transparent history and is not the final shipped model report. See ../EVALUATION.md and ../artifacts/ for the revised, reused-benchmark results.

# Measured evaluation results

These are measured results from the frozen real-data run, not synthetic fixtures. The final test was evaluated after model selection and probability/interval calibration. No tuning followed inspection of these results.

## Cohort and splits

The full financially eligible metadata cohort contains 4,910 films. The final fair multimodal comparison uses the identical 2,601-film subset with validated downloaded posters (100% decoded coverage in every split). Missing retrievals reflect source/access limitations and can introduce selection bias.

| Split | Films | Release dates |
|---|---:|---|
| train | 1560 | 1922-01-11 to 2007-03-23 |
| validation | 390 | 2007-04-05 to 2011-02-04 |
| calibration | 260 | 2011-02-08 to 2013-09-06 |
| test | 391 | 2013-09-07 to 2016-12-25 |

## Validation ablations

The fixed selection criterion was lowest validation MAE. Metadata-only won. Validation classification probabilities below are not yet calibrated.

| Features | MAE | RMSE | R² | ROC AUC | Brier | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| tabular | 1.796 | 3.421 | -0.070 | 0.647 | 0.216 | 0.544 |
| tabular_text | 1.813 | 3.346 | -0.023 | 0.619 | 0.218 | 0.564 |
| tabular_poster | 1.827 | 3.413 | -0.065 | 0.632 | 0.216 | 0.572 |
| multimodal | 1.810 | 3.309 | -0.001 | 0.650 | 0.210 | 0.575 |

## Untouched final test

| Model | MAE | RMSE | R² | ROC AUC | Brier | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Metadata (validation winner) | 2.408 | 4.929 | -0.051 | 0.681 | 0.206 | 0.602 |
| Full multimodal demonstration | 2.401 | 4.940 | -0.055 | 0.561 | 0.223 | 0.502 |
| Metadata + plot (missing-poster fallback) | 2.435 | 4.969 | -0.068 | 0.585 | 0.221 | 0.524 |
| Constant training-median / training-prior baseline | 2.379 | 4.943 | -0.057 | 0.500 | 0.226 | 0.500 |

MAE/RMSE are in ROI-ratio units: MAE 2.408 is approximately 240.8 percentage points of ROI error. The learned regressors do not beat the constant baseline on test MAE. Negative R² means worse squared-error performance than predicting the test-set mean (the latter is a metric reference, not a deployable fitted baseline). Poster/text additions do not show a robust predictive gain. The full multimodal downside classifier is only slightly better than chance discrimination on this test. This is a working research prototype, not a reliable commercial prediction system.

Metadata-only has the strongest retained downside discrimination (AUC 0.681). Its test Brier improves from 0.210 uncalibrated to 0.206 after held-out sigmoid calibration; multimodal improves from 0.232 to 0.223. Improvement does not establish individual-film certainty. Test downside prevalence is 34.3%.

## Residual intervals

| Model | Target coverage | Measured test coverage | Mean width (ROI units) |
|---|---:|---:|---:|
| multimodal | 90.0% | 88.2% | 6.535 |
| tabular | 90.0% | 88.5% | 6.577 |
| tabular_text | 90.0% | 88.0% | 6.687 |

Intervals are wide and do not achieve nominal 90% on this temporal holdout. They are empirical residual intervals, not guaranteed future coverage or expected-profit confidence intervals.

## Artifacts and reproducibility

- metrics.json: complete measured metrics, reliability bins, data audits and split bounds
- selection_lock.json: validation choice written before final-test evaluation
- split_manifest.csv: exact rows assigned to each split
- test_predictions.csv: each held-out prediction and interval for independent checking
- model_card.json: feature/target definitions, limitations, source/poster hash, dependency versions
- model.joblib: trusted locally trained models; never load an untrusted replacement

Run python -m pytest to check software behavior. Synthetic tests check correctness only. See MODEL_METHODS.md and DATA_SOURCES.md for the full design and data-use caveats.
