# Quality assurance and release assessment

Reviewed 13 October 2026 against the supplied final model artifacts.

## Release assessment

**Suitable for a local academic demonstration with the stated limitations. Not validated for investment decisions or dependable commercial forecasting.** The project has a functioning Python training/inference pipeline and Streamlit interface. All three feature branches are implemented with real metadata, plot text and poster pixels. The image branch uses handcrafted descriptors, not a pretrained semantic vision model.

The final regression evidence is modest: metadata-only improves MAE over the training-median baseline by approximately 2.4% on a **reused exploratory temporal benchmark**, while the full multimodal model does not beat that baseline. Negative R² and wide intervals remain important limitations. No statistical significance or profit guarantee is established.

## Checks executed

- **17 automated tests passed**, including model, independent audit and frontend tests. The final full run completed in 11.45 seconds on Python 3.12.
- Python compilation passed for the app, model package and scripts.
- Command-line inference using `examples/input.json` and the supplied final artifact completed with finite outputs, a valid probability and appropriate extrapolation/model warnings.
- The only test warning was environment-specific physical CPU-core detection; execution fell back to logical cores and completed.

Reproduction commands from the project root after dependency installation:

```bash
python -m pytest -q
python -m py_compile app.py movie_roi/*.py scripts/*.py
python -m movie_roi predict --input examples/input.json --artifacts artifacts
```

### Data, features and evaluation

- Independently joined all 4,910 financial-cohort rows to the raw source and verified conversion from million USD to nominal USD, the ROI formula and unique film IDs.
- Verified the final model uses **2,601 poster-available films**, not all 4,910. Every included poster was decoded successfully. Source CSV hash matches the final model card.
- Checked chronological split separation, including keeping same-day releases together: 1,560 training, 390 validation, 260 calibration and 391 benchmark rows.
- Verified that changing revenue, ROI, popularity, vote count or ratings does not change predictive features. Training-only vocabulary and metadata categories remain unchanged when held-out rows are transformed.
- Verified that different actual poster pixels change the 73-dimensional visual descriptor, and each modality combination has the expected feature dimensions.
- Checked target-transform inversion, the strict revenue-below-budget downside event and valid probability/interval outputs.
- Recomputed saved MAE, RMSE, R², ROC AUC, Brier score, balanced accuracy and interval coverage from all 391 saved final predictions. These match `artifacts/metrics.json`.
- Independently recomputed the constant training-median regression and training-prior probability baselines.

### Frontend and inference

- Tested missing-artifact behavior: estimation is disabled rather than replaced with fabricated results.
- Tested invalid budgets, dates, runtime, empty fields, corrupt images and oversized uploads.
- Tested repeated submissions, scenario switching and clearing stale results after an invalid submission.
- Tested valid poster upload through the prediction helper and real-artifact multimodal inference.
- Verified a missing/unreadable poster changes a requested visual model to the separately trained metadata-plus-plot fallback. The result identifies the actual model and uses that model's own interval coverage; a valid poster uses all three modalities.
- The app accepts poster images, not arbitrary model uploads. It re-encodes poster data to a temporary file and removes it after inference. Loading untrusted joblib/pickle artifacts remains unsafe and is explicitly discouraged.

## Final measured evidence

All rows below refer to the same 391-film reused temporal benchmark, released 7 September 2013 through 25 December 2016.

| Model | ROI MAE | ROI RMSE | R² | Downside AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Metadata, validation-selected | 2.322 | 4.910 | -0.043 | 0.681 | 0.206 |
| Metadata + plot + poster | 2.386 | 4.990 | -0.077 | 0.561 | 0.223 |
| Training-median / training-prior baseline | 2.379 | 4.943 | -0.057 | 0.500 | 0.226 |

MAE 2.322 represents approximately **232.2 percentage points of proxy ROI error**, not 2.322%. Metadata-only interval coverage is 87.7%; multimodal coverage is 88.0%, against a nominal 90% target. Average interval widths are about 6.28 ROI units. These are empirical outcome intervals, not confidence intervals for expected investor returns.

The downside probability means reported worldwide box-office revenue is below reported production budget. It does not measure investor loss probability, loss severity or net profit. Marketing costs, exhibitor/distributor shares and other cash flows are missing. Inverse-transformed log-target predictions are not calibrated conditional expected revenue.

## Evaluation history and limits

The first frozen run was evaluated before a bounded follow-on comparison of 20 train/validation-only regression candidates. Initial artifacts and results are preserved in `artifacts_initial/` and `experiments/INITIAL_EVALUATION.md`. The final model choices used training/validation comparisons, but the initial benchmark outcomes had already been seen. Consequently, final results are **exploratory reused-benchmark evidence, not a fresh independent test**. A new later-time dataset is needed for an independent estimate of the revised system. No further model tuning followed the final evaluation.

All ablations share the same poster-complete cohort. Image availability/acquisition selected 2,601 of 4,910 eligible films, introducing an additional potential selection bias. Historical source metadata and poster versions are not guaranteed to be the versions available before release. Source financial accuracy, inflation and currency comparability are not independently verified. Chronological drift invalidates a blanket future-coverage guarantee.

## Not verified

- No prospective real-release forecasts, investor profitability or commercial suitability.
- No statistical significance claim for the small baseline improvement.
- No independent fresh temporal test after the follow-on comparison.
- No rendered-browser pixel-level visual assessment or mobile/accessibility audit; frontend verification used Streamlit's app-testing framework and prediction helpers.
- No fresh dependency installation on Windows/macOS, production deployment, concurrent-user load test or external security audit.
- No guarantee of continued poster-source access or unrestricted redistribution rights. See `DATA_SOURCES.md` for source attribution and third-party artwork caveats.

Use `README.md` for startup, `MODEL_METHODS.md` for the design, and `EVALUATION.md` plus saved JSON/CSV artifacts for the measured evidence. Synthetic fixtures test software correctness only and are not presented as predictive-performance evidence.
