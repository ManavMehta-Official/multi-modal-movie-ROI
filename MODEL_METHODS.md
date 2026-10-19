# Movie ROI: methodology and interpretation

## What is predicted?

The target is a **box-office ROI proxy**, worldwide reported revenue divided by production budget, minus one. An ROI of 0.5 means reported gross receipts are 1.5 times budget. It is not profit to an investor: cinemas/distributors retain part of receipts, marketing and financing costs are omitted, and ancillary revenues are absent. Do not use this student prototype to make investment decisions.

Regression models fit log(1 + revenue / budget), then inversely transform the result. Final metadata, metadata + plot and metadata + poster branches use ExtraTrees squared-error regression; the full multimodal branch uses absolute-error histogram gradient boosting. This stabilizes the target's heavy tail but the resulting point prediction is **not conditional expected revenue or expected ROI**. All headline regression errors are evaluated on the original, untransformed ROI scale, without trimming test outliers. Extreme ratios can therefore dominate RMSE and R-squared. Negative R-squared is a valid poor result, not an error to hide.

A separate classifier estimates P(proxy ROI < 0). This is a probability of a defined event, not the magnitude of a potential loss and not an uncertainty score. The 0.5 threshold is fixed in advance for balanced accuracy; it is not an optimized economic action rule. Predicted ROI and downside probability are separate models and need not cross zero/0.5 together.

## Input modalities

1. **Metadata:** log(1+budget), runtime with a missing flag, release year/month, genres, original language. Dictionaries are fitted on training data only. Unknown categories are ignored. Missing runtime is replaced by a fixed 100 minutes, rather than a statistic estimated using future data.
2. **Plot text:** training-only TF-IDF (up to 10,000 unigram/bigram features, English stop words) followed by training-only truncated SVD (up to 32 components). Unseen words never alter the vocabulary. Blank text gives zero text features. English token handling can underrepresent other languages.
3. **Poster pixels:** 73 transparent features: HSV channel histograms (24); mean and standard deviation of RGB in six spatial regions (36); global brightness and gradient statistics (4); brightness quantiles (8); availability flag (1). Posters are resized to 64 × 96. This is **classical image processing, not a pretrained CNN, CLIP or semantic image embedding**. It may capture layout and palette but cannot reliably recognize actors or plot concepts. For feature extraction, missing/unreadable posters use zeros and an explicit missing indicator. For actual inference, a missing poster switches any visual-model request to the separately trained, calibrated metadata + plot model; the result reports the actual model and warning. This avoids pretending a zero image is in-distribution for the complete-case multimodal fit.

No observed revenue, target ROI, user rating, vote count or popularity is an input. Film titles are for display/deduplication, not predictors. Nevertheless, historical metadata and poster versions are not verified as pre-release snapshots: this is a retrospective approximation to a pre-release task, not a demonstrated prospective forecasting system. Runtime and release schedule may also change before release.

## Evaluation design

Movies are sorted by release date and assigned approximately 60%/15%/10%/15% to training, validation, calibration and final test. Movies on the same date remain together. Exact dates and counts are in artifacts/metrics.json and artifacts/split_manifest.csv.

Four ablations use the same training/validation cohort and fixed hyperparameters:

- Metadata only
- Metadata + plot
- Metadata + poster
- Metadata + plot + poster

The initial run used histogram gradient boosting: 120 squared-log regression iterations and 100 classification iterations, at most 15 leaves, L2 regularization 5, seed 42, no internal early stopping. After the initial test was viewed, a bounded follow-on experiment compared 20 fixed train/validation candidates: five regressors/objectives across four modalities (squared-log HGB, absolute-log HGB, absolute-raw HGB, standardized Ridge log, ExtraTrees log). That experiment never scored test rows; however, because the first test had already been seen, the later test evaluation is explicitly a reused exploratory benchmark, not independent testing.

Final per-modality validation winners are recorded in experiments/final_regression_specs.json: ExtraTrees for all three partial-modality branches, HGB absolute-log for full multimodal. ExtraTrees uses 160 trees, minimum 8 samples per leaf, max_features=0.75, seed 42. HGB retains the stated architecture settings with absolute-error objective. Classifiers stay paired to each modality and unchanged; selecting regression MAE does not establish an independently optimal risk classifier. The validation winner minimizes original-scale MAE; classification metrics are reported but do not determine selection. Validation probabilities are uncalibrated. Each iteration writes its validation selection to selection_lock.json before evaluating the temporal benchmark. The first run is preserved under artifacts_initial; the candidate comparisons and original report are in experiments. The supplied final artifacts must not be called an untouched-test result. The full multimodal model is also retained as an explicitly designated demonstration, regardless of whether it wins. Metadata + plot is always retained and calibrated as the predefined missing-poster fallback; its model-specific reused-benchmark metrics and intervals are saved too. Failure to beat simpler ablations is an honest result, not evidence that more modalities always help.

Base models are trained only on the training slice, never refitted using validation or calibration. A separate chronological calibration slice fits sigmoid/logistic calibration to each retained classifier's log-odds. The reused temporal benchmark reports calibrated Brier score, uncalibrated Brier, ROC AUC, balanced accuracy and reliability bins. Sigmoid fitting does not guarantee good calibration under drift; compare actual Brier scores and bin counts.

Regression reports MAE, RMSE and R-squared. A constant training-median regression baseline and training-prior probability baseline are evaluated on the same final test. Constant baselines are calculated directly, mathematically equivalent to median DummyRegressor and prior DummyClassifier. Poster coverage by split is reported. Calibration/test sample size and class prevalence must accompany claims.

## Uncertainty

The calibration slice, separate from training and validation, supplies absolute residuals on original ROI scale. For target 90% coverage, the radius is the sorted residual at rank ceil((n+1) × 0.90), capped at n. Intervals are point prediction ± radius, clipped below at the physical proxy minimum -1. These are constant-width residual intervals, not film-specific uncertainty estimates, and can be broad.

Under exchangeability this is a split-conformal-style construction; chronological temporal drift violates that assumption, so this project does **not** promise 90% future coverage. Report measured final-test coverage and average width next to the target. Do not describe the interval as a confidence interval for an expected return or guaranteed profit range.

## Data selection and reproducibility

See DATA_SOURCES.md and data/data_audit.json for upstream source, licensing caveats and filters. The final fit uses data/movies_multimodal.csv, the frozen subset with successfully retrieved, validated posters. All four ablations use this identical cohort. The full financial metadata cohort remains in data/movies.csv. Poster availability can introduce additional selection bias; a failed remote retrieval is not a film attribute, and incomplete access is not evidence of missing original poster art. Financially incomplete rows are excluded. In particular, zero reported revenue may mean missing data and is excluded rather than treated as certain failure. The canonical educational cohort also restricts budget to at least USD 1 million and dates to 1920–2016. This biases the task toward films with reported finances and excludes many tiny-budget or failed films. Nominal USD comparability is assumed, not independently verified; inflation is not adjusted. Do not extrapolate claims to today's releases without newer, audited data.

The training CLI refuses to run when fewer than 80% of poster files exist; finish downloading first. An intentional partial-image experiment can explicitly set --min-poster-coverage 0, but must not be described as an image-complete evaluation. The direct Python API permits synthetic no-image fixtures. Actual decoded poster coverage is always recorded by split.

Artifacts include source CSV SHA-256, a poster filename/content hash manifest digest, library versions, fixed seed, split manifest and test predictions. Reproduction requires the same source and poster bytes; downloaded posters could later change. Some numerical drift can occur across numerical-library versions. Only load model.joblib that you trained yourself or trust: joblib/pickle can execute code, so the web app never accepts model uploads.

Synthetic test fixtures check software behavior only. Their metrics are never evidence of real-world predictive performance. No further model changes were made after the second evaluation. A fresh, later temporal holdout is required for an independent estimate of the revised pipeline. Do not present the reused benchmark as untouched or tune further against it.

## Primary technical references

- scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- TF-IDF API: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Histogram gradient boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
