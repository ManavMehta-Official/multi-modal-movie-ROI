# Frame · Movie ROI & Risk Lab

An academic Python project for estimating a **box-office / production-budget ROI proxy** and the modeled probability that reported worldwide revenue falls below production budget. A Streamlit frontend combines structured metadata, plot text and optional poster imagery.

This is a retrospective research prototype, **not an investment advisor**. It does not estimate investor cash flows, studio profit or a guaranteed return. Treat the dataset and images as educational research material; read [DATA_SOURCES.md](DATA_SOURCES.md) before reuse.

## Quick start

Python 3.11 or 3.12 is recommended. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Nothing is deployed publicly. If trained artifacts are included, the interface can predict immediately. Otherwise train first:

```bash
python scripts/freeze_data.py
python -m movie_roi train --data data/movies_multimodal.csv --output artifacts --regression-specs experiments/final_regression_specs.json --reused-benchmark
python -m streamlit run app.py --server.address 127.0.0.1
```

The bundled frozen training cohort contains 2,601 movies with verified cached posters, drawn from 4,910 eligible metadata rows. Every ablation uses the same 2,601-film cohort for a fair comparison. Poster availability adds selection bias. Training and prediction use local files; no download is needed for the bundled run. The historical downloader is retained for provenance, but further downloads were stopped after upstream access denials. Do not retry denied requests or bypass those restrictions. Poster upload is optional. If an image-dependent model is requested without a valid poster, inference uses a separately trained metadata + plot model and labels that fallback explicitly. It does not fabricate a poster or call a text-only prediction multimodal. The offline freeze command validates the currently cached posters and rebuilds the complete-case cohort and audit. Run it only before retraining; the cohort hash must stay aligned with the saved model. The source and cohort audits record filtering and provenance.

To use a different artifact folder, set the `MOVIE_ROI_ARTIFACTS` environment variable before launching the app. Only load artifacts you trained or trust: joblib/pickle files can execute code.

## Interface

- **Prediction:** choose a fictional editable scenario; enter budget in USD, release date, runtime, genres and plot. Optionally supply a PNG/JPEG poster you have permission to use. Submit to see modeled revenue, proxy ROI, downside probability and an uncertainty range. Export the submitted inputs and result as JSON.
- **Model choice:** the default multimodal demonstration combines all available input types. Switch to the validation-selected model to use the variant selected by validation ROI error. Extra modalities are not assumed to improve performance.
- **Evaluation:** real saved metrics, chronological split dates, modality ablations and a baseline comparison. There are no hardcoded or random display metrics.
- **About:** target definition, model provenance and limitations.

The built-in scenarios are handwritten fiction. They do not contain actual revenue or preset results. Results stay labeled as the last submitted snapshot until another estimate is requested. Missing model artifacts disable estimation and display the training command.

### Poster input

The interface validates PNG/JPEG decoding, a 10 MB size cap and a 25-million-pixel limit. Valid images are re-encoded into a temporary local PNG for prediction and removed afterward. The current visual encoder uses classical color, layout and texture descriptors; it is not a pretrained semantic vision model. Plot features use TF-IDF and train-only dimensionality reduction.

## Reproducible command-line prediction

```bash
python -m movie_roi predict --input examples/input.json --artifacts artifacts
```

The model API is `from movie_roi import predict`; call `predict(record, artifact_dir="artifacts", poster_path=None)`. Supply `model_name="multimodal"` to force the multimodal variant. Records use `title`, `budget`, `release_date` (YYYY-MM-DD), `runtime`, pipe-separated `genres`, `overview`, and optional `original_language`. Actual revenue must never be an inference input.

## Evaluation design

The pipeline uses chronologically ordered train, validation, calibration and test splits. Validation compares feature combinations; calibration is separate from model selection; the latest chronological split is now a **reused exploratory temporal benchmark**. Initial test results were inspected before a bounded follow-on train/validation model comparison, so the final metrics are not a fresh independent test. Earlier artifacts and search provenance are retained; no independent final-test claim is made. Read the generated `artifacts/metrics.json` and `artifacts/model_card.json` for the exact split counts, selected model, metrics, feature definitions and run provenance.

Metrics include ROI MAE/RMSE/R², downside ROC AUC, Brier score and balanced accuracy. Uncertainty outputs report nominal interval coverage and observed temporal benchmark coverage. These measure dataset-level behavior and are not guarantees for an individual movie.

## What the target means

`proxy_ROI = (worldwide_box_office_revenue - production_budget) / production_budget`

`downside = proxy_ROI < 0`

A value of 1.0 means revenue is twice production budget; it does **not** mean an investor doubled their money. Marketing, distribution fees, exhibitors' share, financing costs and other revenues are missing. Reported financial completeness causes selection bias; zero/unreported revenue is excluded rather than mislabeled as loss. Nominal dollars are not inflation-adjusted. Historical metadata can have been updated after release, so as-of-release provenance is not guaranteed. The model is therefore an academic pre-release-feature experiment, not a backtested trading strategy.

## Checks

```bash
python -m pytest -q
python -m py_compile app.py movie_roi/*.py scripts/*.py
```

See the delivered QA report for checks actually executed in the build environment. Retraining can change results if the upstream dataset changes; preserve the raw download and its audit hash for reproducibility.

## Suggested one-week presentation

1. State the proxy target and missing profit components.
2. Show the data source, cleaning filters and chronological splits.
3. Explain structured, text and image feature branches.
4. Compare validation ablations and exploratory temporal benchmark and baseline results honestly.
5. Demonstrate one fictional input with and without a poster.
6. Discuss broad uncertainty, selection bias and limits of deployment.

## Files

- `app.py`: local interactive frontend
- `movie_roi/`: feature extraction, training and prediction
- `scripts/prepare_data.py`: original documented dataset/poster preparation
- `scripts/freeze_data.py`: offline complete-case cohort validation and freeze
- `data/`: local metadata, audit and available posters
- `artifacts/`: trusted trained model and saved evaluation, when generated
- `examples/input.json`: command-line example
- `tests/`: reproducibility and validation checks
- `DATA_SOURCES.md`: attribution, provenance and usage limits
