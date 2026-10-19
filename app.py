"""Local, pre-release-only interface to the trained movie ROI research model."""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get('MOVIE_ROI_ARTIFACTS', str(ROOT / 'artifacts')))
GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
          'Science Fiction', 'Thriller', 'War', 'Western']
# Handwritten fictional scenarios. No target/revenue data enters a prediction.
SCENARIOS = {
    'Original scenario': {'title': 'The Last Observatory', 'budget': 20_000_000.0,
        'runtime': 110, 'genres': ['Drama', 'Science Fiction'],
        'overview': 'A young astronomer and her estranged father reunite at an isolated observatory after receiving a mysterious signal from deep space.'},
    'Fictional indie drama': {'title': 'A Place by the River', 'budget': 2_000_000.0,
        'runtime': 95, 'genres': ['Drama'],
        'overview': 'A retired teacher returns to her hometown to save its public library, finding an unlikely ally in a teenager searching for a place to belong.'},
    'Fictional adventure': {'title': 'Atlas of the Lost Sky', 'budget': 80_000_000.0,
        'runtime': 125, 'genres': ['Adventure', 'Fantasy'],
        'overview': 'Two rival explorers discover a map to a city floating above the clouds and must join forces before a ruthless collector reaches it first.'},
}


def validate_inputs(record: dict, poster_bytes: bytes | None = None) -> list[str]:
    errors = []
    if not str(record.get('title', '')).strip():
        errors.append('Add a working title for this scenario.')
    for name, low, high in [('budget', 1, 1_000_000_000), ('runtime', 1, 600)]:
        try:
            value = float(record[name])
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f'{name.capitalize()} must be between {low:,} and {high:,}.')
    if not str(record.get('overview', '')).strip():
        errors.append('Add a plot overview so the text model has an input.')
    if not record.get('genres'):
        errors.append('Choose at least one genre.')
    try:
        dt.date.fromisoformat(str(record.get('release_date', '')))
    except ValueError:
        errors.append('Choose a valid release date.')
    if poster_bytes is not None:
        if len(poster_bytes) > 10 * 1024 * 1024:
            errors.append('Use a poster smaller than 10 MB.')
        else:
            try:
                with Image.open(io.BytesIO(poster_bytes)) as im:
                    if im.width * im.height > 25_000_000:
                        errors.append('Use a poster with at most 25 million pixels.')
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                errors.append('The poster could not be read. Upload a valid PNG or JPEG.')
    return errors


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def run_prediction(record: dict, poster_bytes: bytes | None = None, model_name: str | None = None) -> dict:
    from movie_roi import predict
    if poster_bytes is None:
        return predict(record, artifact_dir=str(ARTIFACTS), model_name=model_name)
    # Decode and re-encode to avoid trusting a supplied filename or extension.
    with tempfile.TemporaryDirectory(prefix='movie-poster-') as folder:
        path = Path(folder) / 'poster.png'
        with Image.open(io.BytesIO(poster_bytes)) as image:
            image.convert('RGB').save(path)
        return predict(record, artifact_dir=str(ARTIFACTS), poster_path=str(path), model_name=model_name)


def pct(value) -> str:
    return 'Unavailable' if value is None else f'{value:.1%}'


def main():
    st.set_page_config(page_title='Frame / Movie ROI Lab', page_icon='🎬', layout='wide')
    st.markdown('''<style>
      .stApp {background: #f6f7f9;}
      .block-container {padding-top: 2.3rem; max-width: 1180px;}
      h1,h2,h3 {letter-spacing: -0.035em; color: #18263b;}
      [data-testid="stMetric"] {background:white;border:1px solid #e1e6ed;border-radius:14px;padding:18px;}
      [data-testid="stMetricLabel"] {color:#536279;}
      [data-testid="stMetricValue"] {color:#18263b;}
      [data-testid="stForm"] {background:white;border-radius:16px;}
      .eyebrow {color:#257870;letter-spacing:.16em;font-size:12px;font-weight:700;}
      .hero {color:#5b6678;font-size:17px;max-width:820px;margin-bottom:25px;}
      .stTabs [data-baseweb="tab-list"] {gap:28px;}
      footer {visibility:hidden;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">FRAME / ACADEMIC RESEARCH LAB</div>', unsafe_allow_html=True)
    st.title('A first look at film performance.')
    st.markdown('<div class="hero">Explore how pre-release metadata, a story synopsis, and an optional poster inform a movie’s modeled box-office outcome.</div>', unsafe_allow_html=True)
    metrics = read_json(ARTIFACTS / 'metrics.json')
    card = read_json(ARTIFACTS / 'model_card.json')
    ready = (ARTIFACTS / 'model.joblib').exists()
    with st.sidebar:
        st.subheader('🎬 Frame')
        st.caption('Movie ROI & risk · Research prototype')
        st.divider()
        if ready:
            st.success('Trained model available')
            st.caption(f"Selected model: {metrics.get('selected_model', 'See evaluation')}")
        else:
            st.warning('Training artifacts needed')
        st.markdown('**Inputs only available before release**')
        st.caption('Budget · Release date · Runtime · Genres · Plot · Optional poster')
        st.divider()
        st.caption('Box-office proxy ≠ investor return. Marketing costs, distribution fees and revenue sharing are not included.')
        st.caption('For academic demonstration only. Not an investment recommendation.')
    prediction_tab, evaluation_tab, about_tab = st.tabs(['Prediction', 'Evaluation', 'About'])
    with prediction_tab:
        if not ready:
            st.info('Train the model first. This interface will never substitute a made-up prediction.')
            st.code('python -m movie_roi train --data data/movies_multimodal.csv --output artifacts --regression-specs experiments/final_regression_specs.json --reused-benchmark', language='bash')
        model_mode = st.radio('Model to use', ['Multimodal demonstration', 'Validation-selected model'], horizontal=True, help='Multimodal uses metadata, text and poster features. The validation-selected model may use fewer modalities if they performed better.')
        scenario = st.selectbox('Start with a scenario', list(SCENARIOS), help='All built-in scenarios are fictional and have no known revenue.')
        sample = SCENARIOS[scenario]
        st.caption('Fictional examples are editable. Their predictions come from the trained model, not preset results.')
        with st.form('scenario_form'):
            left, right = st.columns([1.5, 1], gap='large')
            with left:
                title = st.text_input('Working title', sample['title'], key=f'title-{scenario}', max_chars=200)
                c1, c2 = st.columns(2)
                budget = c1.number_input('Production budget (USD)', min_value=1.0, max_value=1_000_000_000.0,
                    value=sample['budget'], step=100_000.0, format='%.0f', key=f'budget-{scenario}')
                runtime = c2.number_input('Runtime (minutes)', min_value=1, max_value=600,
                    value=sample['runtime'], key=f'runtime-{scenario}')
                release = st.date_input('Planned release date', value=dt.date(2027, 7, 16),
                    min_value=dt.date(1900, 1, 1), max_value=dt.date(2100, 12, 31), key=f'release-{scenario}')
                language = st.selectbox('Original language', ['en', 'fr', 'es', 'de', 'it', 'ja', 'ko', 'zh', 'hi', 'unknown'], format_func=lambda x: {'en':'English', 'fr':'French', 'es':'Spanish', 'de':'German', 'it':'Italian', 'ja':'Japanese', 'ko':'Korean', 'zh':'Chinese', 'hi':'Hindi', 'unknown':'Other / unknown'}[x], key=f'language-{scenario}')
                genres = st.multiselect('Genres', GENRES, default=sample['genres'], key=f'genres-{scenario}')
                overview = st.text_area('Plot overview', sample['overview'], height=130,
                    max_chars=10000, key=f'overview-{scenario}')
            with right:
                st.markdown('**Poster (optional)**')
                st.caption('Upload a PNG or JPEG you have permission to use. Without a valid poster, a separately trained metadata + plot model is used. Results identify the model actually used.')
                poster = st.file_uploader('Choose a poster', type=['png', 'jpg', 'jpeg'], key=f'poster-{scenario}')
                st.caption('Maximum 10 MB · 25 million pixels')
                st.info('Only pre-release signals belong here. Do not add ratings, actual revenue, popularity or opening-weekend results.')
            submitted = st.form_submit_button('Estimate performance', type='primary', disabled=not ready, width='stretch')
        if submitted:
            st.session_state.pop('prediction_result', None)
            record = {'title': title.strip(), 'budget': budget, 'runtime': runtime,
                'release_date': release.isoformat(), 'original_language': language, 'genres': '|'.join(genres), 'overview': overview.strip()}
            poster_bytes = poster.getvalue() if poster else None
            errors = validate_inputs(record, poster_bytes)
            for error in errors:
                st.error(error)
            if not errors:
                try:
                    with st.spinner('Combining pre-release signals…'):
                        result = run_prediction(record, poster_bytes, 'multimodal' if model_mode == 'Multimodal demonstration' else None)
                    st.session_state['prediction_result'] = {'record': record, 'result': result}
                except Exception as exc:
                    st.error(f'Prediction could not be completed: {exc}')
                    st.caption('Check that the model and its dependencies were trained/installed with the supplied requirements.')
        stored = st.session_state.get('prediction_result')
        if stored:
            result, record = stored['result'], stored['record']
            st.divider()
            st.subheader(f"Result · {record['title']}")
            st.caption(f"Model used: {result.get('selected_model', 'Not recorded')}. Snapshot of the last submitted inputs; submit again after editing.")
            c1, c2, c3 = st.columns(3)
            c1.metric('Predicted box-office revenue', f"${result.get('predicted_revenue', record['budget'] * (1 + result['predicted_roi'])):,.0f}")
            c2.metric('Revenue–budget proxy ROI', pct(result['predicted_roi']))
            c3.metric('Modeled P(revenue < budget)', pct(result.get('downside_probability')))
            interval = result.get('roi_interval', {})
            if interval:
                st.markdown(f"**ROI uncertainty range: {pct(interval.get('lower'))} to {pct(interval.get('upper'))}**")
                st.caption(f"Nominal coverage: {pct(interval.get('nominal_coverage'))}. Temporal benchmark coverage: {pct(interval.get('test_empirical_coverage'))}. Coverage is a dataset-level check, not a guarantee for this movie.")
            st.caption('Proxy ROI = (worldwide box-office revenue − production budget) / production budget. Revenue below budget is not the same as an investor-loss event.')
            for warning in result.get('warnings', []):
                st.warning(str(warning))
            with st.expander('Submitted inputs and model output'):
                st.json(stored)
            st.download_button('Download this result (JSON)', json.dumps(stored, indent=2),
                file_name='movie_scenario_result.json', mime='application/json')
    with evaluation_tab:
        st.subheader('Evidence before confidence.')
        st.caption('These values are loaded from the saved training run. They are never hand-entered demo metrics.')
        if not metrics:
            st.info('No evaluation has been saved yet. Train the model to populate this view.')
        else:
            if str(metrics.get('test_usage', '')).startswith('Reused'):
                st.warning('Exploratory reused benchmark: these movies were already evaluated before the follow-on model comparison. These results are not a fresh independent test.')
            st.markdown(f"**Selected model:** {metrics.get('selected_model', 'Not recorded')}")
            final = metrics.get('final_test', {})
            if final:
                columns = st.columns(3)
                for col, label, key in zip(columns, ['Benchmark ROI MAE', 'Benchmark ROC AUC', 'Benchmark Brier score'], ['mae', 'roc_auc', 'brier']):
                    val = final.get(key)
                    col.metric(label, f'{val:.3f}' if isinstance(val, (int, float)) else 'Unavailable')
                st.caption('MAE measures proxy ROI error (lower is better). ROC AUC measures ranking of revenue-below-budget risk (higher is better). Brier score measures probability error (lower is better).')
            baseline = metrics.get('baseline_test', {})
            if isinstance(final.get('mae'), (int, float)) and isinstance(baseline.get('mae'), (int, float)):
                if final['mae'] >= baseline['mae']:
                    st.warning('The selected model did not beat the simple baseline on benchmark ROI MAE. Treat these estimates as an experimental result, not reliable forecasting evidence.')
                st.caption(f"Selected-model ROI MAE: {final['mae']:.3f} · Baseline ROI MAE: {baseline['mae']:.3f}")
            multimodal = metrics.get('multimodal_final_test', {})
            if isinstance(multimodal.get('mae'), (int, float)) and isinstance(baseline.get('mae'), (int, float)) and multimodal['mae'] >= baseline['mae']:
                st.warning('The multimodal model did not beat the simple baseline on benchmark ROI MAE. Additional inputs have not demonstrated better ROI accuracy here.')
            if isinstance(final.get('r2'), (int, float)) and final['r2'] < 0:
                st.caption('Selected-model R² is negative: squared-error performance is worse than predicting the benchmark mean. A small MAE gain alone is weak evidence of forecasting value.')
            if metrics.get('multimodal_final_test'):
                st.markdown('**Multimodal temporal benchmark**')
                st.dataframe(pd.DataFrame([metrics['multimodal_final_test']], index=['multimodal']), width='stretch')
            if metrics.get('validation_ablations'):
                st.markdown('**Validation ablations**')
                st.dataframe(pd.DataFrame.from_dict(metrics['validation_ablations'], orient='index'), width='stretch')
            if metrics.get('splits'):
                st.markdown('**Chronological data splits**')
                st.dataframe(pd.DataFrame.from_dict(metrics['splits'], orient='index'), width='stretch')
            with st.expander('Complete metrics and baseline comparison'):
                st.json(metrics)
            st.download_button('Download evaluation (JSON)', json.dumps(metrics, indent=2), file_name='evaluation.json', mime='application/json')
    with about_tab:
        st.subheader('What this prototype can tell you')
        st.markdown('This project tests whether combining structured pre-release information, plot text and poster features improves an academic movie-performance model. It estimates a box-office/production-budget proxy and the probability that box-office revenue falls below that budget.')
        st.markdown('**Important limits**\n- Box office is not studio receipts or profit. Marketing, exhibitors’ share, financing, taxes and other income are omitted.\n- Dataset metadata may be recorded after release even when the feature itself could exist before release. This is a retrospective experiment.\n- Missing budgets/revenues, poster-availability selection bias, inflation, changing audience behavior and out-of-distribution stories limit generalization.\n- A visually richer model is not automatically more accurate. Compare the saved validation ablations and exploratory temporal benchmark results.\n- Do not interpret the output as a guaranteed return or a real-world investment recommendation.')
        if card:
            with st.expander('Model card and run provenance', expanded=True):
                st.json(card)
        st.caption('Runs locally. Uploaded posters are decoded to a temporary file for prediction and removed afterward. Only load model artifacts from a source you trust: joblib/pickle files can execute code.')


if __name__ == '__main__':
    main()
