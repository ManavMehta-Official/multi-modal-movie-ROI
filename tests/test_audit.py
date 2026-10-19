"""Independent audit checks: semantics, source units, leakage and modality wiring."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from movie_roi.features import FeatureBuilder, combine, poster_features
from movie_roi.model import chronological_split, read_data, reg_predict, scores

ROOT = Path(__file__).resolve().parents[1]

def frame(n=120):
    return pd.DataFrame({'id':range(n), 'title':[f'Film {i}' for i in range(n)],
        'budget':np.full(n, 2_000_000.), 'revenue':np.where(np.arange(n)%2, 1_000_000., 6_000_000.),
        'release_date':pd.date_range('2000-01-01', periods=n, freq='7D'),
        'runtime':np.full(n,100), 'genres':['Drama']*n, 'original_language':['en']*n,
        'overview':['Family finds a secret home' if i%2 else 'Explorer travels across ocean' for i in range(n)],
        'poster_path':['']*n})

def test_target_inverse_and_probability_event():
    class Fixed:
        def predict(self, x): return np.log1p(np.array([0., .5, 1., 3.]))
    np.testing.assert_allclose(reg_predict(Fixed(), None), [-1., -.5, 0., 2.], atol=1e-12)
    result=scores(np.array([-.5, 0., 2.]), np.array([-.5, 0., 2.]), np.array([1.,0.,0.]))
    assert result['brier']==0
    assert result['mae']==0
    assert result['downside_prevalence']==1/3

def test_target_columns_do_not_enter_features():
    data=frame(); builder=FeatureBuilder().fit(data.iloc[:70])
    before=builder.transform(data.iloc[70:])
    modified=data.iloc[70:].copy()
    modified['revenue']=999_999_999_999
    modified['roi']=123456
    modified['popularity']=999999
    modified['vote_average']=10
    modified['vote_count']=999999
    after=builder.transform(modified)
    for key in before: np.testing.assert_array_equal(before[key], after[key])
    assert not any('revenue' in s or 'roi' in s or 'vote' in s or 'popularity' in s for s in builder.metadata.feature_names_)

def test_heldout_categories_and_tokens_never_refit():
    data=frame(); builder=FeatureBuilder().fit(data.iloc[:70])
    original_vocab=dict(builder.text.vocabulary_)
    original_names=list(builder.metadata.feature_names_)
    heldout=data.iloc[70:].copy()
    heldout['overview']='heldoutuniquetoken'
    heldout['genres']='HeldoutUniqueGenre'
    transformed=builder.transform(heldout)
    assert builder.text.vocabulary_==original_vocab
    assert builder.metadata.feature_names_==original_names
    assert np.isfinite(combine(transformed,'multimodal')).all()

def test_modality_dimensions_and_actual_poster_pixels(tmp_path):
    red=tmp_path/'red.png'; blue=tmp_path/'blue.png'
    Image.new('RGB',(40,60),'red').save(red)
    Image.new('RGB',(40,60),'blue').save(blue)
    assert not np.array_equal(poster_features(red),poster_features(blue))
    data=frame(); data.loc[0,'poster_path']=str(red)
    f=FeatureBuilder().fit(data).transform(data)
    assert f['poster'].shape==(len(data),73)
    assert f['poster'][0,-1]==1 and f['poster'][1,-1]==0
    widths={k:v.shape[1] for k,v in f.items()}
    assert combine(f,'multimodal').shape[1]==sum(widths.values())
    assert combine(f,'tabular_text').shape[1]==widths['metadata']+widths['text']
    assert combine(f,'tabular_poster').shape[1]==widths['metadata']+widths['poster']

def test_equal_dates_are_not_split():
    data=frame(); data['release_date']=pd.to_datetime('2000-01-01')+pd.to_timedelta(np.arange(len(data))//4,unit='D')
    splits=chronological_split(data)
    seen=set()
    for part in splits.values():
        assert not seen.intersection(set(part.release_date))
        seen.update(part.release_date)
    assert sum(map(len,splits.values()))==len(data)

def test_bundled_source_units_and_cohort():
    raw=pd.read_csv(ROOT/'data/raw/movies_complete.csv')
    cleaned,_=read_data(ROOT/'data/movies.csv')
    joined=cleaned.merge(raw[['id','budget_musd','revenue_musd']],on='id',validate='one_to_one')
    assert len(joined)==len(cleaned)==4910
    np.testing.assert_allclose(joined.budget, joined.budget_musd*1_000_000)
    np.testing.assert_allclose(joined.revenue, joined.revenue_musd*1_000_000)
    np.testing.assert_allclose(joined.roi, joined.revenue/joined.budget-1)
    assert cleaned.id.is_unique and cleaned.budget.ge(1_000_000).all()
    splits=chronological_split(cleaned)
    parts=list(splits.values())
    assert all(a.release_date.max()<b.release_date.min() for a,b in zip(parts,parts[1:]))

def test_frozen_complete_case_cohort_and_artifact_metrics():
    import hashlib
    import json
    import pytest
    path=ROOT/'data/movies_multimodal.csv'
    metric_path=ROOT/'artifacts/metrics.json'
    if not path.exists() or not metric_path.exists():
        pytest.skip('Frozen real-data artifact not present; synthetic software tests remain independent.')
    data,_=read_data(path)
    metrics=json.loads(metric_path.read_text())
    card=json.loads((ROOT/'artifacts/model_card.json').read_text())
    assert len(data)==2601
    assert card['data_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(poster_features(p)[-1]==1 for p in data.poster_path)
    assert sum(v['n'] for v in metrics['splits'].values())==len(data)
    splits=chronological_split(data)
    for name,part in splits.items(): assert len(part)==metrics['splits'][name]['n']
    pred=pd.read_csv(ROOT/'artifacts/test_predictions.csv')
    assert len(pred)==len(splits['test'])==391
    np.testing.assert_allclose(pred.roi,splits['test'].roi)
    for name, saved in metrics['test_by_model'].items():
        recomputed=scores(pred.roi,pred[name+'_predicted_roi'],pred[name+'_downside_probability'])
        for key in ['mae','rmse','r2','roc_auc','brier','balanced_accuracy']:
            np.testing.assert_allclose(recomputed[key],saved[key],atol=1e-12)
        observed=((pred.roi>=pred[name+'_interval_lower'])&(pred.roi<=pred[name+'_interval_upper'])).mean()
        np.testing.assert_allclose(observed,metrics['uncertainty_by_model'][name]['test_empirical_coverage'])
    baseline=scores(pred.roi,np.full(len(pred),splits['train'].roi.median()),np.full(len(pred),(splits['train'].roi<0).mean()))
    for key in ['mae','rmse','r2','roc_auc','brier']:
        np.testing.assert_allclose(baseline[key],metrics['baseline_test'][key],atol=1e-12)

def test_real_artifact_fallback_and_modalities():
    import json
    import pytest
    from movie_roi.model import predict
    out=ROOT/'artifacts'
    if not (out/'model.joblib').exists(): pytest.skip('Real-data artifact not yet present.')
    record={'budget':20_000_000,'runtime':110,'release_date':'2027-07-16','genres':'Drama','overview':'A family returns home','original_language':'en'}
    without=predict(record,out,model_name='multimodal')
    assert without['selected_model']=='tabular_text'
    assert without['requested_model']=='multimodal'
    assert without['used_modalities']==['metadata','plot']
    assert not without['poster_available']
    metrics=json.loads((out/'metrics.json').read_text())
    assert without['roi_interval']['test_empirical_coverage']==metrics['uncertainty_by_model']['tabular_text']['test_empirical_coverage']
    poster=next((ROOT/'data/posters').glob('*.jpg'))
    with_image=predict(record,out,poster_path=poster,model_name='multimodal')
    assert with_image['selected_model']=='multimodal' and with_image['poster_available']
    assert with_image['used_modalities']==['metadata','plot','poster']
    assert with_image['roi_interval']['test_empirical_coverage']==metrics['uncertainty_by_model']['multimodal']['test_empirical_coverage']
