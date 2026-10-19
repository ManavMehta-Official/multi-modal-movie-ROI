"""Synthetic fixtures test software correctness, never model performance claims."""
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from movie_roi.features import FeatureBuilder,poster_features,IMAGE_DIM
from movie_roi.model import chronological_split, read_data, train, predict

@pytest.fixture
def fixture_data(tmp_path):
    n=160; rng=np.random.default_rng(42)
    image=tmp_path/'poster.jpg';Image.new('RGB',(30,40),(200,30,10)).save(image)
    df=pd.DataFrame({'id':range(n),'title':['Synthetic '+str(i) for i in range(n)],'release_date':pd.date_range('1990-01-01',periods=n,freq='30D'),'budget':rng.uniform(1e6,20e6,n),'runtime':rng.integers(80,140,n),'genres':['Drama|Adventure']*n,'original_language':['en']*n,'overview':['Explorer discovers secret island' if i%2 else 'Family finds new home' for i in range(n)],'poster_path':[str(image)]*n})
    df['revenue']=df.budget*np.where(np.arange(n)%2,.5,3)
    path=tmp_path/'movies.csv';df.to_csv(path,index=False)
    return path

def test_poster_descriptors(tmp_path):
    assert len(poster_features(None))==IMAGE_DIM
    assert not poster_features('/missing/image.jpg')[-1]
    p=tmp_path/'p.png';Image.new('RGB',(24,30),'red').save(p)
    assert poster_features(p)[-1]==1
    assert np.isfinite(poster_features(p)).all()

def test_chronological_and_train_only_vocabulary(fixture_data):
    df,_=read_data(fixture_data);splits=chronological_split(df)
    assert splits['train'].release_date.max()<splits['validation'].release_date.min()
    assert splits['validation'].release_date.max()<splits['calibration'].release_date.min()
    assert splits['calibration'].release_date.max()<splits['test'].release_date.min()
    builder=FeatureBuilder().fit(splits['train'])
    heldout=splits['test'].copy();heldout['overview']='uniquetestonlyword'
    builder.transform(heldout)
    assert 'uniquetestonlyword' not in builder.text.vocabulary_

def test_target_and_invalid_financial_rows(fixture_data):
    df,_=read_data(fixture_data)
    assert np.allclose(df.roi,df.revenue/df.budget-1)
    raw=pd.read_csv(fixture_data);raw.loc[0,'budget']=0;raw.loc[1,'revenue']=0;raw.to_csv(fixture_data,index=False)
    cleaned,q=read_data(fixture_data)
    assert len(cleaned)==158
    assert q['excluded_or_duplicate_rows']==2

def test_end_to_end(fixture_data,tmp_path):
    out=tmp_path/'artifacts';metrics=train(fixture_data,out)
    assert set(metrics['validation_ablations'])=={'tabular','tabular_text','tabular_poster','multimodal'}
    assert (out/'selection_lock.json').exists()
    result=predict({'title':'Synthetic new sample','budget':2e6,'release_date':'2015-01-01','overview':'A family moves home','runtime':90,'genres':'Drama'},out,model_name='multimodal')
    assert 0<=result['downside_probability']<=1
    assert result['predicted_roi']>=-1
    assert result['roi_interval']['lower']<=result['predicted_roi']<=result['roi_interval']['upper']
    assert not result['poster_available']
    assert result['selected_model']=='tabular_text'
    assert result['requested_model']=='multimodal'
    assert result['used_modalities']==['metadata','plot']
    assert result['warnings']
    assert json.loads((out/'metrics.json').read_text())['final_test']['n']>0
    for budget in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError):predict({'budget':budget,'release_date':'2020-01-01'},out)

def test_training_poster_readiness_guard(fixture_data,tmp_path):
    df=pd.read_csv(fixture_data);df['poster_path']='';df.to_csv(fixture_data,index=False)
    with pytest.raises(ValueError,match='Finish poster download'):
        train(fixture_data,tmp_path/'not-trained',min_poster_coverage=.8)
    assert not (tmp_path/'not-trained'/'model.joblib').exists()
