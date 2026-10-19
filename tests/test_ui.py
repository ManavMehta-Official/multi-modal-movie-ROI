"""Frontend checks. Synthetic test fits verify wiring, not predictive quality."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import validate_inputs
from streamlit.testing.v1 import AppTest


def valid_record():
    return {'title':'Fictional test', 'budget':2_000_000, 'runtime':90,
            'release_date':'2027-01-01', 'genres':'Drama', 'overview':'A family reunites.'}


def test_input_validation():
    assert not validate_inputs(valid_record())
    for budget in (0, -1, float('nan'), float('inf'), 1e12):
        assert validate_inputs({**valid_record(), 'budget':budget})
    for key, value in [('title',''),('genres',''),('overview',''),('release_date','bad'),('runtime',0)]:
        assert validate_inputs({**valid_record(),key:value})
    assert validate_inputs(valid_record(),b'not an image')
    assert validate_inputs(valid_record(),b'x'*(10*1024*1024+1))
    image = io.BytesIO(); Image.new('RGB',(20,30),'blue').save(image,format='PNG')
    assert not validate_inputs(valid_record(),image.getvalue())


def test_missing_artifacts(monkeypatch,tmp_path):
    monkeypatch.setenv('MOVIE_ROI_ARTIFACTS',str(tmp_path/'missing'))
    at=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not at.exception
    assert [tab.label for tab in at.tabs]==['Prediction','Evaluation','About']
    assert at.button[0].disabled
    assert not at.metric


@pytest.fixture(scope='module')
def trained_ui_model(tmp_path_factory):
    from movie_roi import train
    root=tmp_path_factory.mktemp('ui-synthetic')
    n=160; rng=np.random.default_rng(42)
    d=pd.DataFrame({'id':range(n),'title':['Test '+str(i) for i in range(n)],
       'release_date':pd.date_range('1990-01-01',periods=n,freq='30D'),
       'budget':rng.uniform(1e6,20e6,n),'runtime':100,'genres':'Drama',
       'overview':['A family finds a new home' if i%2 else 'An explorer discovers an island' for i in range(n)],
       'original_language':'en'})
    d['revenue']=d.budget*np.where(np.arange(n)%2,.5,3)
    d.to_csv(root/'movies.csv',index=False)
    train(root/'movies.csv',root/'artifacts')
    return root/'artifacts'


def test_repeated_submits_and_invalid_input(monkeypatch,trained_ui_model):
    monkeypatch.setenv('MOVIE_ROI_ARTIFACTS',str(trained_ui_model))
    at=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not at.exception
    assert not at.button[0].disabled
    at.button[0].click().run(timeout=30)
    assert not at.exception
    first=at.session_state['prediction_result']['result']
    assert not first['poster_available']
    assert first['selected_model']=='tabular_text'
    assert first['requested_model']=='multimodal'
    at.button[0].click().run(timeout=30)
    assert at.session_state['prediction_result']['result']['predicted_roi']==first['predicted_roi']
    at.text_area[0].set_value('')
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert any('plot overview' in e.value for e in at.error)
    assert 'prediction_result' not in at.session_state
    at.selectbox[0].select('Fictional indie drama').run()
    assert not at.exception
    assert at.text_input[0].value=='A Place by the River'
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert at.session_state['prediction_result']['record']['title']=='A Place by the River'
    assert 'revenue' not in at.session_state['prediction_result']['record']


def test_uploaded_poster_prediction(monkeypatch,trained_ui_model):
    import app
    monkeypatch.setattr(app,'ARTIFACTS',trained_ui_model)
    image=io.BytesIO(); Image.new('RGB',(24,32),'red').save(image,format='PNG')
    result=app.run_prediction(valid_record(),image.getvalue(),model_name='multimodal')
    assert result['poster_available']
    assert 0 <= result['downside_probability'] <= 1
