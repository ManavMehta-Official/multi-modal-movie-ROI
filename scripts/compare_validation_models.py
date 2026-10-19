"""Bounded follow-on experiment. Reads train/validation only for fitting and scoring.
The earlier test has already been seen; this is exploratory validation development.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits
from movie_roi.model import read_data,chronological_split,MODALITIES
from movie_roi.features import FeatureBuilder,combine

@threadpool_limits.wrap(limits=2)
def main():
    d,_=read_data('data/movies_multimodal.csv');s=chronological_split(d)
    tr=s['train'];va=s['validation'];del s,d
    b=FeatureBuilder().fit(tr); ft=b.transform(tr);fv=b.transform(va)
    y=tr.roi.to_numpy();yv=va.roi.to_numpy(); logy=np.log1p(y+1)
    result={'scope':'Exploratory train/validation comparison after initial test was seen. This script never evaluates test rows.','baseline_validation_mae':float(mean_absolute_error(yv,np.full(len(yv),np.median(y)))),'candidates':{}}
    for name in MODALITIES:
        x=combine(ft,name);v=combine(fv,name)
        for spec in ['hgb_squared_log','hgb_absolute_log','hgb_absolute_raw','ridge_log','extratrees_log']:
            if spec.startswith('hgb'):
                model=HistGradientBoostingRegressor(loss='absolute_error' if 'absolute' in spec else 'squared_error',max_iter=120,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=42)
            elif spec=='ridge_log':model=make_pipeline(StandardScaler(),Ridge(alpha=10))
            else:model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=8,max_features=.75,n_jobs=2,random_state=42)
            raw=spec.endswith('_raw');model.fit(x,y if raw else logy)
            pred=model.predict(v);pred=pred if raw else np.expm1(pred)-1;pred=np.maximum(-1,pred)
            row={'mae':float(mean_absolute_error(yv,pred)),'rmse':float(np.sqrt(mean_squared_error(yv,pred))),'r2':float(r2_score(yv,pred))}
            result['candidates'][name+'__'+spec]=row
            print(name,spec,row,flush=True)
    result['best_by_validation_mae']=min(result['candidates'],key=lambda k:result['candidates'][k]['mae'])
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/validation_candidates.json').write_text(json.dumps(result,indent=2))
    print('BEST',result['best_by_validation_mae'],flush=True)
if __name__=='__main__':main()
