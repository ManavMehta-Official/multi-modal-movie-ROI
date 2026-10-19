"""Reproducible chronological ablation, calibration and inference pipeline."""
import hashlib
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss, balanced_accuracy_score
from .features import FeatureBuilder, combine, poster_features

SEED=42
MODALITIES=('tabular','tabular_text','tabular_poster','multimodal')

def read_data(path):
    path=Path(path)
    df=pd.read_csv(path)
    required={'title','release_date','budget','revenue','overview'}
    if not required.issubset(df): raise ValueError('Missing columns: '+str(required-set(df)))
    original=len(df)
    for c in ('budget','revenue'): df[c]=pd.to_numeric(df[c],errors='coerce')
    df['release_date']=pd.to_datetime(df['release_date'],errors='coerce')
    # Zero revenue is indistinguishable from unreported revenue in TMDB snapshots.
    df=df[np.isfinite(df.budget)&np.isfinite(df.revenue)&(df.budget>0)&(df.revenue>0)&df.release_date.notna()].copy()
    idcol='id' if 'id' in df else None
    df=df.drop_duplicates(idcol or ['title','release_date'])
    for c,default in [('runtime',np.nan),('genres',''),('original_language','unknown'),('poster_path','')]:
        if c not in df: df[c]=default
    df['poster_path']=df['poster_path'].fillna('').map(lambda p:str((path.parent/str(p)).resolve()) if p and not Path(str(p)).is_absolute() else str(p))
    df=df.sort_values(['release_date','title']).reset_index(drop=True)
    df['roi']=df.revenue/df.budget-1
    if len(df)<100: raise ValueError('Need at least 100 valid rows for four chronological splits.')
    return df,{'raw_rows':original,'eligible_rows':len(df),'excluded_or_duplicate_rows':original-len(df),'zero_revenue_policy':'Excluded: zero may mean missing, creating selection bias; not treated as genuine zero receipts.','currency_assumption':'Budget and revenue are assumed comparable nominal USD, without inflation adjustment.','extreme_roi_rows_over_100':int((df.roi>100).sum())}

def chronological_split(df):
    # Keep movies released on the same calendar date in the same split.
    dates=df.release_date
    boundaries=[dates.iloc[int(len(df)*q)] for q in (.60,.75,.85)]
    out={'train':df[dates<boundaries[0]], 'validation':df[(dates>=boundaries[0])&(dates<boundaries[1])], 'calibration':df[(dates>=boundaries[1])&(dates<boundaries[2])], 'test':df[dates>=boundaries[2]]}
    if any(len(x)<5 for x in out.values()): raise ValueError('Insufficient distinct dates for chronological splits.')
    return out

def scores(y, pred, prob):
    labels=np.asarray(y)<0
    return {'mae':float(mean_absolute_error(y,pred)), 'rmse':float(np.sqrt(mean_squared_error(y,pred))), 'r2':float(r2_score(y,pred)), 'roc_auc':float(roc_auc_score(labels,prob)) if len(np.unique(labels))==2 else None, 'brier':float(brier_score_loss(labels,prob)), 'balanced_accuracy':float(balanced_accuracy_score(labels,np.asarray(prob)>=.5)), 'n':len(y), 'downside_prevalence':float(labels.mean())}

def reliability_bins(y, prob, n_bins=5):
    labels=np.asarray(y)<0; prob=np.asarray(prob)
    result=[]
    for lo in np.linspace(0,1,n_bins+1)[:-1]:
        hi=lo+1/n_bins; mask=(prob>=lo)&((prob<hi) if hi<.99999 else (prob<=hi))
        result.append({'lower':float(lo),'upper':float(hi),'n':int(mask.sum()),'mean_prediction':float(prob[mask].mean()) if mask.any() else None,'observed_fraction':float(labels[mask].mean()) if mask.any() else None})
    return result

def logit(prob):
    p=np.clip(prob,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def make_regressor(spec):
    if spec in ('hgb_squared_log','hgb_absolute_log'):
        return HistGradientBoostingRegressor(loss='absolute_error' if spec=='hgb_absolute_log' else 'squared_error',max_iter=120,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=SEED)
    if spec=='extratrees_log':
        return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=8,max_features=.75,n_jobs=2,random_state=SEED)
    if spec=='ridge_log':return make_pipeline(StandardScaler(),Ridge(alpha=10))
    raise ValueError('Unsupported final regression specification: '+str(spec))

def reg_predict(model,x):
    # Target is log(1+revenue/budget), not raw investor return.
    return np.maximum(-1,np.expm1(model.predict(x))-1)

@threadpool_limits.wrap(limits=2)
def train(data_csv, artifact_dir='artifacts', poster_dir=None, min_poster_coverage=0.0, regression_specs=None, reused_benchmark=False):
    df,quality=read_data(data_csv)
    audit_path=Path(data_csv).parent/'data_audit.json'
    if audit_path.exists():
        quality['source_audit']=json.loads(audit_path.read_text())
    cohort_path=Path(data_csv).parent/'cohort_audit.json'
    if cohort_path.exists():quality['cohort_audit']=json.loads(cohort_path.read_text())
    quality['training_csv']=Path(data_csv).name
    quality['image_complete_case_warning']='When training on the poster-available cohort, exclusion by image availability can bias the sample; all reported modality ablations use the same cohort.'
    if poster_dir:
        df['poster_path']=[str(Path(poster_dir)/Path(p).name) if p else '' for p in df.poster_path]
    if not 0<=min_poster_coverage<=1: raise ValueError('min_poster_coverage must be between 0 and 1.')
    available=float(np.mean([bool(p) and Path(p).is_file() for p in df.poster_path]))
    if available<min_poster_coverage:
        raise ValueError(f'Only {available:.1%} of poster files exist, below required {min_poster_coverage:.1%}. Finish poster download before multimodal training.')
    splits=chronological_split(df)
    builder=FeatureBuilder().fit(splits['train'])
    # No fitting, model choice, threshold choice or calibration uses final-test labels.
    feats={k:builder.transform(v) for k,v in splits.items() if k!='test'}
    models={}; ablations={}
    specs=regression_specs or {name:'hgb_squared_log' for name in MODALITIES}
    if set(specs)!=set(MODALITIES):raise ValueError('Provide a regression specification for every modality.')
    for name in MODALITIES:
        x=combine(feats['train'],name); y=splits['train'].roi.to_numpy()
        if len(np.unique(y<0))<2: raise ValueError('Training data needs both downside classes.')
        reg=make_regressor(specs[name]).fit(x,np.log1p(y+1))
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=SEED).fit(x,y<0)
        xv=combine(feats['validation'],name)
        ablations[name]=scores(splits['validation'].roi,reg_predict(reg,xv),clf.predict_proba(xv)[:,1])
        models[name]=(reg,clf)
    # Lock model decision using validation only. All-modality model remains accessible for demonstration.
    selected=min(MODALITIES,key=lambda n:ablations[n]['mae'])
    root=Path(artifact_dir); root.mkdir(parents=True,exist_ok=True)
    (root/'selection_lock.json').write_text(json.dumps({'selected_model':selected,'criterion':'lowest validation MAE','retained_models':sorted({selected,'multimodal','tabular_text'}),'validation_ablations':ablations,'seed':SEED,'regression_specs':specs,'reused_benchmark':reused_benchmark},indent=2))
    bundles={}
    for name in {selected,'multimodal','tabular_text'}:
        reg,clf=models[name]; xc=combine(feats['calibration'],name); yc=splits['calibration'].roi.to_numpy()
        cp=clf.predict_proba(xc)[:,1]
        calibrator=LogisticRegression(C=1.0,random_state=SEED).fit(logit(cp),yc<0) if len(np.unique(yc<0))==2 else None
        residuals=np.abs(yc-reg_predict(reg,xc))
        rank=min(len(yc),math.ceil((len(yc)+1)*.9))
        q=float(np.sort(residuals)[rank-1])
        bundles[name]={'regressor':reg,'classifier':clf,'calibrator':calibrator,'residual_quantile':q}
    # Final test is evaluated after locking the above decisions.
    ft=builder.transform(splits['test']); yt=splits['test'].roi.to_numpy()
    test_results={}; uncertainty={}
    test_predictions=splits['test'][['title','release_date','budget','revenue','roi']].copy()
    for name,b in bundles.items():
        xt=combine(ft,name); p=b['classifier'].predict_proba(xt)[:,1]
        raw_p=p.copy()
        if b['calibrator'] is not None:p=b['calibrator'].predict_proba(logit(p))[:,1]
        pred=reg_predict(b['regressor'],xt)
        test_results[name]=scores(yt,pred,p)
        test_results[name]['uncalibrated_brier']=float(brier_score_loss(yt<0,raw_p))
        test_results[name]['reliability_bins']=reliability_bins(yt,p)
        test_results[name]['calibration_method']='Held-out sigmoid logistic regression' if b['calibrator'] is not None else 'Uncalibrated: calibration slice has only one class'
        q=b['residual_quantile']; lo=np.maximum(-1,pred-q); hi=pred+q
        test_predictions[name+'_predicted_roi']=pred
        test_predictions[name+'_downside_probability']=p
        test_predictions[name+'_interval_lower']=lo
        test_predictions[name+'_interval_upper']=hi
        uncertainty[name]={'nominal_coverage':.9,'test_empirical_coverage':float(np.mean((yt>=lo)&(yt<=hi))), 'mean_interval_width':float(np.mean(hi-lo)), 'calibration_n':len(splits['calibration']), 'method':'Held-out absolute-residual interval; temporal shift breaks exchangeability, so 90% coverage is a target, not a guarantee.'}
    test_predictions.to_csv(root/'test_predictions.csv',index=False)
    ytrain=splits['train'].roi.to_numpy()
    baseline=scores(yt,np.full(len(yt),np.median(ytrain)),np.full(len(yt),np.mean(ytrain<0)))
    quality['poster_coverage_by_split']={k:float(np.mean([poster_features(p)[-1] for p in v.poster_path])) for k,v in splits.items()}
    metrics={'selected_model':selected,'validation_ablations':ablations,'final_test':test_results[selected],'multimodal_final_test':test_results['multimodal'],'text_fallback_final_test':test_results['tabular_text'],'test_by_model':test_results,'baseline_test':baseline,'splits':{k:{'n':len(v),'start':str(v.release_date.min().date()),'end':str(v.release_date.max().date())} for k,v in splits.items()},'uncertainty':uncertainty[selected],'multimodal_uncertainty':uncertainty['multimodal'],'uncertainty_by_model':uncertainty,'data_quality':quality,'classification_threshold':.5,'classification_threshold_reason':'Fixed 0.5 probability for descriptive binary metrics; no economic decision threshold was optimized.','regression_specs':specs,'test_usage':('Reused exploratory temporal benchmark: initial results were seen before a bounded train/validation-only follow-on model comparison. This is not a fresh independent test; see artifacts_initial and experiments/validation_candidates.json. No further test-guided tuning.' if reused_benchmark else 'Initial test evaluated once after validation choice and calibration; do not tune based on this result.')}
    artifact={'builder':builder,'models':bundles,'selected_model':selected,'metrics':metrics,'train_budget_range':[float(splits['train'].budget.min()),float(splits['train'].budget.max())],'train_year_range':[int(splits['train'].release_date.dt.year.min()),int(splits['train'].release_date.dt.year.max())]}
    joblib.dump(artifact,root/'model.joblib',compress=3)
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    card={'target':'Box-office ROI proxy = worldwide revenue / production budget - 1. Not investor profit.','point_estimate':'Inverse-transformed log(1+revenue/budget) prediction, using the documented per-modality objective. This is not a calibrated conditional expected ROI or expected revenue.','regression_specs':specs,'evaluation_history':metrics['test_usage'],'features':['budget','runtime','genres','original_language','release year/month','plot TF-IDF + train-only SVD','poster HSV histograms, spatial RGB statistics, grayscale texture'], 'excluded_features':['revenue as input','ROI as input','popularity','votes','ratings','post-release performance'], 'image_encoder':'Classical 73-dimensional pixel descriptors, not a pretrained neural embedding; no semantic claims.', 'text_encoder':'TF-IDF up to 10,000 unigrams/bigrams, train-only truncated SVD up to 32 dimensions.', 'limitations':['Poster-available complete-case cohort can introduce image availability selection bias.','Marketing/distribution costs and exhibitor splits unavailable.','Reported financial completeness creates selection bias.','Historical plots/posters may have been updated after release; pre-release provenance is unverified.','No inflation or currency verification.','Temporal drift, franchises, extreme ratios and missing posters weaken generalization.','Probability of proxy ROI below zero is not loss severity or investment advice.','Residual intervals are broad, empirical and not guaranteed under temporal shift.'],'seed':SEED,'versions':{'python':sys.version.split()[0],'scikit_learn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},'data_sha256':hashlib.sha256(Path(data_csv).read_bytes()).hexdigest(),'poster_manifest_sha256':hashlib.sha256('\n'.join(str(Path(p).name)+':'+hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df.poster_path if p and Path(p).is_file()).encode()).hexdigest(),'references':['https://scikit-learn.org/stable/modules/calibration.html','https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html']}
    (root/'model_card.json').write_text(json.dumps(card,indent=2))
    report=df[['title','release_date','budget','revenue','roi']].copy()
    report['split']=''
    for k,v in splits.items():report.loc[v.index,'split']=k
    report.to_csv(root/'split_manifest.csv',index=False)
    return metrics

@threadpool_limits.wrap(limits=2)
def predict(record,artifact_dir='artifacts',poster_path=None,model_name=None):
    """Load only artifacts you trained/trust. Never expose arbitrary pickle uploads."""
    budget=float(record.get('budget',0))
    if not np.isfinite(budget) or budget<=0:raise ValueError('Budget must be a finite number greater than zero.')
    dt=pd.to_datetime(record.get('release_date'),errors='coerce')
    if pd.isna(dt):raise ValueError('Provide a valid release_date (YYYY-MM-DD).')
    model=joblib.load(Path(artifact_dir)/'model.joblib')
    name=model_name or model['selected_model']
    if name not in model['models']:raise ValueError('Available models: '+', '.join(model['models']))
    r=dict(record,release_date=dt,budget=budget,overview=str(record.get('overview') or ''),poster_path=poster_path or record.get('poster_path') or '')
    df=pd.DataFrame([r]);features=model['builder'].transform(df)
    available=bool(features['poster'][0,-1]);warnings=[];requested_name=name
    if not available and name in ('multimodal','tabular_poster'):
        name='tabular_text'
        warnings.append('Poster missing or unreadable: using the separately trained and calibrated metadata + plot fallback model, not zero-image multimodal inference.')
    x=combine(features,name);b=model['models'][name]
    roi=float(reg_predict(b['regressor'],x)[0]);p=b['classifier'].predict_proba(x)[:,1]
    if b['calibrator'] is not None:p=b['calibrator'].predict_proba(logit(p))[:,1]
    if not available and requested_name not in ('multimodal','tabular_poster'):warnings.append('Poster missing or unreadable; the selected nonvisual model does not require it.')
    if 'poster' not in name and name!='multimodal' and requested_name==name:warnings.append('Validation-selected model does not use poster features; choose the multimodal demonstration to use all modalities.')
    if not r['overview'].strip():warnings.append('Plot missing: zero text features used.')
    if not model['train_budget_range'][0]<=budget<=model['train_budget_range'][1]:warnings.append('Budget is outside the training range.')
    if dt.year>model['train_year_range'][1]:warnings.append('Release year is beyond the training period; temporal extrapolation is uncertain.')
    u=model['metrics']['uncertainty_by_model'][name]
    return {'predicted_roi':roi,'predicted_revenue':budget*(1+roi),'downside_probability':float(p[0]),'roi_interval':{'lower':max(-1,roi-b['residual_quantile']),'upper':roi+b['residual_quantile'],'nominal_coverage':.9,'test_empirical_coverage':u['test_empirical_coverage']},'selected_model':name,'requested_model':requested_name,'used_modalities':{'tabular':['metadata'],'tabular_text':['metadata','plot'],'tabular_poster':['metadata','poster'],'multimodal':['metadata','plot','poster']}[name],'poster_available':available,'warnings':warnings,'risk_definition':'Probability that worldwide box-office revenue is below reported production budget. Not actual investor loss probability.'}
