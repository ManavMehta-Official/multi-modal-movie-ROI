"""Validate cached posters and build an immutable-in-practice complete-case cohort.
No network access. Re-run only before training, never during a training run.
"""
import collections, hashlib, json
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    data=ROOT/'data';source=data/'movies.csv';d=pd.read_csv(source,keep_default_na=False)
    logs=data/'poster_download_log.jsonl';errors={}
    if logs.exists():
        for line in logs.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('status')=='failed':errors[int(r['id'])]=r.get('error','unknown')
    records=[]
    for row in d.to_dict('records'):
        p=data/row['poster_path'];status='not_acquired';digest=None;size=0
        if p.is_file():
            try:
                with Image.open(p) as im:im.verify()
                status='valid';digest=sha(p);size=p.stat().st_size
            except Exception as e:status='invalid_image'
        else:
            err=errors.get(int(row['id']),'')
            if '403' in err:status='access_denied'
            elif '404' in err or '410' in err:status='unavailable'
            elif err:status='download_failed'
        records.append({'id':int(row['id']),'poster_path':row['poster_path'],'status':status,'sha256':digest,'bytes':size,'last_error':errors.get(int(row['id']))})
    valid={r['id'] for r in records if r['status']=='valid'}
    cc=d[d.id.isin(valid)].sort_values(['release_date','id']).copy()
    cc.to_csv(data/'movies_multimodal.csv',index=False)
    manifest=data/'poster_manifest.json';manifest.write_text(json.dumps(records,indent=2))
    years=pd.to_datetime(d.release_date).dt.year;cy=pd.to_datetime(cc.release_date).dt.year
    yearly=pd.DataFrame({'full_cohort':years.value_counts(),'complete_case':cy.value_counts()}).fillna(0).astype(int).sort_index();yearly.index.name='release_year';yearly.to_csv(data/'cohort_by_year.csv')
    summary={'full_cohort_rows':len(d),'complete_case_rows':len(cc),'coverage':len(cc)/len(d),'status_counts':dict(collections.Counter(r['status'] for r in records)),'poster_bytes':sum(r['bytes'] for r in records),'full_csv_sha256':sha(source),'multimodal_csv_sha256':sha(data/'movies_multimodal.csv'),'poster_manifest_sha256':sha(manifest),'complete_case_date_min':cc.release_date.min(),'complete_case_date_max':cc.release_date.max(),'selection_rule':'Include every row whose matching cached poster passes PIL.verify; no filtering on revenue/ROI outcomes beyond original cohort filters.','acquisition_order':'Fixed seed42 shuffled full cohort; cache began with8 oldest films before shuffle.','missingness':'HTTP403 access denials, HTTP404 unavailable assets and unattempted images after broad access failures; downloads stopped after broad access failures; one denied URL was inadvertently retried by an in-flight batch and remained denied.','limitation':'Complete-case cohort is selected by image availability/acquisition and may differ from full cohort; ablations must use identical complete-case train/validation/calibration/test rows.'}
    (data/'cohort_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
