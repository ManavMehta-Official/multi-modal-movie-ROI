"""Download attributed public movie metadata and optionally cache bounded thumbnails.
No API keys required. For noncommercial educational research; see DATA_SOURCES.md.
"""
import argparse, csv, hashlib, io, json, re, time, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://raw.githubusercontent.com/TjanMichela/Movie-Data-Analysis/main/movies_complete.csv'
LICENSE='https://raw.githubusercontent.com/TjanMichela/Movie-Data-Analysis/main/LICENSE'
def fetch(url,path):
    with urllib.request.urlopen(url,timeout=90) as response: content=response.read()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
def prepare():
    raw=ROOT/'data/raw/movies_complete.csv'
    if not raw.exists(): fetch(SOURCE,raw)
    license_path=ROOT/'data/raw/LICENSE'
    if not license_path.exists(): fetch(LICENSE,license_path)
    d=pd.read_csv(raw);raw_rows=len(d)
    d['release_date']=pd.to_datetime(d.release_date,errors='coerce')
    d['budget']=pd.to_numeric(d.budget_musd,errors='coerce')*1_000_000
    d['revenue']=pd.to_numeric(d.revenue_musd,errors='coerce')*1_000_000
    d=d[(d.budget>=1_000_000)&(d.revenue>0)&d.release_date.notna()&d.release_date.dt.year.between(1920,2016)].copy()
    d=d.sort_values(['release_date','id']).drop_duplicates('id')
    d['movie_id']=d['id'].astype(int)
    d['poster_url']=d.poster_path.fillna('').str.extract(r"src=['\"]([^'\"]+)",expand=False).fillna('')
    d['poster_url']=d.poster_url.str.replace('http://','https://',regex=False).str.replace('/w185//','/w185/',regex=False)
    d['poster_path']=d.movie_id.map(lambda x:f'posters/{x}.jpg')
    d['poster_local_path']=d.poster_path
    d['release_date']=d.release_date.dt.strftime('%Y-%m-%d')
    cols=['id','movie_id','title','budget','revenue','release_date','runtime','genres','overview','original_language','poster_path','poster_local_path','poster_url','production_companies','director','cast']
    d[cols].fillna('').to_csv(ROOT/'data/movies.csv',index=False)
    audit={'source_url':SOURCE,'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'raw_rows':raw_rows,'retained_rows':len(d),'unique_ids':d.id.nunique(),'date_min':d.release_date.min(),'date_max':d.release_date.max(),'missing_overview':int(d.overview.isna().sum()),'missing_poster_url':int(d.poster_url.eq('').sum()),'filters':['budget >= 1,000,000 USD','revenue > 0 USD','release year 1920–2016','deduplicate movie id'],'selection_warning':'Unknown/zero revenue excluded; observed-reporting selection bias. Historical metadata not guaranteed as-of-release snapshots.','units':'budget and revenue converted from million USD to nominal USD'}
    (ROOT/'data/data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
    return d

def posters(d,limit,workers,batch_size=0):
    if (ROOT/"data/DOWNLOAD_STOPPED.json").exists():
        raise RuntimeError("Poster acquisition is stopped after access restrictions; use the supplied cache and freeze_data.py. Do not bypass denied targets.")
    jobs=d.sample(frac=1,random_state=42).iloc[:limit].to_dict('records');out=ROOT/'data/posters';out.mkdir(exist_ok=True)
    permanent_failures=set()
    log_path=ROOT/'data/poster_download_log.jsonl'
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                record=json.loads(line)
                if any(code in record.get('error','') for code in ['HTTP Error 404','HTTP Error 410','403']): permanent_failures.add(int(record['id']))
            except (ValueError,KeyError): pass
    jobs=[r for r in jobs if int(r['id']) not in permanent_failures]
    if batch_size:
        jobs=[r for r in jobs if not (ROOT/'data'/r['poster_path']).exists()][:batch_size]
    gate=threading.Lock(); next_start=[0.0]
    def one(row):
        path=ROOT/'data'/row['poster_path'];url=row['poster_url']
        if path.exists():return {'id':row['id'],'status':'cached','bytes':path.stat().st_size}
        if not url.startswith('https://image.tmdb.org/t/p/w185/'):return {'id':row['id'],'status':'invalid_url','bytes':0}
        # Globally cap starts at 2 requests/second, including cold-cache bursts.
        with gate:
            time.sleep(max(0,next_start[0]-time.monotonic()))
            next_start[0]=time.monotonic()+0.5
        try:
            with urllib.request.urlopen(url,timeout=30) as r:
                content=r.read(2_000_000)
            with Image.open(io.BytesIO(content)) as image: image.verify()
            path.write_bytes(content)
            return {'id':row['id'],'status':'downloaded','bytes':len(content)}
        except urllib.error.HTTPError as exc:
            if exc.code==429:
                with gate: next_start[0]=max(next_start[0],time.monotonic()+60)
            return {'id':row['id'],'status':'failed','bytes':0,'error':str(exc)}
        except Exception as exc:return {'id':row['id'],'status':'failed','bytes':0,'error':str(exc)}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(one,r) for r in jobs]):
            result=future.result();results.append(result)
            with (ROOT/'data/poster_download_log.jsonl').open('a') as log: log.write(json.dumps(result)+'\n')
            if len(results)%100==0:print(f'Posters attempted {len(results)}/{len(jobs)}; usable {sum(r["bytes"]>0 for r in results)}',flush=True)
    (ROOT/'data/poster_download_log.json').write_text(json.dumps(results,indent=2))
    print('POSTERS',len(results),sum(r['bytes']>0 for r in results),sum(r['bytes'] for r in results),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--posters',action='store_true');parser.add_argument('--poster-limit',type=int,default=5000);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--batch-size',type=int,default=0);args=parser.parse_args()
    d=prepare()
    if args.posters:posters(d,max(0,min(args.poster_limit,5000)),max(1,min(args.workers,32)),args.batch_size)
