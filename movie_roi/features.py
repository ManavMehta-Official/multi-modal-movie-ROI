"""Training-only vocabulary fitting and transparent poster-pixel descriptors."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

IMAGE_DIM = 73

def poster_features(path):
    """72 color/layout/edge features + explicit availability indicator, no external weights."""
    try:
        if not path or not Path(str(path)).is_file():
            return np.zeros(IMAGE_DIM, dtype=np.float32)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB').resize((64, 96))
            rgb = np.asarray(im, dtype=np.float32) / 255
            hsv = np.asarray(im.convert('HSV'), dtype=np.float32) / 255
        hist = np.concatenate([np.histogram(hsv[:, :, c], bins=8, range=(0, 1))[0] / (64*96) for c in range(3)])
        grid = np.concatenate([np.r_[rgb[y:y+32, x:x+32].mean((0, 1)), rgb[y:y+32, x:x+32].std((0, 1))] for y in (0,32,64) for x in (0,32)])
        gray = rgb.mean(2)
        grad = np.r_[np.abs(np.diff(gray, axis=0)).mean(), np.abs(np.diff(gray, axis=1)).mean(), gray.mean(), gray.std()]
        quantiles = np.quantile(gray, np.linspace(.1,.8,8))
        return np.r_[hist, grid, grad, quantiles, 1].astype(np.float32)
    except (OSError, ValueError, Image.DecompressionBombError):
        return np.zeros(IMAGE_DIM, dtype=np.float32)

def metadata_records(df):
    records=[]
    for r in df.to_dict('records'):
        dt=pd.to_datetime(r.get('release_date'), errors='coerce')
        runtime=pd.to_numeric(r.get('runtime'), errors='coerce')
        item={'log_budget': float(np.log1p(float(r['budget']))), 'runtime': float(runtime) if np.isfinite(runtime) else 100., 'runtime_missing':float(not np.isfinite(runtime)), 'year':float(dt.year), 'month':str(dt.month), 'language':str(r.get('original_language') or 'unknown')}
        genres = str(r.get('genres') or '').replace(',', '|').split('|')
        item.update({'genre='+g.strip():1. for g in genres if g.strip()})
        records.append(item)
    return records

class FeatureBuilder:
    def fit(self, df):
        self.metadata=DictVectorizer(sparse=False).fit(metadata_records(df))
        texts=df['overview'].fillna('').astype(str)
        self.text=TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', min_df=2, sublinear_tf=True)
        try:
            raw=self.text.fit_transform(texts)
        except ValueError:
            self.text=TfidfVectorizer(max_features=10000, min_df=1, token_pattern=r'(?u)\b\w+\b')
            raw=self.text.fit_transform(texts.replace('', 'missingplot'))
        k=max(1,min(32, raw.shape[1]-1, raw.shape[0]-1))
        self.svd=TruncatedSVD(n_components=k, random_state=42).fit(raw) if raw.shape[1]>1 else None
        return self
    def transform(self, df):
        raw=self.text.transform(df['overview'].fillna('').astype(str))
        return {'metadata':self.metadata.transform(metadata_records(df)), 'text':self.svd.transform(raw) if self.svd else raw.toarray(), 'poster':np.vstack([poster_features(p) for p in df['poster_path']])}

def combine(features, name):
    keys={'tabular':['metadata'], 'tabular_text':['metadata','text'], 'tabular_poster':['metadata','poster'], 'multimodal':['metadata','text','poster']}[name]
    return np.hstack([features[k] for k in keys])
