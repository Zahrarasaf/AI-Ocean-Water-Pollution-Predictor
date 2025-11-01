# src/make_dataset.py
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import radians, sin, cos, sqrt, asin

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c

def build(input_csv="data/raw/noaa_in_situ.csv", output_csv="data/processed/dataset_ready.csv"):
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    # rename heuristics (edit if your columns named differently)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'latitude' in lc: rename_map[c] = 'lat'
        if 'longitude' in lc: rename_map[c] = 'lon'
        if 'sst' in lc and 'degree' in lc: rename_map[c] = 'sst'
        if 'sss' in lc and 'psu' in lc and 'sat' not in lc: rename_map[c] = 'sss'
    df = df.rename(columns=rename_map)
    # parse time
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if time_cols:
        df['time_utc'] = pd.to_datetime(df[time_cols[0]], errors='coerce')
    else:
        df['time_utc'] = pd.NaT
    # numeric coercion
    for col in ['lat','lon','sst','sss']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
    # compute sss_grad_per_km with nearest neighbors
    coords = df[['lat','lon']].values
    try:
        nbrs = NearestNeighbors(n_neighbors=6).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        sss_vals = df['sss'].values if 'sss' in df.columns else np.full(len(df), np.nan)
        mean_grad = []
        for i, inds in enumerate(indices):
            diffs = []
            for j in inds[1:]:
                if np.isnan(sss_vals[i]) or np.isnan(sss_vals[j]):
                    continue
                dkm = haversine(df.loc[i,'lon'], df.loc[i,'lat'], df.loc[j,'lon'], df.loc[j,'lat'])
                if dkm == 0: continue
                diffs.append(abs(sss_vals[i]-sss_vals[j]) / dkm)
            mean_grad.append(np.nanmean(diffs) if diffs else np.nan)
        df['sss_grad_per_km'] = mean_grad
    except Exception as e:
        df['sss_grad_per_km'] = np.nan
    # temporal features
    df['dayofyear'] = df['time_utc'].dt.dayofyear
    df['hour'] = df['time_utc'].dt.hour
    # fill missing small fields with median
    for col in ['sst','sss','sss_grad_per_km','dayofyear','hour']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df.to_csv(output_csv, index=False)
    print(f"Saved processed dataset to {output_csv} (n={len(df)})")

if __name__ == "__main__":
    build()
