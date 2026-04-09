"""stream_processor.py
Compute per-source-window features from CSV and write to Parquet/CSV.
Usage:
  python stream_processor.py --csv /path/to/Midterm_53_group.csv --out data/flow_features.parquet --window 60
The script is resilient to slightly different column names. It creates columns:
  window_start, src_ip, proto_count, avg_len, std_len, pkt_count
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def detect_cols(df):
    # heuristics to find typical column names
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # possible names
    for target in ['time','timestamp','ts']:
        if target in cols:
            mapping['time'] = cols[target]; break
    for target in ['source','src','src_ip','source ip']:
        if target in cols:
            mapping['source'] = cols[target]; break
    for target in ['destination','dst','dst_ip','destination ip']:
        if target in cols:
            mapping['dest'] = cols[target]; break
    for target in ['length','len','packet length','pkt_len']:
        if target in cols:
            mapping['length'] = cols[target]; break
    for target in ['protocol','proto']:
        if target in cols:
            mapping['protocol'] = cols[target]; break
    # fallback names
    return mapping

def process(csv_path, out_path, window_seconds=60, chunksize=200000):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        mapping = detect_cols(chunk)
        if 'time' in mapping:
            chunk['ts'] = pd.to_numeric(chunk[mapping['time']], errors='coerce')
        else:
            chunk['ts'] = pd.Timestamp.now().astype(int)/1e9
        src_col = mapping.get('source', None)
        if src_col is None:
            # create a synthetic source grouping if missing
            chunk['Source'] = 'unknown'
            src_col = 'Source'
        else:
            chunk['Source'] = chunk[src_col].astype(str)
        length_col = mapping.get('length', None)
        if length_col is None:
            chunk['Length'] = pd.to_numeric(chunk.get('Length', 0), errors='coerce').fillna(0)
        else:
            chunk['Length'] = pd.to_numeric(chunk[length_col], errors='coerce').fillna(0)
        chunk['window_id'] = (chunk['ts'] // window_seconds).astype('Int64')
        # aggregate per window & source
        agg = chunk.groupby(['window_id','Source']).agg(
            window_start = ('window_id','min'),
            src_ip = ('Source','first'),
            proto_count = ('Length','count'),
            avg_len = ('Length','mean'),
            std_len = ('Length','std'),
            pkt_count = ('Length','count')
        ).reset_index(drop=True)
        frames.append(agg)
    if frames:
        feats = pd.concat(frames, ignore_index=True)
    else:
        feats = pd.DataFrame(columns=['window_start','src_ip','proto_count','avg_len','std_len','pkt_count'])
    feats['std_len'] = feats['std_len'].fillna(0.0)
    feats.to_parquet(outp, index=False)
    print(f"Saved features to {outp}, shape={feats.shape}")
    return feats

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='data/flow_features.parquet')
    p.add_argument('--window', type=int, default=60)
    args = p.parse_args()
    process(args.csv, args.out, window_seconds=args.window)

if __name__ == '__main__':
    main()
