"""model_train.py
Train IsolationForest and Autoencoder on features produced by stream_processor.py
Usage:
  python model_train.py --features data/flow_features.parquet
Outputs:
  - models/iso_model.joblib
  - models/autoencoder_saved/
  - data/flow_features_with_preds.parquet
"""
import argparse, joblib, numpy as np, pandas as pd, os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def train(features_path):
    Path('models').mkdir(exist_ok=True)
    df = pd.read_parquet(features_path)
    if df.empty:
        print("No features found. Run stream_processor.py first."); return
    X = df[['proto_count','avg_len','std_len','pkt_count']].fillna(0).values
    # IsolationForest (train ideally on benign windows only if labels available)
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(X)
    joblib.dump(iso, 'models/iso_model.joblib')
    print("Saved IsolationForest to models/iso_model.joblib")
    # Autoencoder
    input_dim = X.shape[1]
    tf.random.set_seed(42)
    ae = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X, X, epochs=15, batch_size=128, validation_split=0.1, verbose=2)
    ae.save('models/autoencoder_saved.keras')
    print("Saved Autoencoder to models/autoencoder_saved/")
    # produce predictions and basic metrics (if label present)
    preds = iso.predict(X)  # 1 normal, -1 anomaly
    df['iso_pred'] = (preds == -1).astype(int)
    # if label column exists (case-insensitive), try to compute metrics
    label_col = None
    for c in df.columns:
        if c.lower() in ['label','label_name','info','attack','attack_label']:
            label_col = c; break
    if label_col is not None:
        y = df[label_col].apply(lambda v: 1 if str(v).strip().lower() not in ['benign','normal','0','nan','none'] else 0)
        print("Classification report for IsolationForest (using provided labels):")
        print(classification_report(y, df['iso_pred']))
    df.to_parquet('data/flow_features_with_preds.parquet', index=False)
    print("Saved predictions to data/flow_features_with_preds.parquet")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='data/flow_features.parquet')
    args = p.parse_args()
    train(args.features)
