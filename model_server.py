"""model_server.py - FastAPI model server
Run: uvicorn model_server:app --reload --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np, os, tensorflow as tf

class WindowFeatures(BaseModel):
    proto_count: float
    avg_len: float
    std_len: float
    pkt_count: float

app = FastAPI(title="Anomaly Model Server")

iso = None; ae = None
def load_models():
    global iso, ae
    if iso is None:
        iso = joblib.load('models/iso_model.joblib')
    if ae is None:
        ae = tf.keras.models.load_model('models/autoencoder_saved.keras')
    return iso, ae

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(w: WindowFeatures):
    iso, ae = load_models()
    x = np.array([[w.proto_count, w.avg_len, w.std_len, w.pkt_count]], dtype=float)
    iso_raw = int(iso.predict(x)[0])
    recon = ae.predict(x)
    mse = float(np.mean((x - recon)**2))
    return {'iso_raw': iso_raw, 'is_anomaly_iso': iso_raw == -1, 'recon_mse': mse, 'is_anomaly_ae': mse > 1.0}
