# Anomaly Project v2 - R3-ready bundle
Dataset included: /mnt/data/Midterm_53_group.csv

Quick steps (single-machine mode):
1. Create virtual environment and install requirements:
   pip install -r requirements.txt
2. Extract features from CSV:
   python stream_processor.py --csv Midterm_53_group.csv --out data/flow_features.parquet --window 60
3. Train models:
   python model_train.py --features data/flow_features.parquet
4. Optionally start model server:
   uvicorn model_server:app --reload --port 8000
5. Run dashboard:
   streamlit run dashboard.py

Notes:
- The pipeline works without Kafka/Spark for demonstration. For high-throughput use Kafka + Spark (see earlier bundle notes).
- Produced files: data/flow_features.parquet, data/flow_features_with_preds.parquet, models/iso_model.joblib, models/autoencoder_saved/
