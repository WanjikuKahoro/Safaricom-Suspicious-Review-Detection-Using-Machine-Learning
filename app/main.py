from fastapi import FastAPI
import joblib
import pandas as pd

from .schemas import ReviewFeatures
from .feature_engineering import build_features

app = FastAPI(title = "Suspicious Review Prediction")

model_path = "model/model.joblib"
threshold_path = "model/threshold.joblib"

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)

@app.get('/start_server')
def start_server():
    return{'status':'server is running'}

@app.post("/predict")
def predict(features: ReviewFeatures):

    payload = features.model_dump()

    raw_df = pd.DataFrame([payload])

    X = build_features(raw_df)

    # predict probability of suspicious class
    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)

    return  {
        "is_suspicious_pred": pred,
        "suspicious_probability": float(proba),
        "threshold": threshold
    }
