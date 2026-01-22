from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

# Initialize App
app = FastAPI(title="Medical Churn Predictor", version="1.0")

# Load Model (Global State)
try:
    model = joblib.load("churn_model_lgb.pkl")
    print("Model loaded successfully.")
except:
    model = None
    print("WARNING: Model not found. Run train_pipeline.py first.")

# Define Input Schema
class HospitalStats(BaseModel):
    avg_img_mean: float
    avg_img_contrast: float
    primary_modality: int  # 0=CT, 1=MR
    scan_count: int

@app.get("/")
def home():
    return {"message": "Medical Imaging Churn API is running"}

@app.post("/predict")
def predict_churn(stats: HospitalStats):
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Prepare features in the exact order trained
    # Features: [avg_img_mean, avg_img_contrast, primary_modality, scan_count]
    features = np.array([[
        stats.avg_img_mean,
        stats.avg_img_contrast,
        stats.primary_modality,
        stats.scan_count
    ]])

    # Inference
    churn_prob = model.predict(features)[0]
    churn_prediction = int(churn_prob > 0.5)

    return {
        "churn_probability": float(churn_prob),
        "is_churn_risk": bool(churn_prediction),
        "risk_level": "HIGH" if churn_prob > 0.7 else "LOW"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)