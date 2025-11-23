import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import logic from preprocessing.py
from src.preprocessing import feature_engineering

class Transaction(BaseModel):
    Time: float
    Amount: float
    class Config:
        extra = "allow" 

app = FastAPI(title="Fraud Detection API", version="1.0")

artifacts = {}

@app.on_event("startup")
def load_artifacts():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')

        print("Loading artifacts...")
        artifacts["model"] = joblib.load(os.path.join(models_dir, "model.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        artifacts["threshold"] = joblib.load(os.path.join(models_dir, "threshold.pkl"))
        print("Artifacts loaded successfully!")
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/")
def index():
    return {"status": "running", "message": "Fraud detection API is ready"}

@app.post("/predict")
def predict(transaction: Transaction):
    if not artifacts:
         raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    try:
        # Convert JSON to df
        input_data = transaction.dict()
        df = pd.DataFrame([input_data])
        
        # Feature engineering
        df_processed = feature_engineering(df)
        
        # Scaling amount
        df_processed['Amount'] = artifacts["scaler"].transform(df_processed[['Amount']])
        model = artifacts["model"]
        
        # Check what data model expects
        expected_cols = model.feature_names_in_
        df_processed = df_processed[expected_cols]
        
        # Prediction
        proba = model.predict_proba(df_processed)[0, 1]
        
        # Threshold decision
        threshold = artifacts["threshold"]
        is_fraud = bool(proba >= threshold)
        
        return {
            "fraud_probability": float(proba),
            "is_fraud": is_fraud,
            "threshold_used": float(threshold),
            "message": "ALERT: FRAUD DETECTED" if is_fraud else "Transaction Normal"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))