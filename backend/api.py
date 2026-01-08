from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction Backend")

# Load trained model
model = joblib.load("models/model.pkl")

@app.get("/")
def health_check():
    return {"status": "Backend is running"}

@app.post("/predict")
def predict_churn(data: dict):
    """
    Expects customer features as JSON
    Returns churn probability and risk level
    """
    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]

    if prob < 0.33:
        risk = "Low"
    elif prob < 0.66:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk
    }
