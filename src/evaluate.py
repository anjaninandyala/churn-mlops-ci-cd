# src/evaluate.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/eval_metrics.json"


def load_data():
    print("📥 Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def load_model():
    print("🔍 Loading trained model...")
    model = joblib.load(MODEL_PATH)
    return model


def evaluate(model, X_test, y_test):
    print("📊 Evaluating model performance...")

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    }

    print(f"✔ Metrics: {metrics}")
    return metrics


def save_metrics(metrics):
    print("💾 Saving metrics to JSON...")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✔ Metrics saved to {METRICS_PATH}")


def main():
    print("🚀 Starting evaluation pipeline...")

    df = load_data()

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model()
    metrics = evaluate(model, X_test, y_test)
    save_metrics(metrics)

    print("✨ Evaluation completed!")


if __name__ == "__main__":
    main()
