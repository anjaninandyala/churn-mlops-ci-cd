# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"


def load_data():
    print("📥 Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_data(df):
    print("🧹 Cleaning data...")

    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop ID column
    df = df.drop("customerID", axis=1)

    return df


def encode_features(df):
    print("🔤 Encoding categorical features...")

    categorical_cols = df.select_dtypes(include=["object"]).columns

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/label_encoders.pkl")

    return df


def scale_features(df):
    print("📏 Scaling numerical features...")

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return df


def save_processed_data(df):
    print("💾 Saving processed dataset...")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"✔ Processed data saved to {PROCESSED_DATA_PATH}")


def save_column_order(df):
    # Remove target column "Churn"
    feature_cols = [col for col in df.columns if col != "Churn"]

    joblib.dump(feature_cols, "models/columns.pkl")
    print("✔ Saved feature column order to models/columns.pkl")



def main():
    print("🚀 Starting data preprocessing pipeline...")

    df = load_data()
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    save_processed_data(df)
    save_column_order(df)

    print("✨ Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
