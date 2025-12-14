# src/train_model.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

# Tree models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Optional models
try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except:
    CAT_AVAILABLE = False


# =====================================================
# PATHS
# =====================================================
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/model.pkl"
LEADERBOARD_PATH = "models/leaderboard.csv"
METRICS_PATH = "models/metrics.json"
REPORTS_DIR = "reports"

# SELECT RANKING METRIC (CHANGE THIS IF YOU WANT)
metric_to_rank_by = "accuracy"      # Options: accuracy, roc_auc, f1_score


# =====================================================
# LOAD DATA
# =====================================================
def load_processed_data():
    print("📥 Loading processed data...")
    return pd.read_csv(PROCESSED_DATA_PATH)


# =====================================================
# PLOTS
# =====================================================
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, val, ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =====================================================
# MODEL DICTIONARY
# =====================================================
def get_models():
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),
    }

    if LGB_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

    if CAT_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            random_state=42
        )

    return models


# =====================================================
# TRAIN + EVALUATE SINGLE MODEL
# =====================================================
def evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score))
    }

    return metrics, y_pred, y_score


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("🚀 Starting Auto-Model Training Pipeline...")

    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_processed_data()

    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    leaderboard = []

    best_model = None
    best_score = -1
    best_name = ""

    # Train each model
    for name, model in models.items():
        print(f"\n🔥 Training {name}...")

        metrics, y_pred, y_score = evaluate(model, X_train, X_test, y_train, y_test)

        print(f"✔ {name} — {metric_to_rank_by}: {metrics[metric_to_rank_by]:.4f}")

        leaderboard.append({"model": name, **metrics})

        # PLOTS
        plot_confusion(y_test, y_pred, f"{name} Confusion Matrix",
                       f"{REPORTS_DIR}/cm_{name}.png")

        plot_roc(y_test, y_score, f"{name} ROC Curve",
                 f"{REPORTS_DIR}/roc_{name}.png")

        # Pick best model according to selected metric
        if metrics[metric_to_rank_by] > best_score:
            best_score = metrics[metric_to_rank_by]
            best_model = model
            best_name = name

    # Save leaderboard
    lb_df = pd.DataFrame(leaderboard)
    lb_df.to_csv(LEADERBOARD_PATH, index=False)
    print("\n🏆 Leaderboard saved!")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Best Model Saved: {best_name}")

    # Save correct JSON for Streamlit
    summary = {
        "best_model": best_name,
        "best_metric_name": metric_to_rank_by,
        "best_metric_value": float(best_score)  
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n✨ Training Pipeline Complete!")
    print(f"🏆 Best Model: {best_name} ({metric_to_rank_by}={best_score:.4f})")


if __name__ == "__main__":
    main()
