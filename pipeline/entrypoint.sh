#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Pipeline container started"

# Move to repo root (when container run with WORKDIR /app this is not necessary)
cd /app

# 1. Preprocess
echo "🔹 Running data preprocessing..."
python src/data_preprocessing.py

# 2. Train
echo "🔹 Training model..."
python src/train_model.py

# 3. Evaluate
echo "🔹 Evaluating model..."
python src/evaluate.py

# After running, model should be in models/model.pkl and metrics in models/metrics.json
echo "✅ Pipeline finished. model and metrics are available in /app/models"
