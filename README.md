
# Containerization of Data Science Workflows for CI/CD  
### Telecom Customer Churn Prediction (MLOps Project)

## 📌 Project Overview
This project demonstrates how **Data Science workflows can be containerized and automated using CI/CD pipelines**.  
It focuses on predicting **customer churn in the telecom industry** using machine learning, while showcasing **DevOps + MLOps practices** such as Docker, GitHub Actions, and modular pipelines.

The system includes:
- A complete ML pipeline (data preprocessing → training → evaluation)
- A Streamlit-based interactive dashboard (frontend)
- A FastAPI backend for model inference
- Docker-based containerization
- Automated CI/CD using GitHub Actions

---

## 🎯 Problem Statement
Customer churn refers to customers leaving a service provider.  
In the telecom industry, churn rates can reach **15–25% annually**, making early prediction critical.

This project predicts churn in advance so that companies can:
- Identify high-risk customers
- Take preventive retention actions
- Reduce customer loss and revenue impact

---

## 📊 Dataset
**Telco Customer Churn Dataset**

Contains:
- Customer demographics (gender, senior citizen, dependents)
- Account information (tenure, contract, billing)
- Services used (internet, security, tech support)
- Target variable: `Churn`

---

## 🔄 Data Science Workflow
1. **Data Ingestion**
2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
3. **Model Training**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - (Optional) LightGBM, CatBoost
4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC Curve & Confusion Matrix
5. **Model Selection**
   - Best model selected automatically
6. **Model Deployment**
   - Served via backend API
   - Visualized using Streamlit

---

## 🧠 Machine Learning Models
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Voting / Best Model Selection  

The **best-performing model** is automatically saved and used for predictions.

---

## 🖥️ Frontend (Streamlit Dashboard)
- KPI metrics (Churn rate, High-risk customers)
- Segment analysis (Contract, Internet, Payment Method)
- Feature importance visualization
- Top at-risk customers table
- Individual customer churn prediction
- Retention recommendations for high-risk customers

---

## ⚙️ Backend (FastAPI)
- REST API for churn prediction
- Separates model logic from UI
- Enables scalable deployment
- Used by Streamlit for predictions

---

## 🐳 Containerization (Docker)
- Frontend and backend run in separate containers
- Ensures environment consistency
- Easily deployable on any system

Run everything with:
```bash
docker compose up --build
````

---

## 🔁 CI/CD Pipeline (GitHub Actions)

On every push:

1. Run data preprocessing
2. Train ML models
3. Evaluate model performance
4. Save model and metrics as artifacts
5. Build Docker image

This ensures **continuous integration and automation of ML workflows**.

---

## 🏗️ Project Structure

```
churn-mlops-ci-cd/
│
├── data/
│ ├── raw/ # Original dataset (telco_churn.csv)
│ └── processed/ # Preprocessed dataset (auto-generated)
│
├── src/
│ ├── data_preprocessing.py # Data cleaning & feature engineering
│ ├── train_model.py # Model training & selection
│ └── evaluate.py # Model evaluation
│
├── models/
│ ├── model.pkl # Best trained ML model
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ ├── columns.pkl
│ └── metrics.json # Evaluation summary
│
├── reports/
│ ├── cm_.png # Confusion matrices
│ └── roc_.png # ROC curves
│
├── backend/
│ ├── api.py # FastAPI backend
│ └── init.py
│
├── app/
│ ├── app.py # Streamlit frontend
│ └── Dockerfile # Frontend Dockerfile
│
├── pipeline/
│ └── Dockerfile # ML pipeline container
│
├── docker-compose.yml # Orchestrates frontend + backend
├── requirements.txt # Python dependencies
├── .github/workflows/
│ └── ci-cd.yaml # CI/CD workflow
└── README.md
```

---

## 🚀 How to Run Locally

### Option 1: Docker (Recommended)

```bash
docker compose up --build
```

Frontend: [http://localhost:8501](http://localhost:8501)
Backend: [http://localhost:8000](http://localhost:8000)

---

### Option 2: Without Docker

```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate.py
streamlit run app/app.py
```

---

## ✅ Key Outcomes

* Automated ML pipeline
* Containerized deployment
* CI/CD-enabled model training
* Business-focused churn insights
* Clean MLOps architecture

---

## 🔮 Future Enhancements

* Add database for prediction history
* Advanced hyperparameter tuning
* Cloud deployment (AWS / GCP)
* Role-based dashboards
* Real-time streaming data

---

## 📌 Technologies Used

* Python
* Scikit-learn
* Streamlit
* FastAPI
* Docker & Docker Compose
* GitHub Actions
* Pandas, NumPy, Matplotlib, Plotly

---

## 👤 Author

**Anjani Nandyala**
Third Year B.Tech (CSE)
