


# Containerization of Data Science Workflows for CI/CD  
## Telecom Customer Churn Prediction (MLOps Project)



## 📌 Project Overview
This project demonstrates how **data science workflows can be containerized and automated using CI/CD pipelines**.  
It focuses on predicting **customer churn in the telecom industry** using machine learning, while showcasing **DevOps and MLOps practices** such as Docker, GitHub Actions, and modular pipelines.

The system includes:
- A complete ML pipeline (data preprocessing → training → evaluation)
- A Streamlit-based interactive dashboard (frontend)
- A FastAPI backend for model inference
- Docker-based containerization
- Automated CI/CD using GitHub Actions

This project ensures **reproducibility, scalability, and automation** of machine learning workflows.

---

## 🎯 Problem Statement
Customer churn refers to customers discontinuing a service.  
In the telecom industry, churn rates can reach **15–25% annually**, making early prediction critical.

By predicting churn in advance, telecom companies can:
- Identify high-risk customers
- Apply targeted retention strategies
- Reduce customer loss and revenue impact

---

## 📊 Dataset
**Telco Customer Churn Dataset**

The dataset contains:
- Customer demographics (gender, senior citizen, dependents)
- Account information (tenure, contract type, billing method)
- Services used (internet, security, tech support, streaming)
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
   - ROC Curve and Confusion Matrix
5. **Model Selection**
   - Best-performing model selected automatically
6. **Model Deployment**
   - Served through a FastAPI backend
   - Visualized using Streamlit frontend

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
The Streamlit dashboard provides:
- KPI metrics (churn rate, high-risk customers)
- Segment analysis (contract type, internet service, payment method)
- Feature importance visualization
- Top at-risk customers table
- Individual customer churn prediction
- Retention recommendations for high-risk customers

---

## ⚙️ Backend (FastAPI)
- Provides REST APIs for churn prediction
- Separates model inference logic from the UI
- Enables modular and scalable architecture
- Streamlit frontend communicates with backend via HTTP requests

---

## 🐳 Containerization (Docker)
- Frontend and backend run in separate containers
- Ensures environment consistency across systems
- Simplifies deployment and scaling

Run the entire system using:
```bash
docker compose up --build
````

---

## 🔁 CI/CD Pipeline (GitHub Actions)

On every push to the main branch, the pipeline automatically:

1. Runs data preprocessing
2. Trains machine learning models
3. Evaluates model performance
4. Saves model and metrics as artifacts
5. Builds Docker images

This ensures **continuous integration and automation of ML workflows**.

---

## 🏗️ Project Structure

```
churn-mlops-ci-cd/
│
├── data/
│   ├── raw/                  # Original dataset (telco_churn.csv)
│   └── processed/            # Preprocessed dataset (auto-generated)
│
├── src/
│   ├── data_preprocessing.py # Data cleaning & feature engineering
│   ├── train_model.py        # Model training & selection
│   └── evaluate.py           # Model evaluation
│
├── models/
│   ├── model.pkl             # Best trained ML model
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── columns.pkl
│   └── metrics.json          # Evaluation summary
│
├── reports/
│   ├── cm_*.png              # Confusion matrices
│   └── roc_*.png             # ROC curves
│
├── backend/
│   ├── api.py                # FastAPI backend
│   └── __init__.py
│
├── app/
│   ├── app.py                # Streamlit frontend
│   └── Dockerfile            # Frontend Dockerfile
│
├── pipeline/
│   └── Dockerfile            # ML pipeline container
│
├── docker-compose.yml        # Orchestrates frontend + backend
├── requirements.txt          # Python dependencies
├── .github/workflows/
│   └── ci-cd.yaml             # CI/CD workflow
└── README.md
```

---

## 🚀 How to Run Locally

### Option 1: Docker (Recommended)

```bash
docker compose up --build
```

* Frontend: [http://localhost:8501](http://localhost:8501)
* Backend: [http://localhost:8000](http://localhost:8000)

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

* End-to-end automated ML pipeline
* Containerized frontend and backend
* CI/CD-enabled model training and evaluation
* Business-focused churn insights
* Clean and scalable MLOps architecture

---

## 🔮 Future Enhancements

* Database integration for prediction history
* Advanced hyperparameter tuning
* Cloud deployment (AWS / GCP)
* Role-based dashboards
* Real-time streaming data integration

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
B.Tech – Computer Science & Engineering
Final Year Project
**Title:** Containerization of Data Science Workflows for CI/CD

