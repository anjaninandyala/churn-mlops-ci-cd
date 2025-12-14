
# 📊 Telco Customer Churn Prediction – MLOps Project

## 1. Project Overview
This project demonstrates **Containerization of Data Science Workflows for CI/CD** using Docker, Streamlit, and GitHub Actions.  
The workflow automates:

- Data preprocessing  
- Model training  
- Model evaluation  
- Frontend deployment with Streamlit  

It showcases **MLOps principles**, ensuring your ML pipeline runs consistently on any system.

---

## 2. Features
- Load and preprocess Telco customer dataset  
- Train a Logistic Regression model for churn prediction  
- Evaluate model performance (Accuracy, Precision, Recall, F1 Score)  
- Streamlit frontend for live predictions and metrics  
- Containerized using Docker (pipeline + frontend)  
- CI/CD with GitHub Actions

---

## 3. Project Structure

```

churn-prediction-project/
│
├── data/
│   ├── raw/                 # Original dataset (telco_churn.csv)
│   └── processed/           # Processed dataset (generated automatically)
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── model.pkl             # Saved ML model
│
├── app/
│   ├── app.py                # Streamlit frontend
│   └── Dockerfile            # Docker container for frontend
│
├── pipeline/
│   ├── Dockerfile            # Pipeline container
│   └── entrypoint.sh         # Script to run preprocessing, training, evaluation
│
├── .github/workflows/
│   └── ci-cd.yaml            # GitHub Actions workflow for CI/CD
│
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis
│
├── requirements.txt
└── README.md

````

---

## 4. Setup & Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd churn-prediction-project
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create data folders**

```bash
mkdir -p data/raw
mkdir -p data/processed
```

4. **Place dataset**
   Place `telco_churn.csv` inside `data/raw/`.

---

## 5. Run Locally (Python)

1. **Preprocess data**

```bash
python src/data_preprocessing.py
```

2. **Train model**

```bash
python src/train_model.py
```

3. **Evaluate model**

```bash
python src/evaluate.py
```

4. **Run Streamlit frontend**

```bash
streamlit run app/app.py
```

Open browser at: `http://localhost:8501`

---

## 6. Run with Docker

### 🟢 Pipeline Container

```bash
docker build -t churn-pipeline ./pipeline
docker run --rm churn-pipeline
```

### 🟢 Streamlit Frontend Container

```bash
docker build -t churn-streamlit ./app
docker run -p 8501:8501 churn-streamlit
```

---

## 7. CI/CD (GitHub Actions)

* Triggers on push or pull request to `main` branch
* Steps:

  1. Install dependencies
  2. Run preprocessing, training, evaluation
  3. Save model and metrics as artifacts
  4. Build Streamlit Docker image

---

## 8. Folder Contents

* `data/raw` → Original CSV
* `data/processed` → Preprocessed CSV
* `models` → Saved model and metrics
* `src` → ML pipeline scripts
* `app` → Streamlit UI + Dockerfile
* `pipeline` → Docker pipeline + entrypoint script
* `.github/workflows` → CI/CD YAML
* `notebooks` → EDA
* `requirements.txt` → Dependencies

---

## 9. References

* [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Streamlit Documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
* Docker Documentation: [https://docs.docker.com](https://docs.docker.com)
* GitHub Actions: [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

---

## 10. Authors

**Anjani Nandyala**
B.Tech CSE – Final Year
Project: Containerization of Data Science Workflows for CI/CD

```

---

If you want, I can **next create a ready-to-use project architecture diagram (`architecture_diagram.png`)** that matches this README and is perfect for your viva/demo.  

Do you want me to do that next?
```
