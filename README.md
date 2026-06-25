# fraud-mlops-realtime
# fraud-mlops-realtime
# 🔴 Real-Time Fraud Detection MLOps Pipeline

![CI](https://github.com/Surya-2324/fraud-mlops-realtime/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Kafka](https://img.shields.io/badge/kafka-3.5-orange)
![MLflow](https://img.shields.io/badge/mlflow-2.14-blue)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

> A production-grade real-time MLOps pipeline that ingests live transaction streams via Apache Kafka, scores them with XGBoost, detects model drift using Evidently AI, and automatically retrains + promotes new models via MLflow — all containerised with Docker Compose and monitored via a live FastAPI dashboard.

**Inspired by JP Morgan's production fraud detection architecture** which runs 450+ ML models delivering $2B in annual value.

---

## 📊 Live Results

| Metric | Value |
|--------|-------|
| Transactions Processed | 25,000+ |
| Fraud Detection Rate | 7.56% |
| Model ROC-AUC | 1.0000 |
| Drift Score (PSI) | 0.889 |
| Features Drifted | 8 / 9 |
| Inference Latency | < 10ms |
| Throughput | 2 events/sec |

---


---

## 🛠️ Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Message Broker | Apache Kafka 3.5 | Industry standard for real-time event streaming. Named in 60%+ of senior ML Engineer JDs |
| Feature Store | Redis 7 | Sub-millisecond reads for rolling account features |
| ML Model | XGBoost + SHAP | Best tabular fraud detection algorithm. Used by JP Morgan, PayPal, Stripe |
| Experiment Tracking | MLflow 2.14 | Model registry with alias-based zero-downtime promotion |
| Drift Detection | Evidently AI | PSI, KL divergence, Wasserstein distance — 20+ statistical tests built in |
| API | FastAPI | Async REST endpoint with auto-refreshing live dashboard |
| Containerisation | Docker Compose | Entire system starts with one command |
| CI/CD | GitHub Actions | Tests run on every push — green badge on README |

---

## 🚀 Quick Start

**Prerequisites:** Docker Desktop, Python 3.11

```bash
# 1. Clone the repo
git clone https://github.com/Surya-2324/fraud-mlops-realtime.git
cd fraud-mlops-realtime

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure
docker-compose up -d

# 4. Train the model
python training/train.py

# 5. Start all services (each in a separate terminal)
python src/producer.py
python -m src.consumer
python src/monitor.py
python src/retrainer.py
uvicorn src.api:app --port 8000 --reload
```

Then open:
- **Live Dashboard** → http://localhost:8000
- **MLflow UI** → http://localhost:5000

---

## 📸 Screenshots

### Live Dashboard
<!-- Add your dashboard screenshot here -->
![Dashboard](docs/dashboard.png)

### MLflow Model Registry
<!-- Add your MLflow screenshot here -->
![MLflow](docs/mlflow.png)

### Evidently Drift Report
<!-- Add your drift report screenshot here -->
![Drift](docs/drift.png)

---

## 🔍 How It Works

### 1. Transaction Stream
The producer generates synthetic bank transactions every 0.5 seconds and publishes them to the Kafka `transactions` topic. Each transaction includes amount, merchant category, hour of day, account age, and weekend flag.

### 2. Real-Time Inference
The consumer reads from Kafka, pulls the account's rolling transaction history from Redis (last 10 transactions, stored per account_id), builds a 9-feature vector, and scores it with XGBoost in under 10ms. Predictions are published to the `predictions` topic.

### 3. Drift Detection
Every 2 minutes, Evidently AI reads the last 200 predictions and compares their feature distributions against the training data using PSI (Population Stability Index). If more than 30% of features have drifted (PSI > 0.2 per feature), it fires a drift alert to the `drift-alerts` Kafka topic.

### 4. Auto-Retraining
The retrainer listens to `drift-alerts`. When triggered, it retrains XGBoost on fresh data, evaluates against the current champion model, and promotes the new version to the MLflow `Production` alias only if ROC-AUC improves. The consumer automatically picks up the new model without restarting.

### 5. Live Dashboard
FastAPI reads live metrics from Redis every 3 seconds and serves them at `/metrics` (JSON) and `/` (HTML dashboard). Shows total scored, fraud rate, drift score, drifted columns, model version, and system status.

---

## 📁 Project Structure

├── src/

│   ├── producer.py      ← synthetic transaction stream

│   ├── consumer.py      ← Kafka consumer + XGBoost inference

│   ├── features.py      ← Redis feature store logic

│   ├── monitor.py       ← Evidently AI drift detection

│   ├── retrainer.py     ← auto-retraining + MLflow promotion

│   └── api.py           ← FastAPI live dashboard

├── training/

│   └── train.py         ← XGBoost training pipeline

├── tests/

│   └── test_producer.py ← unit tests

├── data/

│   └── processed/       ← reference data for drift detection

├── reports/             ← Evidently HTML drift reports

├── docker-compose.yml   ← Kafka + Redis + MLflow

├── requirements.txt

└── README.md

## 📈 Model Performance
Training Data : 50,000 synthetic transactions (3.5% fraud)

Algorithm     : XGBoost with SMOTE oversampling

ROC-AUC       : 1.0000

F1 Score      : 1.0000

Precision     : 1.00

Recall        : 1.00
