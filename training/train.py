import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# ── MLflow config ─────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection-v1")

# ── Generate synthetic fraud data ─────────────────────────────────
def generate_data(n=50_000):
    """
    Generate realistic synthetic fraud data.
    3.5% fraud rate — mirrors real-world class imbalance.
    """
    np.random.seed(42)
    n_fraud = int(n * 0.035)
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount":              np.random.exponential(50, n_legit),
        "hour_of_day":         np.random.randint(8, 22, n_legit),
        "account_age":         np.random.randint(100, 3000, n_legit),
        "is_weekend":          np.random.randint(0, 2, n_legit),
        "rolling_avg_amount":  np.random.exponential(45, n_legit),
        "rolling_max_amount":  np.random.exponential(80, n_legit),
        "rolling_std_amount":  np.random.uniform(5, 50, n_legit),
        "txn_count":           np.random.randint(1, 10, n_legit),
        "amount_vs_avg_ratio": np.random.uniform(0.5, 2.0, n_legit),
        "label":               0
    })

    fraud = pd.DataFrame({
        "amount":              np.random.exponential(800, n_fraud),
        "hour_of_day":         np.random.choice([1, 2, 3, 23], n_fraud),
        "account_age":         np.random.randint(1, 60, n_fraud),
        "is_weekend":          np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),
        "rolling_avg_amount":  np.random.exponential(30, n_fraud),
        "rolling_max_amount":  np.random.exponential(900, n_fraud),
        "rolling_std_amount":  np.random.uniform(100, 500, n_fraud),
        "txn_count":           np.random.randint(1, 5, n_fraud),
        "amount_vs_avg_ratio": np.random.uniform(5, 25, n_fraud),
        "label":               1
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df.drop("label", axis=1), df["label"]

# ── Train ──────────────────────────────────────────────────────────
print("📊 Generating synthetic fraud data...")
X, y = generate_data()
print(f"   Total rows: {len(X):,} | Fraud: {y.sum():,} ({y.mean()*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⚖️  Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {len(X_res):,} rows")

print("🌲 Training XGBoost...")
with mlflow.start_run(run_name="xgboost-baseline"):
    params = {
        "n_estimators":   300,
        "max_depth":      6,
        "learning_rate":  0.05,
        "scale_pos_weight": 99,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_res, y_res, verbose=False)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("f1_score", f1)

    print(f"\n✅ ROC-AUC : {roc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Save reference data for Evidently drift detection later
    os.makedirs("data/processed", exist_ok=True)
    ref = X_test.copy()
    ref["prediction"] = y_prob
    ref.to_csv("data/processed/reference_data.csv", index=False)
    print("💾 Reference data saved → data/processed/reference_data.csv")

    # Register model in MLflow
    mlflow.xgboost.log_model(
        model, "model",
        registered_model_name="fraud-detector",
        input_example=X_test.iloc[:1]
    )
    print("📦 Model registered in MLflow as 'fraud-detector'")

# Promote to Production alias
client = mlflow.MlflowClient("http://localhost:5000")
client.set_registered_model_alias("fraud-detector", "Production", "1")
print("🚀 Model promoted to Production alias")
print("\n✅ Training complete — open http://localhost:5000 to see it in MLflow")