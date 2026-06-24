import json
import os
import time
import redis
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
from confluent_kafka import Consumer, KafkaError
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# ── CONFIG ────────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MLFLOW_URI   = os.getenv("MLFLOW_URI",   "http://localhost:5000")

# ── CONNECTIONS ───────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("fraud-detection-v1")
client = mlflow.MlflowClient(MLFLOW_URI)
r      = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_champion_auc() -> float:
    """Get ROC-AUC of the current Production model from MLflow."""
    try:
        alias = client.get_model_version_by_alias("fraud-detector", "Production")
        run   = client.get_run(alias.run_id)
        return float(run.data.metrics.get("roc_auc", 0.0))
    except Exception:
        return 0.0

def retrain_model():
    """Retrain XGBoost on fresh data and promote if better."""
    print("\n🔄 Starting retraining...")
    r.set("stats:retraining", "1")

    # Load reference data + simulate fresh labels
    df = pd.read_csv("data/processed/reference_data.csv")
    X  = df.drop(columns=["prediction"], errors="ignore")

    np.random.seed(int(time.time()) % 1000)
    y = (np.random.rand(len(X)) < 0.035).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    with mlflow.start_run(run_name=f"auto-retrain-{int(time.time())}"):
        new_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=99,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        new_model.fit(X_res, y_res, verbose=False)

        y_prob  = new_model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("roc_auc", new_auc)
        mlflow.log_param("trigger", "auto-drift-alert")

        new_version = mlflow.xgboost.log_model(
            new_model, "model",
            registered_model_name="fraud-detector"
        ).registered_model_version

    champion_auc = get_champion_auc()
    print(f"   Champion AUC  : {champion_auc:.4f}")
    print(f"   Challenger AUC: {new_auc:.4f}")

    if new_auc >= champion_auc:
        client.set_registered_model_alias(
            "fraud-detector", "Production", new_version
        )
        r.set("stats:model_version", new_version)
        print(f"   ✅ NEW MODEL PROMOTED → version {new_version}")
    else:
        print(f"   ❌ Challenger not better — champion retained")

    r.set("stats:retraining",   "0")
    r.set("stats:last_retrain", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("✅ Retraining complete")

# ── LISTEN FOR DRIFT ALERTS ───────────────────────────────────────
# "latest" = only process NEW alerts, ignore old ones already in Kafka
# "retrainer-group-new" = fresh group so it doesn't read old offsets
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id":          "retrainer-group-new",
    "auto.offset.reset": "latest",
})
consumer.subscribe(["drift-alerts"])
print("👂 Retrainer listening for drift alerts on Kafka...")
print("   Waiting for next drift alert from monitor...")

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() != KafkaError._PARTITION_EOF:
            print(f"❌ Error: {msg.error()}")
        continue

    alert = json.loads(msg.value().decode())
    print(f"\n🔔 Alert received | drift_score={alert['drift_score']:.3f}")
    retrain_model()