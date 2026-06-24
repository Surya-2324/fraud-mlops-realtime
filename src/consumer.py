import json
import os
import redis
import mlflow.xgboost
import pandas as pd
from confluent_kafka import Consumer, Producer, KafkaError
from src.features import get_account_features, update_account_history

# ── CONFIG ────────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MLFLOW_URI   = os.getenv("MLFLOW_URI",   "http://localhost:5000")
MODEL_URI    = "models:/fraud-detector@Production"

# ── FEATURE COLUMNS — hardcoded to match training ─────────────────
FEATURE_COLS = [
    "amount", "hour_of_day", "account_age", "is_weekend",
    "rolling_avg_amount", "rolling_max_amount", "rolling_std_amount",
    "txn_count", "amount_vs_avg_ratio"
]

# ── CONNECTIONS ───────────────────────────────────────────────────
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

mlflow.set_tracking_uri(MLFLOW_URI)
print(f"⬇  Loading model from MLflow: {MODEL_URI}")
model = mlflow.xgboost.load_model(MODEL_URI)
print(f"✅ Model loaded | {len(FEATURE_COLS)} features")

# ── KAFKA ─────────────────────────────────────────────────────────
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id":          "fraud-consumer-group",
    "auto.offset.reset": "earliest",
})
consumer.subscribe(["transactions"])

pred_producer = Producer({"bootstrap.servers": KAFKA_BROKER})

# ── STATS ─────────────────────────────────────────────────────────
total       = 0
fraud_count = 0

print("👂 Listening on topic: transactions")
print("-" * 55)

# ── MAIN LOOP ─────────────────────────────────────────────────────
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        if msg.error().code() != KafkaError._PARTITION_EOF:
            print(f"❌ Error: {msg.error()}")
        continue

    # Parse transaction
    txn = json.loads(msg.value().decode("utf-8"))
    total += 1

    # Build feature vector
    rolling = get_account_features(txn["account_id"], txn["amount"])
    features = {
        "amount":              txn["amount"],
        "hour_of_day":         txn["hour_of_day"],
        "account_age":         txn["account_age"],
        "is_weekend":          txn["is_weekend"],
        "rolling_avg_amount":  rolling["rolling_avg_amount"],
        "rolling_max_amount":  rolling["rolling_max_amount"],
        "rolling_std_amount":  rolling["rolling_std_amount"],
        "txn_count":           rolling["txn_count"],
        "amount_vs_avg_ratio": rolling["amount_vs_avg_ratio"],
    }

    # Score with XGBoost
    X          = pd.DataFrame([features])[FEATURE_COLS]
    fraud_prob = float(model.predict_proba(X)[0][1])
    is_fraud   = fraud_prob > 0.7
    if is_fraud:
        fraud_count += 1

    # Update Redis history for this account
    update_account_history(txn["account_id"], txn["amount"])

    # Publish prediction to Kafka predictions topic
    result = {
        "transaction_id": txn["transaction_id"],
        "fraud_prob":     round(fraud_prob, 4),
        "is_fraud":       is_fraud,
        "features":       features,
        "timestamp":      txn["timestamp"],
    }
    pred_producer.produce("predictions", json.dumps(result).encode())
    pred_producer.poll(0)

    # Update live stats in Redis for dashboard
    r.set("stats:total_scored",  total)
    r.set("stats:fraud_rate",    round(fraud_count / max(total, 1), 4))
    r.set("stats:last_prob",     round(fraud_prob, 4))
    r.set("stats:model_version", "1")

    # Print to terminal
    flag = "🚨 FRAUD" if is_fraud else "   legit"
    print(f"{flag} | txn {txn['transaction_id']} | £{txn['amount']:>7.2f} | prob={fraud_prob:.3f} | {txn['merchant_cat']}")