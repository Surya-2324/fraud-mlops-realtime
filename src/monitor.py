import json
import os
import time
import schedule
import pandas as pd
import redis
from confluent_kafka import Consumer as KConsumer, Producer
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ── CONFIG ────────────────────────────────────────────────────────
KAFKA_BROKER    = os.getenv("KAFKA_BROKER", "localhost:9092")
DRIFT_THRESHOLD = 0.3
WINDOW_SIZE     = 200

# ── CONNECTIONS ───────────────────────────────────────────────────
r        = redis.Redis(host="localhost", port=6379, decode_responses=True)
producer = Producer({"bootstrap.servers": KAFKA_BROKER})

# ── REFERENCE DATA ────────────────────────────────────────────────
print("📂 Loading reference data...")
reference_df = pd.read_csv("data/processed/reference_data.csv")
FEATURE_COLS = [c for c in reference_df.columns if c != "prediction"]
reference_df = reference_df[FEATURE_COLS]
print(f"✅ Reference data loaded | {len(reference_df)} rows | {len(FEATURE_COLS)} features")

# ── READ RECENT PREDICTIONS FROM KAFKA ───────────────────────────
def read_recent_predictions(n=WINDOW_SIZE) -> pd.DataFrame:
    """Read the last N predictions from the Kafka predictions topic."""
    consumer = KConsumer({
        "bootstrap.servers":  KAFKA_BROKER,
        "group.id":           f"monitor-reader-{int(time.time())}",
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe(["predictions"])

    records    = []
    timeout_at = time.time() + 15

    while len(records) < n and time.time() < timeout_at:
        msg = consumer.poll(1.0)
        if msg and not msg.error():
            data = json.loads(msg.value().decode())
            records.append(data["features"])

    consumer.close()
    return pd.DataFrame(records) if records else pd.DataFrame()

# ── DRIFT CHECK ───────────────────────────────────────────────────
def run_drift_check():
    print(f"\n🔍 Running drift check... [{time.strftime('%H:%M:%S')}]")

    current_df = read_recent_predictions()

    if current_df.empty or len(current_df) < 30:
        print("   ⚠  Not enough predictions yet — skipping")
        return

    # Keep only shared columns
    shared = [c for c in FEATURE_COLS if c in current_df.columns]
    ref    = reference_df[shared]
    curr   = current_df[shared]

    # Run Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    # Extract results
    result_dict   = report.as_dict()
    metrics       = result_dict["metrics"][0]["result"]
    share_drifted = metrics.get("share_of_drifted_columns", 0)
    n_drifted     = metrics.get("number_of_drifted_columns", 0)

    print(f"   Drifted columns : {n_drifted} / {len(shared)}")
    print(f"   Share drifted   : {share_drifted:.3f}")

    # Save HTML report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/drift_{int(time.time())}.html"
    report.save_html(report_path)
    print(f"   📄 Report saved → {report_path}")

    # Update Redis for dashboard
    r.set("stats:drift_score", round(share_drifted, 4))
    r.set("stats:drift_cols",  n_drifted)

    # Trigger retraining if drift exceeds threshold
    if share_drifted > DRIFT_THRESHOLD:
        alert = {
            "drift_score": share_drifted,
            "n_drifted":   n_drifted,
            "timestamp":   time.time(),
            "action":      "RETRAIN",
        }
        producer.produce("drift-alerts", json.dumps(alert).encode())
        producer.poll(0)
        print(f"   🚨 DRIFT DETECTED ({share_drifted:.3f} > {DRIFT_THRESHOLD}) → alert sent to Kafka")
    else:
        print(f"   ✅ No significant drift ({share_drifted:.3f} < {DRIFT_THRESHOLD})")

# ── RUN ───────────────────────────────────────────────────────────
print("⏰ Monitor running — checks every 2 minutes")
print("   (first check in 30 seconds to let predictions accumulate)")

schedule.every(2).minutes.do(run_drift_check)

time.sleep(30)
run_drift_check()

while True:
    schedule.run_pending()
    time.sleep(1)