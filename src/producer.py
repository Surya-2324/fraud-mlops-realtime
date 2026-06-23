import json
import random
import time
import os
from confluent_kafka import Producer

# ── CONFIG ──────────────────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC        = "transactions"

# Set to True later to SIMULATE DRIFT (the real world changing)
DRIFT_MODE   = False

# ── KAFKA PRODUCER ───────────────────────────────────────────────
# confluent-kafka Producer — faster than kafka-python
producer = Producer({
    "bootstrap.servers": KAFKA_BROKER,
    "acks": "all",            # wait for broker to confirm receipt
    "retries": 3,
})

def delivery_report(err, msg):
    """Called by Kafka after each message is delivered (or fails)."""
    if err:
        print(f"❌ Delivery failed: {err}")
    # else: silent success (uncomment below to see every ack)
    # else: print(f"✓ offset {msg.offset()}")

def make_transaction() -> dict:
    """
    Generate a synthetic financial transaction.
    
    In NORMAL mode: realistic distribution (small amounts, varied merchants)
    In DRIFT mode:  shifted distribution (large ATM amounts at midnight)
                    → this simulates a new fraud pattern the model hasn't seen
    """
    base = {
        "transaction_id": random.randint(100_000, 999_999),
        "account_id":     random.randint(1, 1000),
        "timestamp":      time.time(),
    }

    if DRIFT_MODE:
        # Simulating drift: late-night high-value ATM withdrawals spike
        base.update({
            "amount":       round(random.uniform(3000, 9999), 2),
            "merchant_cat": "atm",
            "hour_of_day":  random.randint(1, 4),
            "is_weekend":   1,
            "account_age":  random.randint(10, 90),  # new accounts
        })
    else:
        base.update({
            "amount":       round(random.uniform(5, 500), 2),
            "merchant_cat": random.choice(["grocery","online",
                                             "restaurant","atm","travel"]),
            "hour_of_day":  random.randint(8, 22),
            "is_weekend":   random.choice([0, 1]),
            "account_age":  random.randint(30, 3000),
        })

    return base

# ── MAIN LOOP ────────────────────────────────────────────────────
print(f"🚀 Producer started → topic: {TOPIC} | drift_mode: {DRIFT_MODE}")

while True:
    txn = make_transaction()

    # Serialize to JSON bytes and send to Kafka
    producer.produce(
        topic    = TOPIC,
        key      = str(txn["account_id"]).encode(),  # key = account (for partitioning)
        value    = json.dumps(txn).encode("utf-8"),
        callback = delivery_report,
    )

    # poll() triggers delivery_report callbacks
    producer.poll(0)

    print(f"→ Sent txn {txn['transaction_id']} | £{txn['amount']:>8.2f} | {txn['merchant_cat']}")
    time.sleep(0.5)  # 2 transactions per second