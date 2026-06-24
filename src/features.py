import redis
import numpy as np

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_account_features(account_id: int, current_amount: float) -> dict:
    """
    Get rolling features for an account from Redis.
    This is called by the consumer for every incoming transaction.
    """
    key = f"account:{account_id}:history"

    # Get last 10 transaction amounts for this account
    history_raw = r.lrange(key, 0, 9)
    history = [float(x) for x in history_raw] if history_raw else [current_amount]

    avg = np.mean(history)

    return {
        "rolling_avg_amount":  round(avg, 4),
        "rolling_max_amount":  round(float(np.max(history)), 4),
        "rolling_std_amount":  round(float(np.std(history)) if len(history) > 1 else 0.0, 4),
        "txn_count":           len(history),
        "amount_vs_avg_ratio": round(current_amount / (avg + 1e-6), 4),
    }

def update_account_history(account_id: int, amount: float):
    """
    Push new transaction amount into Redis for this account.
    Keeps only the last 10 transactions.
    """
    key = f"account:{account_id}:history"
    r.lpush(key, amount)   # push to front
    r.ltrim(key, 0, 9)     # keep only last 10
    r.expire(key, 86400)   # expire after 24 hours