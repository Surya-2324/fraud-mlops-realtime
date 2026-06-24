import redis
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Fraud MLOps Dashboard", version="1.0")
r   = redis.Redis(host="localhost", port=6379, decode_responses=True)

def rget(key, default="N/A"):
    val = r.get(key)
    return val if val is not None else default

@app.get("/metrics")
def metrics():
    return {
        "total_scored":    rget("stats:total_scored",  "0"),
        "fraud_rate":      rget("stats:fraud_rate",    "0.000"),
        "drift_score":     rget("stats:drift_score",   "0.000"),
        "drifted_columns": rget("stats:drift_cols",    "0"),
        "model_version":   rget("stats:model_version", "1"),
        "is_retraining":   rget("stats:retraining",    "0") == "1",
        "last_retrain":    rget("stats:last_retrain",  "never"),
        "server_time":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Fraud MLOps — Live Dashboard</title>
  <style>
    body { font-family: monospace; background: #08090d;
           color: #dde3f0; padding: 40px; margin: 0; }
    h1   { color: #5b8df7; margin-bottom: 8px; }
    .sub { color: #4a5570; font-size: 13px; margin-bottom: 32px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }
    .card { background: #0f111a; border: 1px solid #1c2035;
            border-radius: 10px; padding: 24px; }
    .val  { font-size: 32px; font-weight: 700; color: #5b8df7; margin-bottom: 6px; }
    .lbl  { font-size: 11px; color: #4a5570; letter-spacing: 0.1em; text-transform: uppercase; }
    .fraud { color: #f06a6a; }
    .safe  { color: #2ecc7a; }
    .warn  { color: #f5c542; }
    .log  { background: #0f111a; border: 1px solid #1c2035;
            border-radius: 10px; padding: 20px; height: 200px;
            overflow-y: auto; font-size: 12px; color: #4a5570; }
    .log div { padding: 3px 0; border-bottom: 1px solid #1c2035; color: #7a86a0; }
  </style>
</head>
<body>
  <h1>🔴 Fraud Detection MLOps — Live</h1>
  <p class="sub">Auto-refreshes every 3 seconds</p>

  <div class="grid" id="grid">Loading...</div>
  <div class="log"  id="log"></div>

  <script>
    const history = [];

    async function refresh() {
      const d = await fetch('/metrics').then(r => r.json());

      const fraudPct = (parseFloat(d.fraud_rate) * 100).toFixed(2);
      const drift    = parseFloat(d.drift_score);
      const status   = d.is_retraining ? '⟳ RETRAINING' : '✅ Stable';
      const driftCls = drift > 0.3 ? 'fraud' : drift > 0.1 ? 'warn' : 'safe';

      document.getElementById('grid').innerHTML = `
        <div class="card">
          <div class="val">${parseInt(d.total_scored).toLocaleString()}</div>
          <div class="lbl">Total Scored</div>
        </div>
        <div class="card">
          <div class="val fraud">${fraudPct}%</div>
          <div class="lbl">Fraud Rate</div>
        </div>
        <div class="card">
          <div class="val ${driftCls}">${drift}</div>
          <div class="lbl">Drift Score (PSI)</div>
        </div>
        <div class="card">
          <div class="val warn">${d.drifted_columns}</div>
          <div class="lbl">Drifted Columns</div>
        </div>
        <div class="card">
          <div class="val">v${d.model_version}</div>
          <div class="lbl">Model Version</div>
        </div>
        <div class="card">
          <div class="val ${d.is_retraining ? 'warn' : 'safe'}">${status}</div>
          <div class="lbl">System Status</div>
        </div>
      `;

      history.unshift(`[${d.server_time}] scored=${d.total_scored} fraud=${fraudPct}% drift=${drift} model=v${d.model_version}`);
      if (history.length > 50) history.pop();
      document.getElementById('log').innerHTML = history.map(h => `<div>${h}</div>`).join('');
    }

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""