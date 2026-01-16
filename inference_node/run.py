# inference_node/run.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import Query
import joblib
import pandas as pd
import numpy as np
import sqlite3
import json
import time
import uuid
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
DB = BASE_DIR / "backend" / "results.db"
os.makedirs(BASE_DIR / "backend", exist_ok=True)

# Load models & meta
clf = joblib.load(ML_DIR / "model_clf.joblib")
iso = joblib.load(ML_DIR / "model_iso.joblib")
meta = joblib.load(ML_DIR / "model_meta.joblib")
features = meta["features"]
median = meta["median"]
std = meta["std"]

# SQLite
# --- SQLite: ensure single DB and correct schema ---
DB = BASE_DIR / "backend" / "results.db"
os.makedirs(BASE_DIR / "backend", exist_ok=True)

conn = sqlite3.connect(str(DB), check_same_thread=False)
# Create a table that matches the INSERT used below
conn.execute("""
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    ts TEXT,
    clinic_id TEXT,
    anomaly INTEGER,
    score REAL,
    iso_anomaly INTEGER,
    iso_score REAL,
    payload TEXT
)
""")
conn.commit()


app = FastAPI()


# allow any origin for hackathon/demo. Replace "*" with specific origins for production.
app.add_middleware(
    CORSMiddleware,
    # allow your frontend origin(s) here, e.g. ["http://127.0.0.1:5500"]
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


class LabRow(BaseModel):
    glucose: float = 100.0
    hemoglobin: float = 14.0
    wbc: float = 7.0
    creatinine: float = 1.0
    bun: float = 14.0
    crp: float = 3.0
    hba1c: float = 5.5
    # optional clinic id for audit / federated simulation
    clinic_id: str = "unknown"


def compute_z_scores(row):
    z = {}
    for f in features:
        val = float(row.get(f, 0.0))
        med = float(median.get(f, 0.0))
        s = float(std.get(f, 1.0)) or 1.0
        z[f] = (val - med) / s
    return z


@app.post("/predict")
def predict(row: LabRow):
    payload = {f: float(getattr(row, f)) for f in features}
    X = pd.DataFrame([payload])

    # -------------------------------
    # Supervised model
    # -------------------------------
    proba = float(clf.predict_proba(X)[:, 1][0]) if hasattr(
        clf, "predict_proba") else float(clf.predict(X)[0])
    label = int(proba > 0.5)

    # -------------------------------
    # Isolation Forest
    # -------------------------------
    iso_score = float(-iso.decision_function(X)[0])  # higher = more anomalous
    iso_label = int(iso.predict(X)[0] == -1)

    # -------------------------------
    # Explanation
    # -------------------------------
    z = compute_z_scores(payload)
    top3 = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:3]
    explanation = [
        {"feature": k, "value": payload[k], "z": round(v, 3)}
        for k, v in top3
    ]

    # -------------------------------
    # Metadata
    # -------------------------------
    rid = str(uuid.uuid4())
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clinic_id = getattr(row, "clinic_id", "unknown")

    # -------------------------------
    # Persist prediction (THIS IS THE KEY PART)
    # -------------------------------
    conn.execute(
        """
        INSERT INTO results (
            id, ts, clinic_id,
            anomaly, score,
            iso_anomaly, iso_score,
            payload
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            rid,
            ts,
            clinic_id,
            int(label),
            proba,
            int(iso_label),
            iso_score,
            json.dumps(payload),
        )
    )
    conn.commit()

    # -------------------------------
    # API response
    # -------------------------------
    return {
        "id": rid,
        "ts": ts,
        "clinic_id": clinic_id,
        "anomaly": bool(label),
        "score": proba,
        "iso_anomaly": bool(iso_label),
        "iso_score": iso_score,
        "explanation": explanation,
    }


# --- Add endpoint to fetch stored predictions ---
@app.get("/results")
def get_results(clinic_id: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=200)):
    """
    Return last `limit` predictions, optionally filtered by clinic_id.
    """
    # Build SQL
    sql = "SELECT id, ts, clinic_id, anomaly, score, iso_anomaly, iso_score, payload FROM results"
    params = []
    if clinic_id:
        sql += " WHERE clinic_id = ?"
        params.append(clinic_id)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    cur = conn.cursor()
    rows = cur.execute(sql, params).fetchall()

    results = []
    for id_, ts, c_id, anomaly, score, iso_anom, iso_score, payload in rows:
        try:
            payload_json = json.loads(payload)
        except Exception:
            payload_json = payload
        results.append({
            "id": id_,
            "ts": ts,
            "clinic_id": c_id,
            "anomaly": bool(anomaly),
            "score": score,
            "iso_anomaly": bool(iso_anom),
            "iso_score": iso_score,
            "payload": payload_json
        })

    return {"count": len(results), "results": results}


@app.get("/results/summary")
def results_summary(hours: int = 24):
    # small debug log so we see the call in server output
    print(f"[DEBUG] results_summary called, hours={hours}", flush=True)

    # Calculate cutoff timestamp (ISO UTC)
    since = time.time() - hours * 3600
    since_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since))

    rows = conn.execute("""
        SELECT clinic_id,
               COUNT(*) as total,
               SUM(anomaly) as abnormal
        FROM results
        WHERE ts >= ?
        GROUP BY clinic_id
        ORDER BY abnormal DESC
    """, (since_ts,)).fetchall()

    clinics = []
    for clinic_id, total, abnormal in rows:
        clinics.append({
            "clinic_id": clinic_id,
            "total": total,
            "abnormal": abnormal,
            "abnormal_rate": round((abnormal / total) if total else 0.0, 3)
        })

    return {
        "window_hours": hours,
        "clinics": clinics
    }
