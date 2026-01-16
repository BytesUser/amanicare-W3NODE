import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            clinic_id TEXT,
            anomaly INTEGER,
            score REAL,
            iso_anomaly INTEGER,
            iso_score REAL,
            payload TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(data: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)
    """, (
        data["id"],
        data["ts"],
        data.get("clinic_id", "unknown"),
        int(data["anomaly"]),
        float(data["score"]),
        int(data["iso_anomaly"]),
        float(data["iso_score"]),
        data["raw_payload"]
    ))
    conn.commit()
    conn.close()
