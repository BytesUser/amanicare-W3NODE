# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Amanicare Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabInput(BaseModel):
    # keep fields flexible: optional fields allowed by not required
    glucose: float = 100.0
    hemoglobin: float = 14.0
    wbc: float = 7.0
    platelets: float = 250
    creatinine: float = 1.0
    bun: float = 14.0
    sodium: float = 140.0
    potassium: float = 4.2
    crp: float = 3.0
    hba1c: float = 5.5


@app.get("/")
def health():
    return {"status": "backend alive"}


@app.post("/analyze")
def analyze(payload: LabInput):
    # lightweight "rule" check for now — replace with model call later
    p = payload.dict()
    anomaly = False
    reason = []
    if p["creatinine"] > 1.8 or p["bun"] > 25:
        anomaly = True
        reason.append("renal pattern")
    if p["wbc"] > 12 and p["crp"] > 20:
        anomaly = True
        reason.append("infection pattern")
    if p["glucose"] > 200 or p["hba1c"] > 8:
        anomaly = True
        reason.append("glycemic")
    return {"received": p, "anomaly": anomaly, "reasons": reason}
