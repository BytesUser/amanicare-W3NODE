import uuid
from datetime import datetime

def run_prediction(payload):
    """
    Demo anomaly detection: only using hemoglobin and glucose for simplicity.
    """
    result = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "anomaly": False,
        "score": 0.0,
        "iso_anomaly": False,
        "iso_score": 0.0
    }

    # Example thresholds for demo purposes
    if payload["hemoglobin"] < 10 or payload["glucose"] > 125:
        result["anomaly"] = True
        result["score"] = 1.0
    return result
