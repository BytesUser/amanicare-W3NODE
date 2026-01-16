# generate_synthetic_lab_data.py
# Synthetic lab dataset generator for "Abnormal Lab Flags" hackathon MVP

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------
# Config
# -------------------------------
np.random.seed(42)  # reproducibility
N = 200  # total records
clinics = [1, 2, 3]  # simulate 3 clinic nodes
anomaly_fraction = 0.1  # 10% of data will have anomalies

# -------------------------------
# Generate base synthetic dataset
# -------------------------------
data = pd.DataFrame({
    "glucose": np.random.normal(95, 10, N),
    "hemoglobin": np.random.normal(14, 1.2, N),
    "wbc": np.random.normal(7, 2, N),
    "creatinine": np.random.normal(1, 0.2, N),
    "bun": np.random.normal(14, 4, N),
    "crp": np.random.normal(3, 2, N),
    "hba1c": np.random.normal(5.5, 0.5, N),
    "timestamp": [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(N)],
    "clinic_id": np.random.choice(clinics, N)
})

# -------------------------------
# Inject anomalies (~10% of rows)
# -------------------------------
num_anomalies = int(N * anomaly_fraction)
for col, multiplier in [("glucose", 1.5), ("wbc", 2.0), ("creatinine", 2.0)]:
    idx = np.random.choice(N, num_anomalies, replace=False)
    data.loc[idx, col] *= multiplier

# -------------------------------
# Label anomalies
# -------------------------------
data["anomaly_label"] = 0
data.loc[
    (data["glucose"] > 140) |
    (data["wbc"] > 14) |
    (data["creatinine"] > 1.8),
    "anomaly_label"
] = 1

# -------------------------------
# from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# Save full dataset
# -------------------------------
data.to_csv(OUTPUT_DIR / "synthetic_lab_data.csv", index=False)

# -------------------------------
# Save separate datasets for each clinic node
# -------------------------------
for clinic in clinics:
    node_data = data[data["clinic_id"] == clinic]
    node_data.to_csv(
        OUTPUT_DIR / f"synthetic_lab_data_node{clinic}.csv",
        index=False
    )

# -------------------------------
# Print summary
# -------------------------------
print("Synthetic lab dataset generated!")
print(f"Total records: {len(data)}")
print(f"Total anomalies: {data['anomaly_label'].sum()}")
for clinic in clinics:
    count = len(data[data["clinic_id"] == clinic])
    print(f"Clinic {clinic} records: {count}")
print(f"CSV files saved in 'output/' folder.")
