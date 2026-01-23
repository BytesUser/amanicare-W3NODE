# Amanicare — Federated Blood Test Anomaly Demo

## Overview
A privacy-preserving clinical lab anomaly detection demo.  
Use it to upload blood test CSVs and view anomaly flags.

## How to Run

1. Clone:
git clone https://github.com/BytesUser/amanicare-W3NODE

cd amanicare-W3NODE
git checkout demo-ready

2. Create env & install:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


3. Start server:


uvicorn app:app --reload


4. Open:


http://127.0.0.1:8000/


5. Use `sample_data.csv` to test.

## CSV Format


patient_id,test_date,hemoglobin,glucose,calcium,potassium
...


## Note
Patient data never leaves the machine; this is a *demo* — not a medical device.