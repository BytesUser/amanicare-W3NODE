from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uuid
import io

app = FastAPI(title="Amanicare Demo")

# Serve static folder (for CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# Dummy anomaly detector: flags glucose > 120 or hemoglobin < 11
def detect_anomaly(df):
    df['anomaly'] = ((df['glucose'] > 120) | (df['hemoglobin'] < 11))
    df['flagged_tests'] = df.apply(lambda row: "Glucose" if row['glucose']>120 else ("Hemoglobin" if row['hemoglobin']<11 else "None"), axis=1)
    df['recommendation'] = df.apply(lambda row: "Review patient" if row['anomaly'] else "Normal", axis=1)
    return df

@app.post("/upload_csv")
async def upload_csv(file: UploadFile):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Only keep required columns
        required_cols = ['patient_id','test_date','hemoglobin','glucose','calcium','potassium']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return JSONResponse({"error": f"Missing columns: {missing}"}, status_code=400)
        df = df[required_cols]

        # Run dummy anomaly detection
        df = detect_anomaly(df)

        # Convert DataFrame to list of dicts for JSON
        result = df.to_dict(orient='records')
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
