# ml/train_model.py
# Train a small RandomForest classifier on synthetic data and an IsolationForest for comparison.
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "output" / "synthetic_lab_data.csv"
ML_DIR = BASE_DIR / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ----
features = [
    "glucose",
    "hemoglobin",
    "wbc",
    "creatinine",
    "bun",
    "crp",
    "hba1c"
]

target = "anomaly_label"
RANDOM_STATE = 42

# ---- Load data ----
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=features + [target])  # keep only rows with values
X = df[features].astype(float)
y = df[target].astype(int)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ---- Supervised model (RandomForest) ----
clf = RandomForestClassifier(
    n_estimators=200, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(
    clf, "predict_proba") else clf.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else None
cm = confusion_matrix(y_test, y_pred).tolist()

# ---- Unsupervised baseline (IsolationForest) ----
iso = IsolationForest(n_estimators=100, contamination=max(
    0.01, y.mean()), random_state=RANDOM_STATE)
iso.fit(X_train)  # train on train set (unsupervised)
iso_pred_test = iso.predict(X_test)  # -1 anomaly, 1 normal
iso_label_test = (iso_pred_test == -1).astype(int)
iso_report = classification_report(y_test, iso_label_test, output_dict=True)
iso_auc = None
try:
    # produce "score" by flipping the sign of decision_function (higher -> more anomalous)
    iso_scores = -iso.decision_function(X_test)  # higher means more anomalous
    if len(set(y_test)) > 1:
        iso_auc = roc_auc_score(y_test, iso_scores)
except Exception:
    iso_auc = None

# ---- Save models and metadata ----
model_meta = {
    "features": features,
    "median": X_train.median().to_dict(),
    "std": X_train.std().replace(0, 1).to_dict(),  # avoid zero-std
    "clf_info": {"class": "RandomForestClassifier", "n_features": len(features)}
}

joblib.dump(clf, ML_DIR / "model_clf.joblib")
joblib.dump(iso, ML_DIR / "model_iso.joblib")
joblib.dump(model_meta, ML_DIR / "model_meta.joblib")

# ---- Write metrics to human-readable file ----
metrics = {
    "supervised": {"classification_report": report, "roc_auc": auc, "confusion_matrix": cm},
    "isolation_forest": {"classification_report": iso_report, "roc_auc": iso_auc}
}
with open(ML_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(ML_DIR / "metrics.txt", "w") as f:
    f.write("RandomForest classification report:\n")
    f.write(pd.DataFrame(report).to_string())
    f.write("\n\nAUC: {}\n".format(auc))
    f.write("\nIsolationForest classification report:\n")
    f.write(pd.DataFrame(iso_report).to_string())
    f.write("\n\nIsolationForest AUC: {}\n".format(iso_auc))

print("Training complete.")
print(f"Saved supervised model -> {ML_DIR / 'model_clf.joblib'}")
print(f"Saved isolation model  -> {ML_DIR / 'model_iso.joblib'}")
print(f"Saved metadata -> {ML_DIR / 'model_meta.joblib'}")
print(f"Saved metrics -> {ML_DIR / 'metrics.json'} and metrics.txt")
