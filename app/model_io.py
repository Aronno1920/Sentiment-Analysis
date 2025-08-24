import os
import joblib
import torch
import json
from pathlib import Path


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = "models"
METRICS_FILE = BASE_DIR / MODEL_DIR / "metrics.json"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


###### Model Related
def save_model(name, obj):
    ensure_model_dir()
    path = os.path.join(MODEL_DIR, f"{name}.pkl")

    if isinstance(obj, tuple) and len(obj) == 3:  
        # Special case for BERT (clf, tokenizer, model)
        clf, tokenizer, model = obj
        joblib.dump((clf, tokenizer), path)  
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}_bert.pt"))
    else:
        joblib.dump(obj, path)

def load_model(name, bert_class=None, bert_model=None):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None

    if name == "bert":
        clf, tokenizer = joblib.load(path)
        if bert_model:
            bert_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{name}_bert.pt")))
        return clf, tokenizer, bert_model
    else:
        return joblib.load(path)
###################################


###### Model Evaluation Metrics Related
def save_metrics(model_name: str, metrics: dict):
    """Save metrics for a specific model into metrics.json (append/update)."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # update only this model's metrics
    all_metrics[model_name] = metrics

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=4)


def load_metrics() -> dict:
    """Load all metrics from metrics.json if exists."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}
###################################