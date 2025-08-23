import os
import joblib
import torch
import json

MODEL_DIR = "model"

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
def save_metrics(name, metrics: dict):
    """Save evaluation metrics into metrics.json in models dir"""
    ensure_model_dir()
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics[name] = metrics
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

def load_metrics():
    """Load all stored metrics from disk"""
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r") as f:
        return json.load(f)
###################################