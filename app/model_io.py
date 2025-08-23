import os
import joblib
import torch

MODEL_DIR = "model"

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(name, obj):
    """Save sklearn / gensim models with joblib, torch separately"""
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
    """Load model back. For BERT, pass bert_model=AutoModel.from_pretrained(...)"""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None

    if name == "bert":
        clf, tokenizer = joblib.load(path)
        if bert_model:  # load weights back
            bert_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{name}_bert.pt")))
        return clf, tokenizer, bert_model
    else:
        return joblib.load(path)
