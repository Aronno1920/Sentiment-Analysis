import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import load_imdb, clean_text, tokenize_for_w2v
from transformers import AutoModel
import torch
################################################
from app.model_train import train_tfidf, train_word2vec, train_bert
from app.model_io import save_model, load_model
from app.model_io import save_metrics, load_metrics
################################################


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="API for Analysising sentiment using IMDB dataset"
    )

# Global model storage
MODELS = {}

class ReviewRequest(BaseModel):
    text: str

###### Load models if already saved on disk
@app.on_event("startup")
def load_models_on_startup():
    """Load models if already saved on disk"""
    global MODELS
    tfidf = load_model("tfidf")
    if tfidf: MODELS["tfidf"] = tfidf

    word2vec = load_model("word2vec")
    if word2vec: MODELS["word2vec"] = word2vec

    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    bert = load_model("bert", bert_model=bert_model)
    if bert: MODELS["bert"] = bert
################################################


###### API health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Welcome! Sentiment Analysis API is operational."}
################################################


###### Model Train - TF-IDF
@app.get("/train_tfidf")
def train_model_tfidf():
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    results = []

    metrics, obj = train_tfidf(train_texts, train_labels, test_texts, test_labels)
    MODELS["tfidf"] = obj
    save_model("tfidf", obj)
    save_metrics("tfidf", metrics)   # <-- save metrics into metrics.json
    results.append({"Model": "TF-IDF + LR", **metrics})

    return {"results": results}
################################################


###### Model Train - Word2Vec
@app.get("/train_word2vec")
def train_model_word2vec():
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    results = []

    metrics, obj = train_word2vec(train_texts, train_labels, test_texts, test_labels)
    MODELS["word2vec"] = obj
    save_model("word2vec", obj)
    save_metrics("word2vec", metrics)  # <-- save metrics into metrics.json
    results.append({"Model": "Word2Vec + LR", **metrics})

    return {"results": results}
################################################


###### Model Train - BERT
@app.get("/train_bert")
def train_model_bert():
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    results = []

    metrics, obj = train_bert(train_texts, train_labels, test_texts, test_labels)
    MODELS["bert"] = obj
    save_model("bert", obj)
    save_metrics("bert", metrics)   # <-- save metrics into metrics.json
    results.append({"Model": "BERT + LR", **metrics})

    return {"results": results}
################################################


###### Show Matrics
@app.get("/metrics")
def get_metrics():
    """Return evaluation metrics for all trained models."""
    metrics = load_metrics()
    if not metrics:
        return {"message": "No metrics found. Please train a model first."}
    return {"metrics": metrics}
################################################



###### Model Train - BERT
@app.post("/predict_single/{model_name}")
def predict_single(model_name: str, req: ReviewRequest):
    text = req.text

    if model_name not in MODELS:
        return {"error": f"Model '{model_name}' not loaded. Train or restart app."}

    if model_name == "tfidf":
        clf, tfidf = MODELS["tfidf"]
        X = tfidf.transform([text])
        pred = clf.predict(X)[0]

    elif model_name == "word2vec":
        clf, w2v = MODELS["word2vec"]
        tokens = tokenize_for_w2v(text)
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        X = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, w2v.vector_size))
        pred = clf.predict(X)[0]

    elif model_name == "bert":
        clf, tokenizer, bert_model = MODELS["bert"]
        bert_model.eval()
        enc = tokenizer([clean_text(text)], truncation=True, padding=True, max_length=512, return_tensors="pt").to(bert_model.device)
        with torch.no_grad():
            out = bert_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            X = pooled.cpu().numpy()
        pred = clf.predict(X)[0]

    return {"text": text, "prediction": "positive" if pred == 1 else "negative"}
################################################



###### Model Train - BERT
@app.post("/predict")
def predict(req: ReviewRequest):
    text = req.text
    results = []

    for model_name in ["tfidf", "word2vec", "bert"]:
        if model_name not in MODELS:
            results.append({
                "model": model_name,
                "error": f"Model '{model_name}' not loaded. Train or restart app."
            })
            continue

        if model_name == "tfidf":
            clf, tfidf = MODELS["tfidf"]
            X = tfidf.transform([text])
            pred = clf.predict(X)[0]

        elif model_name == "word2vec":
            clf, w2v = MODELS["word2vec"]
            tokens = tokenize_for_w2v(text)
            vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
            X = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, w2v.vector_size))
            pred = clf.predict(X)[0]

        elif model_name == "bert":
            clf, tokenizer, bert_model = MODELS["bert"]
            bert_model.eval()
            enc = tokenizer([clean_text(text)], truncation=True, padding=True, max_length=512, return_tensors="pt").to(bert_model.device)
            with torch.no_grad():
                out = bert_model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
                pooled = (out * mask).sum(1) / mask.sum(1)
                X = pooled.cpu().numpy()
            pred = clf.predict(X)[0]

        results.append({
            "model": model_name,
            "prediction": "positive" if pred == 1 else "negative"
        })

    return {"text": text, "results": results}

################################################