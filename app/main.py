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

from app.utils import get_prediction_with_proba
################################################


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="API on sentiment analysis using the IMDB movie reviews dataset. The dataset contains 50,000 reviews equally divided into positive and negative sentiments. Your task is to build models that can classify whether a given review expresses a positive or negative opinion."
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


###### Show Sentiment Analysis with single model
@app.post("/predict_single/{model_name}")
def predict_single(model_name: str, req: ReviewRequest):
    result = get_prediction_with_proba(model_name, req.text, MODELS)
    return {"text": req.text, "result": result}
################################################


###### Show Sentiment Analysis
@app.post("/predict")
def predict(req: ReviewRequest):
    text = req.text
    results = []

    for model_name in ["tfidf", "word2vec", "bert"]:
        results.append(get_prediction_with_proba(model_name, text, MODELS))

    return {"text": text, "results": results}
################################################