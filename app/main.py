import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import load_imdb, clean_text, tokenize_for_w2v
from app.model_train import train_tfidf, train_word2vec, train_bert
from app.model_io import save_model, load_model
from transformers import AutoModel

import torch

app = FastAPI(title="Sentiment Analysis API")

# Global model storage
MODELS = {}

class ReviewRequest(BaseModel):
    text: str

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

@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API"}

@app.get("/train")
def train_models():
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    results = []

    # TF-IDF
    metrics, obj = train_tfidf(train_texts, train_labels, test_texts, test_labels)
    MODELS["tfidf"] = obj
    save_model("tfidf", obj)
    results.append({"Model": "TF-IDF + LR", **metrics})

    # Word2Vec
    metrics, obj = train_word2vec(train_texts, train_labels, test_texts, test_labels)
    MODELS["word2vec"] = obj
    save_model("word2vec", obj)
    results.append({"Model": "Word2Vec + LR", **metrics})

    # BERT
    metrics, obj = train_bert(train_texts, train_labels, test_texts, test_labels)
    MODELS["bert"] = obj
    save_model("bert", obj)
    results.append({"Model": "BERT + LR", **metrics})

    return {"results": results}


@app.post("/predict/{model_name}")
def predict(model_name: str, req: ReviewRequest):
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
