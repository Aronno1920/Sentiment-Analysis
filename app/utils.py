import re
import string
import numpy as np
from bs4 import BeautifulSoup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
################################################


# Remove punctuation table
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_for_w2v(text: str):
    return clean_text(text).split()

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def load_imdb():
    """Load IMDB dataset and return train/test splits."""
    imdb = load_dataset("imdb")
    train_texts = list(imdb["train"]["text"])
    train_labels = list(imdb["train"]["label"])
    test_texts = list(imdb["test"]["text"])
    test_labels = list(imdb["test"]["label"])
    return train_texts, train_labels, test_texts, test_labels


################################################
def get_prediction_with_proba(model_name: str, text: str, MODELS: dict):
    if model_name not in MODELS:
        return {
            "model": model_name,
            "error": f"Model '{model_name}' not loaded. Train or restart app."
        }

    if model_name == "tfidf":
        clf, tfidf = MODELS["tfidf"]
        X = tfidf.transform([text])
        pred = clf.predict(X)[0]
        proba = max(clf.predict_proba(X)[0])

    elif model_name == "word2vec":
        clf, w2v = MODELS["word2vec"]
        tokens = tokenize_for_w2v(text)
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        X = (
            np.mean(vecs, axis=0).reshape(1, -1)
            if vecs else np.zeros((1, w2v.vector_size))
        )
        pred = clf.predict(X)[0]
        proba = max(clf.predict_proba(X)[0])

    elif model_name == "bert":
        clf, tokenizer, bert_model = MODELS["bert"]
        bert_model.eval()
        enc = tokenizer(
            [clean_text(text)],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(bert_model.device)

        with torch.no_grad():
            out = bert_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            X = pooled.cpu().numpy()

        pred = clf.predict(X)[0]
        proba = max(clf.predict_proba(X)[0])

    else:
        return {"model": model_name, "error": "Unsupported model."}

    return {
        "model": model_name,
        "prediction": "positive" if pred == 1 else "negative",
        "probability": f"{proba * 100:.2f}%"
    }
################################################
