import re
import string
import numpy as np
from bs4 import BeautifulSoup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
