import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

from app.utils import clean_text, tokenize_for_w2v, evaluate
from app.config import DEVICE, MAX_FEATURES

# --- TF-IDF ---
def train_tfidf(train_texts, train_labels, test_texts, test_labels):
    tfidf = TfidfVectorizer(preprocessor=clean_text, max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    metrics = evaluate(test_labels, preds)
    return metrics, (clf, tfidf)   # return model + vectorizer
###################################
# def train_tfidf(train_texts, train_labels, test_texts, test_labels):
#     tfidf = TfidfVectorizer(preprocessor=clean_text, max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2)
#     X_train = tfidf.fit_transform(train_texts)
#     X_test = tfidf.transform(test_texts)

#     clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
#     clf.fit(X_train, train_labels)
#     preds = clf.predict(X_test)
#     return evaluate(test_labels, preds)
###################################

# --- Word2Vec ---
def train_word2vec(train_texts, train_labels, test_texts, test_labels):
    train_tokens = [tokenize_for_w2v(t) for t in train_texts]
    test_tokens = [tokenize_for_w2v(t) for t in test_texts]

    w2v = Word2Vec(sentences=train_tokens, vector_size=200, window=5, min_count=2, workers=4, sg=1, epochs=5)

    def sentvec_avg(tokens):
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)

    X_train = np.vstack([sentvec_avg(t) for t in train_tokens])
    X_test = np.vstack([sentvec_avg(t) for t in test_tokens])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    metrics = evaluate(test_labels, preds)
    return metrics, (clf, w2v)
###################################
# def train_word2vec(train_texts, train_labels, test_texts, test_labels):
#     train_tokens = [tokenize_for_w2v(t) for t in train_texts]
#     test_tokens = [tokenize_for_w2v(t) for t in test_texts]

#     w2v = Word2Vec(sentences=train_tokens, vector_size=200, window=5, min_count=2, workers=4, sg=1, epochs=5)

#     def sentvec_avg(tokens):
#         vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
#         return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)

#     X_train = np.vstack([sentvec_avg(t) for t in train_tokens])
#     X_test = np.vstack([sentvec_avg(t) for t in test_tokens])

#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, train_labels)
#     preds = clf.predict(X_test)
#     return evaluate(test_labels, preds)
###################################

# --- BERT ---
def train_bert(train_texts, train_labels, test_texts, test_labels):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    model.eval()

    def bert_encode(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[i:i+batch_size]]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
                pooled = (out * mask).sum(1) / mask.sum(1)
                all_embeddings.append(pooled.cpu().numpy())
        return np.vstack(all_embeddings)

    # Limit dataset for speed
    X_train = bert_encode(train_texts[:5000])
    y_train = train_labels[:5000]
    X_test = bert_encode(test_texts[:2000])
    y_test = test_labels[:2000]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    metrics = evaluate(y_test, preds)
    return metrics, (clf, tokenizer, model)   # return classifier + tokenizer + encoder

###################################
# def train_bert(train_texts, train_labels, test_texts, test_labels):
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
#     model.eval()

#     def bert_encode(texts, batch_size=32):
#         all_embeddings = []
#         for i in range(0, len(texts), batch_size):
#             batch = [clean_text(t) for t in texts[i:i+batch_size]]
#             enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
#             with torch.no_grad():
#                 out = model(**enc).last_hidden_state
#                 mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
#                 pooled = (out * mask).sum(1) / mask.sum(1)
#                 all_embeddings.append(pooled.cpu().numpy())
#         return np.vstack(all_embeddings)

#     # Limit dataset size for speed
#     X_train = bert_encode(train_texts[:5000])
#     y_train = train_labels[:5000]
#     X_test = bert_encode(test_texts[:2000])
#     y_test = test_labels[:2000]

#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)
#     return evaluate(y_test, preds)
###################################