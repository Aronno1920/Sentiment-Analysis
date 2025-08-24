import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import logging, time, math

from app.utils import clean_text, tokenize_for_w2v, evaluate
from app.config import DEVICE, MAX_FEATURES
###################################


###################################
logger = logging.getLogger("trainer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False

def sentence_vector(text, model):
    words = [w for w in text.split() if w in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)
###################################


# --- TF-IDF ---
def train_tfidf(train_texts, train_labels, test_texts, test_labels, max_features=5000):
    logger.info("Step 1/4 [TF-IDF]: Initializing vectorizer...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    # Fit
    logger.info("Step 2/4 [TF-IDF]: Fitting TF-IDF on training texts...")
    start = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    logger.info("TF-IDF fitted in %.1fs (train=%d, test=%d, features=%d)", 
                time.time()-start, X_train.shape[0], X_test.shape[0], X_train.shape[1])

    # Train classifier
    logger.info("Step 3/4 [TF-IDF]: Training LogisticRegression...")
    start = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    logger.info("Classifier trained in %.1fs", time.time()-start)

    # Evaluate
    logger.info("Step 4/4 [TF-IDF]: Evaluating on test set...")
    preds = clf.predict(X_test)
    metrics = evaluate(test_labels, preds)
    logger.info("TF-IDF evaluation: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])

    return metrics, (clf, vectorizer)

# def train_tfidf(train_texts, train_labels, test_texts, test_labels):
#     tfidf = TfidfVectorizer(preprocessor=clean_text, max_features=MAX_FEATURES, ngram_range=(1,2), min_df=2)
#     X_train = tfidf.fit_transform(train_texts)
#     X_test = tfidf.transform(test_texts)

#     clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
#     clf.fit(X_train, train_labels)
#     preds = clf.predict(X_test)

#     metrics = evaluate(test_labels, preds)
#     return metrics, (clf, tfidf)   # return model + vectorizer
###################################

# --- Word2Vec ---
def train_word2vec(train_texts, train_labels, test_texts, test_labels,
                   vector_size=100, window=5, min_count=2, workers=4):
    logger.info("Step 1/5 [Word2Vec]: Training word embeddings...")
    sentences = [t.split() for t in train_texts]
    start = time.time()
    w2v = Word2Vec(sentences, vector_size=vector_size, window=window, 
                   min_count=min_count, workers=workers)
    logger.info("Word2Vec trained in %.1fs (vocab=%d, dim=%d)", 
                time.time()-start, len(w2v.wv), vector_size)

    # Vectorize training set
    logger.info("Step 2/5 [Word2Vec]: Encoding training set...")
    start = time.time()
    X_train = np.array([sentence_vector(t, w2v) for t in train_texts])
    logger.info("Encoded %d training samples in %.1fs", len(X_train), time.time()-start)

    # Vectorize test set
    logger.info("Step 3/5 [Word2Vec]: Encoding test set...")
    start = time.time()
    X_test = np.array([sentence_vector(t, w2v) for t in test_texts])
    logger.info("Encoded %d test samples in %.1fs", len(X_test), time.time()-start)

    # Train classifier
    logger.info("Step 4/5 [Word2Vec]: Training LogisticRegression...")
    start = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    logger.info("Classifier trained in %.1fs", time.time()-start)

    # Evaluate
    logger.info("Step 5/5 [Word2Vec]: Evaluating on test set...")
    preds = clf.predict(X_test)
    metrics = evaluate(test_labels, preds)
    logger.info("Word2Vec evaluation: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])

    return metrics, (clf, w2v)

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

#     metrics = evaluate(test_labels, preds)
#     return metrics, (clf, w2v)
###################################


# --- BERT ---
def train_bert(train_texts, train_labels, test_texts, test_labels,
               limit_train=5000, limit_test=2000, batch_size=32):
    logger.info("Step 1/5: Loading DistilBERT tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    bert_model.eval()
    total_params = sum(p.numel() for p in bert_model.parameters()) / 1e6
    logger.info(f"Loaded BERT on device={DEVICE}; params≈{total_params:.1f}M")

    # Limits for speed
    logger.info("Step 2/5: Preparing datasets with limits for speed...")
    tr_texts = train_texts[:limit_train]
    tr_labels = train_labels[:limit_train]
    te_texts = test_texts[:limit_test]
    te_labels = test_labels[:limit_test]
    logger.info(f"Using {len(tr_texts)} train / {len(te_texts)} test samples.")

    def bert_encode(texts, phase="train"):
        logger.info(f"Step 3/5 ({phase}): Encoding {len(texts)} texts with batch_size={batch_size}...")
        start = time.time()
        all_embeddings = []
        total = len(texts)
        if total == 0:
            logger.warning(f"No texts provided for phase '{phase}'.")
            return np.zeros((0, bert_model.config.hidden_size))

        for i in range(0, total, batch_size):
            batch = [clean_text(t) for t in texts[i:i+batch_size]]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = bert_model(**enc).last_hidden_state  # [B, L, H]
                mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
                pooled = (out * mask).sum(1) / mask.sum(1)  # mean pooling
                all_embeddings.append(pooled.detach().cpu().numpy())

            # progress every 10 batches or on last batch
            b_idx = i // batch_size + 1
            b_total = math.ceil(total / batch_size)
            if b_idx % 10 == 0 or i + batch_size >= total:
                done = min(i + batch_size, total)
                pct = 100.0 * done / total
                logger.info(f"  [{phase}] batch {b_idx}/{b_total} | encoded {done}/{total} ({pct:.1f}%)")

        elapsed = time.time() - start
        logger.info(f"Finished {phase} encoding in {elapsed:.1f}s")
        return np.vstack(all_embeddings)

    # Encode
    X_train = bert_encode(tr_texts, phase="train")
    X_test  = bert_encode(te_texts, phase="test")
    y_train = tr_labels
    y_test  = te_labels

    # Fit LR
    logger.info("Step 4/5: Fitting LogisticRegression on BERT embeddings...")
    start = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    fit_time = time.time() - start
    logger.info(f"Classifier trained in {fit_time:.1f}s")

    # Evaluate
    logger.info("Step 5/5: Evaluating on test set...")
    start = time.time()
    preds = clf.predict(X_test)
    metrics = evaluate(y_test, preds)
    eval_time = time.time() - start
    logger.info(
        "Evaluation done in %.1fs | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        eval_time, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]
    )

    return metrics, (clf, tokenizer, bert_model)



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

#     # Limit dataset for speed
#     X_train = bert_encode(train_texts[:5000])
#     y_train = train_labels[:5000]
#     X_test = bert_encode(test_texts[:2000])
#     y_test = test_labels[:2000]

#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)

#     metrics = evaluate(y_test, preds)
#     return metrics, (clf, tokenizer, model)   # return classifier + tokenizer + encoder
###################################