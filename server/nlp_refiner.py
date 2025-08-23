# nlp_refiner.py
# Lightweight NLP classifier with online learning:
# - TF-IDF on Description + simple numeric features (Amount bucket)
# - SGDClassifier(partial_fit) so we can learn from feedback without full retrain

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

VEC_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
CLF_PATH = os.path.join(MODEL_DIR, "sgd.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# Default taxonomy — extend as you like
DEFAULT_LABELS = [
    "Dining","Groceries","Shopping","Transportation","Utilities",
    "Housing","Health","Entertainment","Subscriptions","Transfers",
    "Income","Fees","Travel","Education","Uncategorized"
]

def _amount_bucket(x: pd.Series) -> sparse.csr_matrix:
    # coarse bins for amount; model learns typical ranges per category
    v = pd.to_numeric(x, errors="coerce").fillna(0.0).values.reshape(-1, 1)
    # bins: very small, small, medium, large, very large
    bins = np.digitize(v, [-100, -25, -5, 5, 25, 100])
    # one-hot encode bins
    n = v.shape[0]; k = 8
    rows = np.repeat(np.arange(n), 1)
    cols = bins.flatten().clip(0, k-1)
    data = np.ones(n)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, k))

def _build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000
    )

def _load_or_init():
    labels = DEFAULT_LABELS
    if os.path.exists(LABELS_PATH):
        try:
            labels = joblib.load(LABELS_PATH)
        except Exception:
            pass

    if os.path.exists(VEC_PATH) and os.path.exists(CLF_PATH):
        try:
            vec = joblib.load(VEC_PATH)
            clf = joblib.load(CLF_PATH)
            return vec, clf, labels
        except Exception:
            pass

    vec = _build_vectorizer()
    clf = SGDClassifier(loss="log_loss")  # probabilities for confidence
    # bootstrap partial_fit with available labels
    clf.partial_fit(np.zeros((1,1)), np.array([labels[-1]]), classes=np.array(labels, dtype=object))
    return vec, clf, labels

_VECTORIZER, _CLF, _LABELS = _load_or_init()

def _featurize(desc: pd.Series, amt: pd.Series):
    X_text = _VECTORIZER.transform(desc.fillna("").astype(str).values)
    X_amt = _amount_bucket(amt)
    return sparse.hstack([X_text, X_amt], format="csr")

def predict_descriptions(df: pd.DataFrame, return_conf=True) -> pd.DataFrame:
    """Return DataFrame with PredictedCategory (+confidence)."""
    global _VECTORIZER, _CLF, _LABELS
    desc = df.get("Description", pd.Series([""]*len(df)))
    amt = pd.to_numeric(df.get("Amount", 0), errors="coerce").fillna(0.0)

    # fit vectorizer vocabulary on the fly if empty
    if len(getattr(_VECTORIZER, "vocabulary_", {})) == 0:
        _VECTORIZER.fit(desc.fillna("").astype(str).values)

    X = _featurize(desc, amt)
    preds = _CLF.predict(X)

    out = pd.DataFrame({"PredictedCategory": preds}, index=df.index)

    if return_conf and hasattr(_CLF, "predict_proba"):
        proba = _CLF.predict_proba(X)  # shape (n, n_labels)
        conf = proba.max(axis=1)
        out["Confidence"] = conf
    else:
        out["Confidence"] = np.nan
    return out

def learn_feedback(samples: pd.DataFrame):
    """samples: DataFrame with Description, Amount, CorrectCategory"""
    global _VECTORIZER, _CLF, _LABELS
    if samples.empty:
        return

    y = samples["CorrectCategory"].astype(str).values
    # update labels if new ones appear
    new_labels = sorted(set(_LABELS) | set(y))
    if new_labels != _LABELS:
        _LABELS = new_labels
        joblib.dump(_LABELS, LABELS_PATH)

    # maintain vectorizer vocab
    if len(getattr(_VECTORIZER, "vocabulary_", {})) == 0:
        _VECTORIZER.fit(samples["Description"].fillna("").astype(str).values)

    X = _featurize(samples["Description"], pd.to_numeric(samples["Amount"], errors="coerce").fillna(0.0))
    _CLF.partial_fit(X, y, classes=np.array(_LABELS, dtype=object))

    # persist artifacts
    joblib.dump(_VECTORIZER, VEC_PATH)
    joblib.dump(_CLF, CLF_PATH)
    joblib.dump(_LABELS, LABELS_PATH)

def labels():
    return list(_LABELS)
