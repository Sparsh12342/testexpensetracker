"""
machinelearningclassification.py
Expose a single function:

    predict_categories(df: pd.DataFrame) -> pd.Series[str]

Expected df columns: at least 'Description' and 'Amount' (strings/numbers).
You can extend features to match your trained pipeline.

Place your trained artifacts under, e.g.:
    1project/models/vectorizer.pkl
    1project/models/model.pkl
or adjust ARTIFACT_DIR below.

Run server with:
    PYTHONPATH=. python3 -m server.app
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  

# Optional: try to load scikit artifacts if available
_ARTIFACTS_LOADED = False
_VECTORIZER = None
_MODEL = None

# Look for artifacts in ../models relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "models")
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")

def _try_load_artifacts():
    global _ARTIFACTS_LOADED, _VECTORIZER, _MODEL
    if _ARTIFACTS_LOADED:
        return

    try:
        import joblib  # scikit-learn joblib
        if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
            _VECTORIZER = joblib.load(VEC_PATH)
            _MODEL = joblib.load(MODEL_PATH)
        _ARTIFACTS_LOADED = True
    except Exception:
        # Silently continue with rules-based fallback if joblib/sklearn not available
        _ARTIFACTS_LOADED = True
        _VECTORIZER = None
        _MODEL = None

def _rules_fallback(descriptions: pd.Series, amounts: pd.Series) -> pd.Series:
    """
    Very lightweight keyword/range rules so the API works before the real model is wired.
    Replace/remove once your artifacts are present.
    """
    cats = []
    for desc, amt in zip(descriptions.fillna("").str.lower(), amounts):
        cat = "Uncategorized"
        if any(k in desc for k in ["rent", "landlord", "lease"]):
            cat = "Housing"
        elif any(k in desc for k in ["uber", "lyft", "gas", "fuel", "metro", "subway", "toll"]):
            cat = "Transportation"
        elif any(k in desc for k in ["amazon", "walmart", "target", "grocery", "whole foods", "trader joe"]):
            cat = "Shopping"
        elif any(k in desc for k in ["netflix", "spotify", "hulu", "prime video"]):
            cat = "Subscriptions"
        elif any(k in desc for k in ["salary", "payroll", "direct deposit", "refund"]):
            cat = "Income"
        elif any(k in desc for k in ["restaurant", "dining", "mc", "bk", "kfc", "starbucks", "chipotle", "pizza"]):
            cat = "Dining"
        elif any(k in desc for k in ["gym", "fitness", "health", "doctor", "pharmacy"]):
            cat = "Health"
        elif any(k in desc for k in ["electric", "water", "gas bill", "utility", "internet", "wifi"]):
            cat = "Utilities"
        elif amt >= 0 and ("transfer" in desc or "zelle" in desc or "venmo" in desc):
            cat = "Transfers"
        cats.append(cat)
    return pd.Series(cats, index=descriptions.index)

def predict_categories(df: pd.DataFrame) -> pd.Series:
    """
    Primary API used by the Flask route. Returns a pd.Series[str] aligned with df.
    If trained artifacts are found, use them; otherwise use a rules fallback.
    """
    _try_load_artifacts()

    # Get features
    desc = df.get("Description", pd.Series([""] * len(df))).astype(str)
    amt = pd.to_numeric(df.get("Amount", 0), errors="coerce").fillna(0.0)

    # If you trained a text model with a vectorizer:
    if _VECTORIZER is not None and _MODEL is not None:
        try:
            # Example: text-only model on description
            X_text = _VECTORIZER.transform(desc.values)

            # If your model used amount too, you can combine here (simple example):
            # from scipy import sparse
            # amt_col = sparse.csr_matrix(amt.values.reshape(-1, 1))
            # X = sparse.hstack([X_text, amt_col], format="csr")
            X = X_text

            preds = _MODEL.predict(X)
            # If your model outputs numeric labels, map to strings here
            # label_mapping = {0: "Housing", 1: "Dining", ...}
            # preds = [label_mapping.get(p, "Uncategorized") for p in preds]
            return pd.Series(preds, index=df.index).astype(str)
        except Exception:
            # If anything fails, fall back to rules so the API still responds
            return _rules_fallback(desc, amt)

    # No artifacts available → rules
    return _rules_fallback(desc, amt)
