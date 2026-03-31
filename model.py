"""
Spam Message Detection — Model Pipeline
========================================
Handles data loading, text preprocessing, model training (TF-IDF + Logistic Regression),
and prediction logic for the Streamlit app.
"""

import os
import re
import string
import urllib.request

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DATA_DIR, "spam.csv")
MODEL_PATH = os.path.join(DATA_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Ensure NLTK data is available
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_or_download_data() -> pd.DataFrame:
    """Load the SMS Spam Collection dataset. Downloads it if not present."""
    if os.path.exists(DATA_PATH):
        # Try common CSV formats
        try:
            df = pd.read_csv(DATA_PATH, encoding="latin-1")
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]]
                df.columns = ["label", "message"]
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            if "label" in df.columns and "message" in df.columns:
                return df.dropna(subset=["label", "message"]).reset_index(drop=True)
        except Exception:
            pass

    # Download from public mirror (tab-separated)
    print("Downloading SMS Spam Collection dataset …")
    urllib.request.urlretrieve(DATASET_URL, DATA_PATH)
    df = pd.read_csv(
        DATA_PATH, sep="\t", header=None, names=["label", "message"]
    )
    return df.dropna(subset=["label", "message"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Clean a single message string."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"\d+", "", text)  # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    """
    Train TF-IDF + Logistic Regression.

    Returns
    -------
    model, vectorizer, metrics : tuple
        metrics is a dict containing everything needed by the dashboard.
    """
    df = df.copy()
    df["clean_message"] = df["message"].apply(preprocess_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # Feature importance (top TF-IDF terms)
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_spam_idx = np.argsort(coefs)[-20:][::-1]
    top_ham_idx = np.argsort(coefs)[:20]
    top_spam_words = [(feature_names[i], coefs[i]) for i in top_spam_idx]
    top_ham_words = [(feature_names[i], coefs[i]) for i in top_ham_idx]

    metrics = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "top_spam_words": top_spam_words,
        "top_ham_words": top_ham_words,
        "X_test": X_test,
    }

    # Persist
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, metrics


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_message(text: str, model, vectorizer):
    """
    Predict spam/ham for a single message.

    Returns
    -------
    label : str   ('Spam' or 'Ham')
    confidence : float  (0–1)
    """
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "Spam" if pred == 1 else "Ham"
    confidence = float(proba[pred])
    return label, confidence


def predict_batch(texts, model, vectorizer):
    """Predict on a list / Series of raw texts."""
    cleaned = [preprocess_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned)
    preds = model.predict(vecs)
    probas = model.predict_proba(vecs)
    labels = ["Spam" if p == 1 else "Ham" for p in preds]
    confidences = [float(probas[i][preds[i]]) for i in range(len(preds))]
    return labels, confidences
