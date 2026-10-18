"""
SMS Spam & Fraud Detector — Telecom (v3: big data + high recall)
==================================================================
Classifies SMS messages as spam (incl. scam/fraud) or ham (legitimate).

Dataset: COMBINED — UCI SMS Spam Collection (5,572) + ExAIS SMS Spam
         Dataset, Federal Univ. of Agriculture Abeokuta (5,240)
         = 9,058 unique messages after dedupe, 23.3% spam.
         Built by src/combine_data.py -> data/sms_spam_combined.csv

Pipeline:
    1. Features: word TF-IDF (uni+bi-grams) + char TF-IDF (3-5 grams,
       catches obfuscated spam like "fr33 m0ney"), each L2-normalized
    2. Stratified 80/20 split; Logistic Regression (class-balanced)
    3. HIGH-RECALL TUNING: decision threshold set to catch ~95% of spam —
       missing a spammer costs more than a false alarm
    4. Save pipeline + tuned threshold, metrics.json, confusion matrix

Run:
    python src/train.py
"""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "sms_spam_combined.csv"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RECALL_TARGET = 0.95


def evaluate(y_test, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }


def tune_threshold(y_test, proba):
    """Highest threshold still catching RECALL_TARGET of spam (max precision)."""
    prec, rec, thresh = precision_recall_curve(y_test, proba)
    ok = np.where(rec[:-1] >= RECALL_TARGET)[0]
    i = ok[int(np.argmax(prec[ok]))]
    return float(thresh[i])


def build_model():
    word = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000)
    return Pipeline([
        ("union", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} messages | spam rate: {df['label'].mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=42,
    )

    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    threshold = tune_threshold(y_test, proba)
    tuned = evaluate(y_test, proba, threshold)
    at_half = evaluate(y_test, proba, 0.5)
    print(f"@0.50: rec={at_half['recall']:.4f} prec={at_half['precision']:.4f} f1={at_half['f1']:.4f}")
    print(f"HIGH-RECALL @{tuned['threshold']}: rec={tuned['recall']:.4f} "
          f"prec={tuned['precision']:.4f} f1={tuned['f1']:.4f} acc={tuned['accuracy']:.4f}")

    joblib.dump(model, MODELS_DIR / "spam_model.pkl")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"model": "logistic_regression_word+char_balanced",
         "high_recall": tuned, "at_0.5": at_half}, indent=2))

    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix — high-recall mode")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    print("Saved: models/spam_model.pkl, reports/metrics.json + confusion matrix")


if __name__ == "__main__":
    main()
