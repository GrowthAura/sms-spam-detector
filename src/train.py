"""
SMS Spam & Fraud Detector — Telecom
====================================
Classifies SMS messages as spam (incl. scam/fraud) or ham (legitimate).

Dataset: SMS Spam Collection, UCI Machine Learning Repository
         (5,572 messages: 4,825 ham / 747 spam)
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Pipeline:
    1. Load messages, map label -> 1 (spam) / 0 (ham)
    2. TF-IDF (uni+bi-grams, English stop words) + classifier
    3. Stratified 80/20 split; compare Naive Bayes vs Logistic Regression
    4. Save best pipeline, metrics.json and confusion-matrix plot

Run:
    python src/train.py
"""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "SMSSpamCollection"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t", header=None,
                     names=["label", "message"], encoding="utf-8")
    df["label"] = (df["label"] == "spam").astype(int)
    return df


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }


def main():
    df = load_data()
    print(f"Loaded {len(df):,} messages | spam rate: {df['label'].mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2,
        stratify=df["label"], random_state=42,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=8000, ngram_range=(1, 2)
    )
    candidates = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = evaluate(name, pipe, X_test, y_test)
        m = results[name]
        print(f"{name:20s} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best_name}")
    joblib.dump(fitted[best_name], MODELS_DIR / "spam_model.pkl")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    cm = confusion_matrix(y_test, fitted[best_name].predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    print("Saved: models/spam_model.pkl, reports/metrics.json + confusion matrix")


if __name__ == "__main__":
    main()
