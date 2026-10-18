"""
Export the trained spam model to docs/model.json for the live web demo.

Architecture: FeatureUnion(word TF-IDF + char TF-IDF) -> LogisticRegression.
The JSON contains everything the static page needs to classify messages in
pure JavaScript (no backend):
  - word vocabulary/idf/stop-words, char vocabulary/idf
  - logistic regression coefficients + intercept + tuned threshold

Run:
    python src/export_web.py
Then open docs/index.html (or host docs/ on GitHub Pages).
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DOCS = BASE / "docs"
DOCS.mkdir(exist_ok=True)

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def word_tfidf(text, vocab, idf, stop_set):
    tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in stop_set]
    counts = {}
    for n in (1, 2):
        for i in range(len(tokens) - n + 1):
            term = " ".join(tokens[i:i + n])
            if term in vocab:
                counts[term] = counts.get(term, 0) + 1
    return _l2({vocab[t]: c * idf[vocab[t]] for t, c in counts.items()})


def char_tfidf(text, vocab, idf):
    counts = {}
    for w in text.lower().split():
        pw = " " + w + " "
        for n in (3, 4, 5):
            for i in range(len(pw) - n + 1):
                term = pw[i:i + n]
                if term in vocab:
                    counts[term] = counts.get(term, 0) + 1
    return _l2({vocab[t]: c * idf[vocab[t]] for t, c in counts.items()})


def _l2(d):
    norm = np.sqrt(sum(v * v for v in d.values())) or 1.0
    return {k: v / norm for k, v in d.items()}


def main():
    model = joblib.load(BASE / "models" / "spam_model.pkl")
    union = model.named_steps["union"]
    clf = model.named_steps["clf"]
    transformers = dict(union.transformer_list)
    wv = transformers["word"]
    cv = transformers["char"]

    metrics = json.loads((BASE / "reports" / "metrics.json").read_text())
    payload = {
        "word_vocab": {str(t): int(i) for t, i in wv.vocabulary_.items()},
        "word_idf": [float(x) for x in wv.idf_],
        "stop_words": sorted(str(w) for w in wv.get_stop_words()),
        "char_vocab": {str(t): int(i) for t, i in cv.vocabulary_.items()},
        "char_idf": [float(x) for x in cv.idf_],
        "coef_word": [float(x) for x in clf.coef_[0][:len(wv.vocabulary_)]],
        "coef_char": [float(x) for x in clf.coef_[0][len(wv.vocabulary_):]],
        "intercept": float(clf.intercept_[0]),
        "threshold": metrics["high_recall"]["threshold"],
        "metrics": metrics["high_recall"],
    }
    (DOCS / "model.json").write_text(json.dumps(payload))
    print(f"Export OK -> docs/model.json "
          f"({len(payload['word_vocab']):,} word + {len(payload['char_vocab']):,} char terms)")

    # --- Verify: JSON-based manual scoring must match predict_proba exactly ---
    stop_set = set(payload["stop_words"])
    w_idf = np.array(payload["word_idf"])
    c_idf = np.array(payload["char_idf"])
    cw = np.array(payload["coef_word"])
    cc = np.array(payload["coef_char"])
    b = payload["intercept"]

    def manual_proba(text):
        w = word_tfidf(text, payload["word_vocab"], w_idf, stop_set)
        c = char_tfidf(text, payload["char_vocab"], c_idf)
        z = b + sum(v * cw[i] for i, v in w.items()) + sum(v * cc[i] for i, v in c.items())
        return 1.0 / (1.0 + np.exp(-z))

    df = pd.read_csv(BASE / "data" / "sms_spam_combined.csv")
    sample = df["message"].sample(200, random_state=11)
    expected = model.predict_proba(sample)[:, 1]
    got = np.array([manual_proba(str(t)) for t in sample])
    max_diff = float(np.max(np.abs(expected - got)))
    print(f"Verified on 200 messages | max |model - manual| = {max_diff:.2e}")
    assert max_diff < 1e-9, "Exported weights do not reproduce the model!"


if __name__ == "__main__":
    main()
