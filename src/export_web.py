"""
Export the trained spam model to docs/model.json for the live web demo.

The JSON contains everything the static page needs to classify messages in
pure JavaScript (no backend):
  - TF-IDF vocabulary (term -> index) + idf weights + stop words
  - Naive Bayes feature_log_prob_ + class_log_prior_

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


def main():
    model = joblib.load(BASE / "models" / "spam_model.pkl")
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    vocab = {str(t): int(i) for t, i in tfidf.vocabulary_.items()}
    stop = sorted(str(w) for w in tfidf.get_stop_words())
    payload = {
        "vocabulary": vocab,
        "idf": [float(x) for x in tfidf.idf_],
        "stop_words": stop,
        "feature_log_prob": [[float(x) for x in row] for row in clf.feature_log_prob_],
        "class_log_prior": [float(x) for x in clf.class_log_prior_],
        "metrics": json.loads((BASE / "reports" / "metrics.json").read_text())["naive_bayes"],
    }
    (DOCS / "model.json").write_text(json.dumps(payload))
    print(f"Export OK -> docs/model.json ({len(vocab):,} terms)")

    # --- Verify: JSON-based manual scoring must match predict_proba exactly ---
    stop_set = set(stop)
    idf = np.array(payload["idf"])
    flp = np.array(payload["feature_log_prob"])
    clp = np.array(payload["class_log_prior"])

    def manual_proba(text):
        tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in stop_set]
        counts = {}
        for n in (1, 2):
            for i in range(len(tokens) - n + 1):
                term = " ".join(tokens[i:i + n])
                if term in vocab:
                    counts[term] = counts.get(term, 0) + 1
        idxs = np.array([vocab[t] for t in counts], dtype=int)
        if len(idxs) == 0:
            scores = clp.copy()  # no known terms -> fall back to class priors
        else:
            vals = np.array([counts[t] * idf[vocab[t]] for t in counts])
            norm = np.sqrt((vals ** 2).sum()) or 1.0
            vals = vals / norm
            scores = clp + (flp[:, idxs] * vals).sum(axis=1)
        m = scores.max()
        exps = np.exp(scores - m)
        return float(exps[1] / exps.sum())

    df = pd.read_csv(BASE / "data" / "SMSSpamCollection", sep="\t",
                     header=None, names=["label", "message"], encoding="utf-8")
    sample = df["message"].sample(200, random_state=11)
    expected = model.predict_proba(sample)[:, 1]
    got = np.array([manual_proba(t) for t in sample])
    max_diff = float(np.max(np.abs(expected - got)))
    print(f"Verified on 200 messages | max |model - manual| = {max_diff:.2e}")
    assert max_diff < 1e-9, "Exported weights do not reproduce the model!"


if __name__ == "__main__":
    main()
