# 2. SMS Spam & Fraud Detector — Telecom

Classifies SMS messages as **spam/fraud** or **ham** (legitimate) — the same
filtering problem every telecom operator runs on its network daily.

**🔴 Live interactive demo:** the trained model runs in the browser —
open `docs/index.html` (or host `docs/` on GitHub Pages), paste any SMS,
and see the verdict plus the top spam signals. The page's JavaScript was
verified to match the Python model exactly (max difference 1.6e-15 on
30 real messages).

## Dataset

**SMS Spam Collection, UCI Machine Learning Repository** — 5,572 messages
(4,825 ham / 747 spam, spam rate 13.4%).

- UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Included in `data/` (`SMSSpamCollection`, tab-separated).

## Approach

1. **Features** — TF-IDF on word uni-grams + bi-grams (8,000 features,
   English stop words removed).
2. **Validation** — stratified 80/20 split.
3. **Models** — Multinomial Naive Bayes vs Logistic Regression, compared on F1.

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Naive Bayes** | **0.970** | **1.000** | **0.772** | **0.871** |
| Logistic Regression | 0.968 | 1.000 | 0.758 | 0.863 |

Naive Bayes won and is the saved model (`models/spam_model.pkl`).
Precision is 1.000 — the model never flags a legit message as spam
(zero false alarms), at the cost of missing ~23% of spam (recall 0.772).
That trade-off is deliberate: blocking real customer messages is worse
than letting a spam through.

## Run it

```bash
pip install -r ../../requirements.txt
python src/train.py     # trains both models, saves best + metrics.json + plot
python src/predict.py   # classifies 6 example messages (spam + legit)
```

## Files

```
data/SMSSpamCollection
src/train.py        # training + evaluation pipeline
src/predict.py      # demo: classify new messages
src/export_web.py   # exports TF-IDF + Naive Bayes weights -> docs/model.json (verified exact)
models/spam_model.pkl
reports/            # metrics.json, confusion_matrix.png
docs/               # GitHub Pages site: index.html (live demo) + model.json + plot
```

## Go live on GitHub Pages

1. Push this folder as its own repo, 2. Settings → Pages → Deploy from branch → `main` → `/docs`.
   The demo needs no backend — the model runs as JavaScript in the visitor's browser.
