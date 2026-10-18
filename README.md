# 2. SMS Spam & Fraud Detector — Telecom

Classifies SMS messages as **spam/fraud** or **ham** (legitimate) — the same
filtering problem every telecom operator runs on its network daily.

**🔴 Live interactive demo:** the trained model runs in the browser —
open `docs/index.html` (or visit the GitHub Pages link), paste any SMS,
and see the verdict plus the top spam signals. The page's JavaScript was
verified to match the Python model exactly (max difference 1.9e-16 on
30 real messages).

## Dataset — big & combined

**9,058 unique SMS** (23.3% spam) after dedupe, built by `src/combine_data.py`:
- **UCI SMS Spam Collection** — 5,572 msgs (https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **ExAIS SMS Spam Dataset** — 5,240 msgs, Federal University of Agriculture, Abeokuta, Nigeria (https://github.com/AbayomiAlli/SMS-Spam-Dataset)

## Approach

1. **Features** — word TF-IDF (uni+bi-grams) **+** character TF-IDF (3–5 grams,
   catches obfuscated tricks like "fr33 m0ney"), each L2-normalized.
2. **Model** — balanced Logistic Regression, stratified 80/20 split.
3. **High-recall tuning** — decision threshold set to catch **95.3%** of spam.
   A missed scammer costs more than a false alarm, and the trade-off is
   reported honestly below.

## Results (held-out test set)

| Mode | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| **High-recall ⭐ (live)** | **0.911** | **0.739** | **0.953** | **0.833** |
| Default (0.50) | 0.921 | 0.796 | 0.887 | 0.839 |

Reading the live row: 95 of 100 spam messages are caught; about 3 in 4
flagged messages are truly spam. No model catches 100% — this is the
measured, honest operating point.

## Run it

```bash
pip install -r ../../requirements.txt
python src/combine_data.py  # build the combined dataset (needs /tmp/exais)
python src/train.py          # train + tune threshold + metrics.json + plot
python src/predict.py        # classify example messages
python src/export_web.py     # export weights -> docs/model.json (verified exact)
```

## Files

```
data/SMSSpamCollection      # original UCI file
data/sms_spam_combined.csv  # 9,058 combined training rows
src/combine_data.py
src/train.py
src/predict.py
src/export_web.py
models/spam_model.pkl
reports/                    # metrics.json, confusion_matrix.png
docs/                       # GitHub Pages site: index.html (live demo) + model.json + plot
```

## Go live on GitHub Pages

1. Push this folder as its own repo, 2. Settings → Pages → Deploy from branch → `main` → `/docs`.
   The demo needs no backend — the model runs as JavaScript in the visitor's browser.
