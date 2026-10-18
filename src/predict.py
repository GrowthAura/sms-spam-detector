"""
Spam detector demo — classify example messages with the trained model.

Run:
    python src/predict.py
"""

from pathlib import Path

import joblib

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "models" / "spam_model.pkl"

EXAMPLES = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim: bit.ly/xyz123",
    "Hey, are we still meeting for lunch tomorrow at 12?",
    "URGENT: Your bank account has been locked. Verify now at secure-bank-login.tk",
    "Your package will arrive Thursday between 2-4pm. Track: ups.com/123",
    "FREE entry to win a brand new iPhone! Text WIN to 55555 now!!!",
    "Mom, can you pick me up from school at 3? Practice got cancelled.",
]


def main():
    model = joblib.load(MODEL)
    for msg in EXAMPLES:
        pred = model.predict([msg])[0]
        proba = model.predict_proba([msg])[0, 1]
        label = "SPAM/FRAUD" if pred == 1 else "ham (legit)"
        print(f"[{label:12s} {proba:5.1%}] {msg[:70]}")


if __name__ == "__main__":
    main()
