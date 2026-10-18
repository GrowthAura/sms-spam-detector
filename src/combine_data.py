"""
Combine UCI SMS Spam Collection + ExAIS SMS Spam Dataset into one big
training file.

UCI:   5,572 msgs (747 spam)   — data/SMSSpamCollection
ExAIS: 5,240 msgs (2,350 spam) — /tmp/exais (Federal Univ. of Agriculture, Abeokuta)
Combined: ~10,812 msgs, ~3,097 spam (deduped)

Run:
    python src/combine_data.py
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EXAIS_DIR = Path("/tmp/exais/exais_data/ExAIS_SMS Spam Dataset")


def load_uci() -> pd.DataFrame:
    df = pd.read_csv(BASE / "data" / "SMSSpamCollection", sep="\t",
                     header=None, names=["label", "message"], encoding="utf-8")
    df["label"] = (df["label"] == "spam").astype(int)
    return df[["message", "label"]]


def load_exais() -> pd.DataFrame:
    frames = []
    for f in sorted(EXAIS_DIR.glob("USER *.csv")):
        df = pd.read_csv(f, header=None)
        label = df[6].map({"SPAM": 1, "HAM": 0})
        # message text was split across comma-separated columns -> rejoin
        msg = df.loc[:, 7:].apply(
            lambda r: ",".join(str(x) for x in r if str(x) != "nan").strip(), axis=1)
        frames.append(pd.DataFrame({"message": msg, "label": label}))
    out = pd.concat(frames, ignore_index=True)
    return out[out["label"].notna()].copy()


def main():
    uci = load_uci()
    exais = load_exais()
    combined = pd.concat([uci, exais], ignore_index=True)
    combined["message"] = combined["message"].str.strip()
    combined = combined[combined["message"].str.len() > 0]
    before = len(combined)
    combined = combined.drop_duplicates(subset=["message"]).reset_index(drop=True)
    print(f"UCI: {len(uci):,} | ExAIS: {len(exais):,} | combined: {before:,} "
          f"-> {len(combined):,} after dedupe")
    print(f"spam: {int(combined['label'].sum()):,} "
          f"({combined['label'].mean():.1%}) | ham: {int((1 - combined['label']).sum()):,}")
    out = BASE / "data" / "sms_spam_combined.csv"
    combined.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
