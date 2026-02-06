import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_new"

SILVER_RAW = DATA / "silver_train_raw_norm_FROZEN.csv"
SILVER_CLN = DATA / "silver_train_clean_v1_FROZEN.csv"
GOLD_RAW_XLSX = DATA / "gold_raw_norm_FROZEN.csv"
GOLD_CLN = DATA / "gold_clean_v1_FROZEN.csv"

PATS = {
    "aita_acronym": re.compile(r"\bAITA\b", re.I),
    "verdicts": re.compile(r"\b(YTA|NTA|ESH|NAH)\b", re.I),
    "tldr_edit_update": re.compile(r"\b(TL;?DR|EDIT|UPDATE)\b", re.I),
    "age_gender_bracket": re.compile(r"\[\s*\d{1,2}\s*[MF]\s*\]", re.I),
}

def rate(df, name):
    txt = df["text"].astype(str)
    print(f"\n[MARKERS] {name} n={len(df)}")
    for k, pat in PATS.items():
        print(f"  {k}: {txt.str.contains(pat).mean():.4f}")

def main():
    silver_raw = pd.read_csv(SILVER_RAW, low_memory=False)
    silver_cln = pd.read_csv(SILVER_CLN, low_memory=False)

    gold_raw = pd.read_csv(GOLD_RAW_XLSX)
    gold_cln = pd.read_csv(GOLD_CLN, low_memory=False)

    # binary subset only (if present)
    if "gold_label" in gold_raw.columns:
        gold_raw = gold_raw[gold_raw["gold_label"].isin(["ADVICE","STORY"])].copy()
    if "gold_label" in gold_cln.columns:
        gold_cln = gold_cln[gold_cln["gold_label"].isin(["ADVICE","STORY"])].copy()

    rate(silver_raw, "silver_raw")
    rate(silver_cln, "silver_clean")
    rate(gold_raw, "gold_raw_binary")
    rate(gold_cln, "gold_clean_binary")

if __name__ == "__main__":
    main()
