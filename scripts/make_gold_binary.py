import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_new"

GOLD_IN = DATA / "gold_raw_norm_FROZEN.csv"
OUT = DATA / "gold_raw_norm_FROZEN.csv"

df = pd.read_csv(str(GOLD_IN))
df = df[df["gold_label"].isin(["ADVICE", "STORY"])].copy()
df.to_csv(str(OUT), index=False, encoding="utf-8-sig")
print("Wrote:", OUT, "rows:", len(df))

