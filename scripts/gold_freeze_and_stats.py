import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_new"

GOLD_IN = DATA / "gold_raw_norm_FROZEN.csv"
OUT_IDS = DATA / "gold_ids_FROZEN.csv"
OUT_STATS = DATA / "gold_stats.txt"

LABEL_COL = "gold_label"   # 你如果叫 label/y/anno_label 就改这里
DOMAIN_COL = "domain"
ID_COL = "doc_id"

def main():
    df = pd.read_csv(GOLD_IN)
    df[ID_COL] = df[ID_COL].astype(str)

    # 1) ids
    df[[ID_COL]].drop_duplicates().to_csv(OUT_IDS, index=False, encoding="utf-8-sig")

    # 2) stats
    lines = []
    lines.append(f"rows: {len(df)}")
    lines.append(f"unique doc_id: {df[ID_COL].nunique()}")
    lines.append("\nLabel counts:")
    lines.append(str(df[LABEL_COL].value_counts(dropna=False)))
    lines.append("\nDomain counts:")
    lines.append(str(df[DOMAIN_COL].value_counts(dropna=False)))

    OUT_STATS.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_STATS.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()