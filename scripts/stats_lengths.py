import re
import pandas as pd
from pathlib import Path

# ===== paths (edit) =====
DATA = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new")
SILVER_PATH = DATA / "silver_train_raw_norm_FROZEN.csv"                  # or your actual silver raw
GOLD_PATH   = DATA / "gold_raw_norm_FROZEN.csv"    # your gold file

# ===== helpers =====
_SENT_SPLIT = re.compile(r"(?:[.!?]+|\n+)\s*", flags=re.MULTILINE)

def count_tokens(text: str) -> int:
    if text is None:
        return 0
    t = str(text).strip()
    if not t:
        return 0
    return len(t.split())

def count_sent_like(text: str) -> int:
    if text is None:
        return 0
    t = str(text).strip()
    if not t:
        return 0
    parts = [p.strip() for p in _SENT_SPLIT.split(t)]
    parts = [p for p in parts if p]  # drop empty
    return len(parts)

def add_len_cols(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["token_len"] = out[text_col].map(count_tokens)
    out["sent_len"] = out[text_col].map(count_sent_like)
    return out

def summarize(df: pd.DataFrame, group_cols, name: str):
    g = df.groupby(group_cols, dropna=False).agg(
        n=("token_len", "size"),
        token_median=("token_len", "median"),
        sent_median=("sent_len", "median"),
        token_max=("token_len", "max"),
        sent_max=("sent_len", "max"),
    ).reset_index()
    print(f"\n==== {name} ====")
    print(g.to_string(index=False))
    return g

def main():
    silver = pd.read_csv(SILVER_PATH, low_memory=False)
    gold = pd.read_csv(GOLD_PATH)

    # pick columns (adjust if your column names differ)
    silver_label_col = "silver_label"   # your current print uses this
    gold_label_col = "gold_label"
    text_col = "text"

    for col in [silver_label_col, text_col]:
        if col not in silver.columns:
            raise ValueError(f"silver missing column: {col}")
    for col in [gold_label_col, text_col]:
        if col not in gold.columns:
            raise ValueError(f"gold missing column: {col}")

    silver2 = add_len_cols(silver, text_col=text_col)
    gold2 = add_len_cols(gold, text_col=text_col)

    # overall
    print("\n==== OVERALL ====")
    print("silver:", len(silver2), "gold:", len(gold2))
    print("silver token median:", int(silver2["token_len"].median()))
    print("gold   token median:", int(gold2["token_len"].median()))
    print("silver sent median:", int(silver2["sent_len"].median()))
    print("gold   sent median:", int(gold2["sent_len"].median()))

    # per label (and domain if you want)
    summarize(silver2, [silver_label_col], "SILVER per label")
    summarize(gold2, ["domain", gold_label_col], "GOLD domain × label")
    summarize(gold2, [gold_label_col], "GOLD per label")

    # optional: save
    out_dir = DATA / "stats_out"
    out_dir.mkdir(exist_ok=True)
    summarize(silver2, [silver_label_col], "SILVER per label").to_csv(out_dir / "silver_len_by_label.csv", index=False)
    summarize(gold2, ["domain", gold_label_col], "GOLD domain × label").to_csv(out_dir / "gold_len_by_domain_label.csv", index=False)
    summarize(gold2, [gold_label_col], "GOLD per label").to_csv(out_dir / "gold_len_by_label.csv", index=False)
    print("\nWrote CSVs to:", out_dir)

if __name__ == "__main__":
    main()