import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

GOLD_FROZEN = DATA / "gold_raw_norm_FROZEN.csv"
OUT_CSV = DATA / "table1_gold_label_by_domain.csv"

LABEL_MAP = {
    "ADVICE": "Advice",
    "STORY": "Disclosure",     # <- 关键：把 STORY 映射成 Disclosure
    "DISCLOSURE": "Disclosure",
    "MIXED": "Mixed",
}

DOMAIN_MAP = {
    "ra": "RA (r/relationship_advice)",
    "confession": "Confession (4 subreddits)",
    "aita": "AITA (r/AmItheAsshole)",
}

def main():
    df = pd.read_csv(GOLD_FROZEN, low_memory=False, na_filter=False)

    # sanity
    need = {"doc_id", "domain", "gold_label"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {GOLD_FROZEN}: {sorted(missing)}")

    df = df.copy()
    df["gold_label"] = df["gold_label"].astype(str).str.upper().str.strip()
    df["domain"] = df["domain"].astype(str).str.lower().str.strip()

    # de-dup by doc_id just in case
    df = df.drop_duplicates(subset=["doc_id"])

    df["Label"] = df["gold_label"].map(LABEL_MAP).fillna(df["gold_label"])
    df["Domain"] = df["domain"].map(DOMAIN_MAP).fillna(df["domain"])

    order_labels = ["Advice", "Disclosure", "Mixed"]

    tab = pd.crosstab(df["Domain"], df["Label"])
    # ensure columns exist
    for c in order_labels:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[order_labels].copy()
    tab["Total"] = tab.sum(axis=1)

    # add total row
    total = pd.DataFrame([tab.sum(axis=0)], index=["Total"])
    tab2 = pd.concat([tab, total], axis=0)

    # pretty print
    print("\nTable 1 — Gold-set label distribution by domain\n")
    print(tab2.to_string())

    # write csv
    tab2.reset_index().rename(columns={"index": "Domain"}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\nWrote:", OUT_CSV)

    # quick percentage check
    mixed = int(tab2.loc["Total", "Mixed"])
    n = int(tab2.loc["Total", "Total"])
    print(f"\nMixed rate: {mixed}/{n} = {mixed/n:.4%}")

if __name__ == "__main__":
    main()