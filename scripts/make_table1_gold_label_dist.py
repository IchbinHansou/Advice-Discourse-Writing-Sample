# scripts/make_table1_table2_from_frozen.py
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

GOLD_FROZEN = DATA / "gold_raw_norm_FROZEN.csv"
SILVER_FROZEN = DATA / "silver_train_raw_norm_FROZEN.csv"

OUT_MD = DATA / "table1_table2_FROZEN.md"
OUT_CSV1 = DATA / "table1_gold_label_by_domain_FROZEN.csv"
OUT_CSV2 = DATA / "table2_length_medians_FROZEN.csv"

LABEL_COL = "gold_label"   # gold file uses gold_label
DOMAIN_COL = "domain"
TEXT_COL = "text"

def word_count(s: str) -> int:
    s = "" if s is None else str(s)
    s = s.strip()
    return 0 if not s else len(s.split())

def main():
    gold = pd.read_csv(GOLD_FROZEN, low_memory=False, na_filter=False)
    if LABEL_COL not in gold.columns:
        raise RuntimeError(f"Gold missing '{LABEL_COL}'. Columns={list(gold.columns)[:30]}")
    if DOMAIN_COL not in gold.columns:
        raise RuntimeError(f"Gold missing '{DOMAIN_COL}'. Columns={list(gold.columns)[:30]}")
    if TEXT_COL not in gold.columns:
        raise RuntimeError(f"Gold missing '{TEXT_COL}'. Columns={list(gold.columns)[:30]}")

    # ---------- Table 1 ----------
    tab1 = (
        gold.pivot_table(index=DOMAIN_COL, columns=LABEL_COL, values="doc_id",
                         aggfunc="count", fill_value=0)
        .reset_index()
    )
    # ensure columns order
    for c in ["ADVICE", "STORY", "MIXED"]:
        if c not in tab1.columns:
            tab1[c] = 0
    tab1 = tab1[[DOMAIN_COL, "ADVICE", "STORY", "MIXED"]]
    tab1["Total"] = tab1[["ADVICE", "STORY", "MIXED"]].sum(axis=1)

    totals = pd.DataFrame([{
        DOMAIN_COL: "Total",
        "ADVICE": int(tab1["ADVICE"].sum()),
        "STORY": int(tab1["STORY"].sum()),
        "MIXED": int(tab1["MIXED"].sum()),
        "Total": int(tab1["Total"].sum())
    }])
    tab1_out = pd.concat([tab1, totals], ignore_index=True)
    tab1_out.to_csv(OUT_CSV1, index=False, encoding="utf-8-sig")

    # ---------- Table 2 ----------
    gold_wc = gold[TEXT_COL].map(word_count)
    gold_median = int(gold_wc.median())

    # per-label medians (gold)
    gold_len_by_label = (
        gold.assign(word_count=gold_wc)
            .groupby(LABEL_COL)["word_count"]
            .median()
            .round()
            .astype(int)
            .to_dict()
    )

    silver = pd.read_csv(SILVER_FROZEN, low_memory=False, na_filter=False)
    if TEXT_COL not in silver.columns:
        raise RuntimeError(f"Silver missing '{TEXT_COL}'. Columns={list(silver.columns)[:30]}")
    silver_wc = silver[TEXT_COL].map(word_count)
    silver_median = int(silver_wc.median())

    rows = [
        {"Split / Label": "Silver (weak, train)", "N": int(len(silver)), "Word median": silver_median},
        {"Split / Label": "Gold (manual, eval)", "N": int(len(gold)), "Word median": gold_median},
        {"Split / Label": "Gold — Advice", "N": int((gold[LABEL_COL] == "ADVICE").sum()),
         "Word median": int(gold_len_by_label.get("ADVICE", 0))},
        {"Split / Label": "Gold — Disclosure", "N": int((gold[LABEL_COL] == "STORY").sum()),
         "Word median": int(gold_len_by_label.get("STORY", 0))},
        {"Split / Label": "Gold — Mixed", "N": int((gold[LABEL_COL] == "MIXED").sum()),
         "Word median": int(gold_len_by_label.get("MIXED", 0))},
    ]
    tab2_out = pd.DataFrame(rows)
    tab2_out.to_csv(OUT_CSV2, index=False, encoding="utf-8-sig")

    # ---------- Write a simple markdown for copy-paste ----------
    mixed_n = int((gold[LABEL_COL] == "MIXED").sum())
    mixed_pct = mixed_n / len(gold) * 100

    md = []
    md.append("## Table 1 — Gold-set label distribution by domain (FROZEN)\n")
    md.append(tab1_out.to_markdown(index=False))
    md.append("\n\n")
    md.append("## Table 2 — Length summary (medians; whitespace-delimited words) (FROZEN)\n")
    md.append(tab2_out.to_markdown(index=False))
    md.append("\n\n")
    md.append(
        f"As detailed in Table 1, the manual annotation yielded "
        f"{int((gold[LABEL_COL]=='ADVICE').sum())} Advice and {int((gold[LABEL_COL]=='STORY').sum())} Disclosure posts, "
        f"alongside {mixed_n} Mixed-intent posts ({mixed_pct:.2f}% of the gold set). "
        f"For the primary binary evaluation, we exclude Mixed posts, resulting in a final evaluation subset of "
        f"{len(gold) - mixed_n} posts.\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")
    print("Wrote:")
    print(" ", OUT_CSV1)
    print(" ", OUT_CSV2)
    print(" ", OUT_MD)

if __name__ == "__main__":
    main()
