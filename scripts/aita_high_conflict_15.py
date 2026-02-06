import os
import pandas as pd

ROOT = r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample"
DATA = os.path.join(ROOT, "data_new")

MERGED = os.path.join(DATA, "gold267_merged_preds_raw_clean.csv")

OUT_XLSX = os.path.join(DATA, "aita_high_conflict_15.xlsx")
OUT_CSV  = os.path.join(DATA, "aita_high_conflict_15.csv")

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing columns. Tried: {candidates}. Found: {list(df.columns)[:30]} ...")
    return None

def main():
    if not os.path.exists(MERGED):
        raise FileNotFoundError(f"Not found: {MERGED}\n(You need to run merge_and_eval_preds.py first.)")

    df = pd.read_csv(MERGED)

    domain_col = pick_col(df, ["domain", "Domain"])
    gold_col   = pick_col(df, ["gold_label", "gold", "label", "y_true"])
    pred_raw   = pick_col(df, ["pred_raw", "pred_tfidf_raw", "pred_x_raw", "pred_raw_tfidf", "pred"], required=False)
    pred_clean = pick_col(df, ["pred_clean", "pred_tfidf_clean", "pred_clean_v1", "pred_x_clean", "pred_cleaned"], required=False)

    # 如果 merged 里没有 pred_raw/pred_clean，就尝试从常见命名里找
    if pred_raw is None:
        pred_raw = pick_col(df, ["pred_label_raw", "pred_label_x", "pred_label"], required=True)
    if pred_clean is None:
        pred_clean = pick_col(df, ["pred_label_clean", "pred_label_clean_v1", "pred_label_y", "pred_label_1"], required=True)

    text_col = pick_col(df, ["text_clean_v1", "text", "raw_text", "body"], required=False)

    aita = df[df[domain_col].astype(str).str.lower().eq("aita")].copy()
    if len(aita) == 0:
        raise ValueError("No rows with domain == 'aita' found.")

    # conflict_score：越大越值得看（raw/clean分歧 + 是否错）
    aita["wrong_raw"]   = (aita[pred_raw].astype(str)   != aita[gold_col].astype(str)).astype(int)
    aita["wrong_clean"] = (aita[pred_clean].astype(str) != aita[gold_col].astype(str)).astype(int)
    aita["raw_vs_clean_diff"] = (aita[pred_raw].astype(str) != aita[pred_clean].astype(str)).astype(int)

    aita["conflict_score"] = (
        2 * aita["raw_vs_clean_diff"] +     # raw/clean 不一致更值得看
        1 * aita["wrong_raw"] +
        1 * aita["wrong_clean"]
    )

    # 只挑“至少有一个错 or raw/clean有分歧”的，避免抽到全对的
    cand = aita[(aita["wrong_raw"] + aita["wrong_clean"] + aita["raw_vs_clean_diff"]) > 0].copy()

    cand = cand.sort_values(
        by=["conflict_score", "raw_vs_clean_diff", "wrong_raw", "wrong_clean"],
        ascending=False
    )

    keep_cols = [c for c in [
        "doc_id", domain_col, gold_col, pred_raw, pred_clean,
        "conflict_score", "raw_vs_clean_diff", "wrong_raw", "wrong_clean",
        text_col
    ] if c is not None and c in cand.columns]

    top15 = cand[keep_cols].head(15)

    top15.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    top15.to_excel(OUT_XLSX, index=False)

    print(f"Saved: {OUT_XLSX}")
    print(f"Saved: {OUT_CSV}")
    print(f"n(AITA)={len(aita)}  candidates={len(cand)}  exported={len(top15)}")

if __name__ == "__main__":
    main()