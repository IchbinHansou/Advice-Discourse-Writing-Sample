from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

LABELS = ["ADVICE", "STORY"]

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro"))

def train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    Xtr = train_df["text"].astype(str).tolist()
    ytr = train_df["silver_label"].astype(str).str.upper().tolist()

    te = test_df.copy()
    te["gold_label"] = te["gold_label"].astype(str).str.upper()
    te = te[te["gold_label"].isin(LABELS)].copy()

    Xte = te["text"].astype(str).tolist()

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=3, max_features=200_000, lowercase=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=1004, class_weight="balanced")
    clf.fit(Xtr_v, ytr)

    pred = clf.predict(Xte_v)

    out = te[["doc_id", "domain", "gold_label"]].copy()
    out["pred_label"] = pred
    return out

def summarize(pred_df: pd.DataFrame) -> Dict:
    overall = macro_f1(pred_df["gold_label"].tolist(), pred_df["pred_label"].tolist())
    by_dom = {}
    for dom in sorted(pred_df["domain"].astype(str).unique().tolist()):
        sub = pred_df[pred_df["domain"].astype(str) == dom]
        by_dom[dom] = {
            "n": int(len(sub)),
            "macro_f1": macro_f1(sub["gold_label"].tolist(), sub["pred_label"].tolist()),
            "label_counts": sub["gold_label"].value_counts().to_dict(),
        }
    return {"overall": {"n": int(len(pred_df)), "macro_f1": overall}, "by_domain": by_dom}

def main():
    root = repo_root()
    data = root / "data_new"

    silver_raw_path   = data / "silver_train_raw_norm_FROZEN.csv"
    silver_clean_path = data / "silver_train_clean_v1_FROZEN.csv"
    gold_raw_path     = data / "gold_raw_norm_FROZEN.csv"
    gold_clean_path   = data / "gold_clean_v1_FROZEN.csv"

    for p in [silver_raw_path, silver_clean_path, gold_raw_path, gold_clean_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    silver_raw = pd.read_csv(silver_raw_path, low_memory=False, na_filter=False)
    silver_cln = pd.read_csv(silver_clean_path, low_memory=False, na_filter=False)
    gold_raw   = pd.read_csv(gold_raw_path, low_memory=False, na_filter=False)
    gold_cln   = pd.read_csv(gold_clean_path, low_memory=False, na_filter=False)

    # RAW model: train on raw silver, test on raw gold
    pr_raw = train_predict(silver_raw, gold_raw)
    out_raw = data / "preds_tfidf_raw_FROZEN.csv"
    pr_raw.to_csv(out_raw, index=False, encoding="utf-8-sig")

    # CLEAN model: train on clean silver, test on clean gold
    pr_cln = train_predict(silver_cln, gold_cln)
    out_cln = data / "preds_tfidf_clean_FROZEN.csv"
    pr_cln.to_csv(out_cln, index=False, encoding="utf-8-sig")

    # alignment sanity
    s1 = set(pr_raw["doc_id"].astype(str).tolist())
    s2 = set(pr_cln["doc_id"].astype(str).tolist())
    if s1 != s2:
        raise RuntimeError(f"doc_id set mismatch between raw and clean preds: raw={len(s1)} clean={len(s2)}")
    mism = int((pr_raw.sort_values("doc_id")["pred_label"].to_numpy() != pr_cln.sort_values("doc_id")["pred_label"].to_numpy()).sum())

    summ = {
        "raw": summarize(pr_raw),
        "clean": summarize(pr_cln),
        "pred_label_mismatches_raw_vs_clean": mism,
        "files": {"raw": str(out_raw), "clean": str(out_cln)},
    }
    (data / "tfidf_preds_frozen_dual_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print("Wrote:", out_raw)
    print("Wrote:", out_cln)
    print("Pred mismatches raw vs clean:", mism)
    print("Summary:", data / "tfidf_preds_frozen_dual_summary.json")

if __name__ == "__main__":
    main()
