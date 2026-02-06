from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


LABELS = ["ADVICE", "STORY"]  # binary eval; MIXED excluded


@dataclass
class Config:
    model: str = "TFIDF(word 1-3)+LogReg"
    ngram_range: Tuple[int, int] = (1, 3)
    min_df: int = 3
    max_df: float = 0.95
    stop_words: str | None = "english"
    max_iter: int = 2000
    C: float = 1.0
    seed: int = 1004


def get_repo_root() -> Path:
    # scripts/xxx.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{name}] Missing columns: {missing}. Found={list(df.columns)[:40]}...")


def _to_upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper()


def _subset_gold_binary(gold: pd.DataFrame) -> pd.DataFrame:
    g = gold.copy()
    g["gold_label"] = _to_upper(g["gold_label"])
    g = g[g["gold_label"].isin(LABELS)].copy()
    return g


def train_and_predict(
    train_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    cfg: Config,
) -> Dict:
    # texts + labels
    X_train = train_df["text"].astype(str).tolist()
    y_train = _to_upper(train_df["silver_label"]).tolist()

    # ✅ 关键：reset_index(drop=True)，保证后面按 sub.index 索引 pred 不越界
    gold_bin = _subset_gold_binary(gold_df).reset_index(drop=True)

    X_test = gold_bin["text"].astype(str).tolist()
    y_test = gold_bin["gold_label"].tolist()

    # vectorizer (ALIGN WITH BASELINE)
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=cfg.stop_words,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # classifier (ALIGN WITH BASELINE)
    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        random_state=cfg.seed,
    )
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)  # numpy array, length = len(gold_bin)

    macro = float(f1_score(y_test, pred, labels=LABELS, average="macro"))
    rep = classification_report(y_test, pred, labels=LABELS, digits=4)

    # per-domain
    by_domain = {}
    if "domain" in gold_bin.columns:
        for dom in sorted(gold_bin["domain"].astype(str).unique().tolist()):
            sub = gold_bin[gold_bin["domain"].astype(str) == dom].copy()
            if len(sub) == 0:
                continue
            sub_pred = pred[sub.index.to_numpy()]  # ✅ 现在 index 是 0..N-1，不会越界
            md = float(f1_score(sub["gold_label"].tolist(), sub_pred, labels=LABELS, average="macro"))
            counts = sub["gold_label"].value_counts().to_dict()
            by_domain[dom] = {"n": int(len(sub)), "macro_f1": md, "counts": counts}

    return {
        "macro_f1": macro,
        "report_text": rep,
        "y_test": y_test,
        "pred": pred.tolist(),
        "gold_bin": gold_bin,
        "by_domain": by_domain,
        "label_counts_test": gold_bin["gold_label"].value_counts().to_dict(),
    }


def export_preds(gold_bin: pd.DataFrame, pred_list, out_csv: Path):
    keep_cols = [c for c in ["doc_id", "domain", "gold_label"] if c in gold_bin.columns]
    out = gold_bin[keep_cols].copy()
    out["pred"] = pred_list
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    root = get_repo_root()
    data = root / "data_new"

    # ===== Inputs (FROZEN) =====
    SILVER_RAW = data / "silver_train_raw_norm_FROZEN.csv"
    GOLD_RAW   = data / "gold_raw_norm_FROZEN.csv"

    SILVER_CLN = data / "silver_train_clean_v1_FROZEN.csv"
    GOLD_CLN   = data / "gold_clean_v1_FROZEN.csv"

    for p in [SILVER_RAW, GOLD_RAW, SILVER_CLN, GOLD_CLN]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    cfg = Config()

    silver_raw = pd.read_csv(SILVER_RAW, low_memory=False)
    gold_raw = pd.read_csv(GOLD_RAW, low_memory=False)

    silver_cln = pd.read_csv(SILVER_CLN, low_memory=False)
    gold_cln = pd.read_csv(GOLD_CLN, low_memory=False)

    _require_cols(silver_raw, ["text", "silver_label"], "silver_raw_norm_frozen")
    _require_cols(gold_raw, ["text", "gold_label"], "gold_raw_norm_frozen")
    _require_cols(silver_cln, ["text", "silver_label"], "silver_clean_v1_frozen")
    _require_cols(gold_cln, ["text", "gold_label"], "gold_clean_v1_frozen")

    # ===== Alignment check: raw gold vs clean gold on binary subset =====
    gA = _subset_gold_binary(gold_raw)[["doc_id", "gold_label"]].copy()
    gB = _subset_gold_binary(gold_cln)[["doc_id", "gold_label"]].copy()
    gA["doc_id"] = gA["doc_id"].astype(str)
    gB["doc_id"] = gB["doc_id"].astype(str)

    same_ids = gA["doc_id"].tolist() == gB["doc_id"].tolist()
    same_labels = gA["gold_label"].tolist() == gB["gold_label"].tolist()
    if (len(gA) != len(gB)) or (set(gA["doc_id"]) != set(gB["doc_id"])) or (not same_labels):
        raise RuntimeError(
            "RAW vs CLEAN gold binary misaligned.\n"
            f"  n_raw={len(gA)} n_clean={len(gB)}\n"
            f"  same_id_order={same_ids} same_labels={same_labels}\n"
            "Fix before generating paired preds/plots."
        )

    # ===== Train + predict (RAW_NORM) =====
    raw_res = train_and_predict(silver_raw, gold_raw, cfg)
    OUT_RAW_PREDS = data / "preds_tfidf_raw_norm_FROZEN_gold.csv"
    export_preds(raw_res["gold_bin"], raw_res["pred"], OUT_RAW_PREDS)

    # ===== Train + predict (CLEAN_V1) =====
    cln_res = train_and_predict(silver_cln, gold_cln, cfg)
    OUT_CLN_PREDS = data / "preds_tfidf_clean_v1_FROZEN_gold.csv"
    export_preds(cln_res["gold_bin"], cln_res["pred"], OUT_CLN_PREDS)

    # ===== Reports =====
    OUT_RAW_REPORT = data / "tfidf_raw_norm_FROZEN_report.txt"
    OUT_CLN_REPORT = data / "tfidf_clean_v1_FROZEN_report.txt"
    OUT_JSON = data / "tfidf_preds_summary_FROZEN.json"

    OUT_RAW_REPORT.write_text(
        "=== RAW_NORM_FROZEN ===\n"
        f"Model: {cfg.model}\n"
        f"Config: {json.dumps(cfg.__dict__)}\n"
        f"Overall macro_f1: {raw_res['macro_f1']:.6f}  n={len(raw_res['gold_bin'])}  label_counts={raw_res['label_counts_test']}\n\n"
        + raw_res["report_text"]
        + "\n\nBy domain:\n"
        + "\n".join([f"- {d}: {v}" for d, v in raw_res["by_domain"].items()])
        + "\n",
        encoding="utf-8",
    )

    OUT_CLN_REPORT.write_text(
        "=== CLEAN_V1_FROZEN ===\n"
        f"Model: {cfg.model}\n"
        f"Config: {json.dumps(cfg.__dict__)}\n"
        f"Overall macro_f1: {cln_res['macro_f1']:.6f}  n={len(cln_res['gold_bin'])}  label_counts={cln_res['label_counts_test']}\n\n"
        + cln_res["report_text"]
        + "\n\nBy domain:\n"
        + "\n".join([f"- {d}: {v}" for d, v in cln_res["by_domain"].items()])
        + "\n",
        encoding="utf-8",
    )

    OUT_JSON.write_text(
        json.dumps(
            {
                "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                "model": cfg.model,
                "raw_norm_frozen": {
                    "pred_file": str(OUT_RAW_PREDS),
                    "macro_f1": raw_res["macro_f1"],
                    "by_domain": raw_res["by_domain"],
                    "n": int(len(raw_res["gold_bin"])),
                    "label_counts_test": raw_res["label_counts_test"],
                },
                "clean_v1_frozen": {
                    "pred_file": str(OUT_CLN_PREDS),
                    "macro_f1": cln_res["macro_f1"],
                    "by_domain": cln_res["by_domain"],
                    "n": int(len(cln_res["gold_bin"])),
                    "label_counts_test": cln_res["label_counts_test"],
                },
                "alignment_check": {
                    "binary_n": int(len(gA)),
                    "same_doc_id_order": bool(same_ids),
                    "same_gold_labels": bool(same_labels),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Wrote preds:")
    print(" ", OUT_RAW_PREDS)
    print(" ", OUT_CLN_PREDS)
    print("Wrote reports:")
    print(" ", OUT_RAW_REPORT)
    print(" ", OUT_CLN_REPORT)
    print("Wrote json:")
    print(" ", OUT_JSON)


if __name__ == "__main__":
    main()
