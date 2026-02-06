from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


LABELS_BINARY = ["ADVICE", "STORY"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these files exist:\n" + "\n".join(str(p) for p in paths)
    )


def normalize_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().upper()


def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False)


def load_gold_raw_excel(p: Path) -> pd.DataFrame:
    # You already have openpyxl via pandas; engine explicit avoids surprises.
    return pd.read_excel(p, engine="openpyxl")


def ensure_text(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    if text_col not in df.columns:
        raise KeyError(f"Missing column '{text_col}'. Columns: {list(df.columns)[:30]}")
    return df[text_col].fillna("").astype(str)


def ensure_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}'. Columns: {list(df.columns)[:30]}")


def prepare_silver(df: pd.DataFrame) -> pd.DataFrame:
    ensure_col(df, "silver_label")
    ensure_col(df, "text")
    df = df.copy()
    df["silver_label"] = df["silver_label"].map(normalize_label)
    df = df[df["silver_label"].isin(LABELS_BINARY)]
    df["text"] = ensure_text(df, "text")
    return df


def prepare_gold(df: pd.DataFrame, label_col: str = "gold_label") -> pd.DataFrame:
    ensure_col(df, label_col)
    ensure_col(df, "text")
    ensure_col(df, "domain")
    df = df.copy()
    df[label_col] = df[label_col].map(normalize_label)
    # main eval excludes MIXED
    df = df[df[label_col].isin(LABELS_BINARY)]
    df["text"] = ensure_text(df, "text")
    df["domain"] = df["domain"].fillna("").astype(str)
    return df


def train_and_eval(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    text_condition: str,
    seed: int = 1004,
    ngram_range=(1, 3),
    min_df=3,
    max_features=200_000,
    C=1.0,
) -> Dict[str, Any]:
    X_train = silver["text"].tolist()
    y_train = silver["silver_label"].tolist()

    vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )
    Xtr = vect.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=2000,
        C=C,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )
    clf.fit(Xtr, y_train)

    X_gold = vect.transform(gold["text"].tolist())
    y_gold = gold["gold_label"].tolist()
    y_pred = clf.predict(X_gold)

    overall_macro_f1 = float(f1_score(y_gold, y_pred, average="macro", labels=LABELS_BINARY))

    overall_report = classification_report(
        y_gold, y_pred, labels=LABELS_BINARY, digits=4, output_dict=True, zero_division=0
    )

    # domain breakdown
    by_domain: Dict[str, Any] = {}
    for dom in sorted(gold["domain"].unique().tolist()):
        sub = gold[gold["domain"] == dom]
        if len(sub) == 0:
            continue
        X_sub = vect.transform(sub["text"].tolist())
        y_sub = sub["gold_label"].tolist()
        y_sub_pred = clf.predict(X_sub)
        macro_f1 = float(f1_score(y_sub, y_sub_pred, average="macro", labels=LABELS_BINARY))
        rep = classification_report(
            y_sub, y_sub_pred, labels=LABELS_BINARY, digits=4, output_dict=True, zero_division=0
        )
        by_domain[dom] = {
            "n": int(len(sub)),
            "macro_f1": macro_f1,
            "report": rep,
            "label_counts": sub["gold_label"].value_counts().to_dict(),
        }

    return {
        "text_condition": text_condition,
        "config": {
            "model": "TFIDF+LogReg",
            "ngram_range": list(ngram_range),
            "min_df": int(min_df),
            "max_features": int(max_features),
            "C": float(C),
            "seed": int(seed),
        },
        "overall": {
            "n": int(len(gold)),
            "macro_f1": overall_macro_f1,
            "report": overall_report,
            "label_counts": gold["gold_label"].value_counts().to_dict(),
        },
        "by_domain": by_domain,
    }


def report_to_text(res: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"=== {res['text_condition'].upper()} ===")
    lines.append(f"Config: {json.dumps(res['config'], ensure_ascii=False)}")
    lines.append(f"Overall: n={res['overall']['n']} macro_f1={res['overall']['macro_f1']:.4f}")
    # pretty print overall per-class
    od = res["overall"]["report"]
    for lab in LABELS_BINARY:
        if lab in od:
            lines.append(
                f"  {lab:6s} P={od[lab]['precision']:.4f} R={od[lab]['recall']:.4f} F1={od[lab]['f1-score']:.4f} support={int(od[lab]['support'])}"
            )

    lines.append("")
    lines.append("By domain (gold, binary only; MIXED excluded):")
    for dom, d in res["by_domain"].items():
        lines.append(f"- {dom}: n={d['n']} macro_f1={d['macro_f1']:.4f} counts={d['label_counts']}")
    return "\n".join(lines)


def main():
    root = repo_root()
    data = root / "data_new"
    print("ROOT:", root)
    print("DATA:", data)

    # inputs
    silver_raw_p = first_existing([data / "silver_train_raw_norm_FROZEN.csv"])
    silver_clean_p = first_existing([data / "silver_train_clean_v1_FROZEN.csv", data / "silver_train_cleaned_v1.csv"])

    gold_raw_p = first_existing([
        data / "gold_raw_norm_FROZEN.csv",
        data / "gold_to_annotate.xlsx",
        data / "gold_to_annotate.csv",
        data / "gold_annotations.xlsx",
        data / "gold_annotations.csv",
    ])

    gold_clean_p = first_existing([
        data / "gold_clean_v1.csv",
        data / "gold_clean_v1_FROZEN.csv",
        data / "gold_annotations_clean_v1.csv",
        data / "gold_to_annotate_V5_final_clean_v1.csv",
    ])

    # load
    silver_raw = prepare_silver(load_csv(silver_raw_p))
    silver_clean = prepare_silver(load_csv(silver_clean_p))

    if gold_raw_p.suffix.lower() in [".xlsx", ".xls"]:
        gold_raw = load_gold_raw_excel(gold_raw_p)
    else:
        gold_raw = load_csv(gold_raw_p)

    gold_raw = prepare_gold(gold_raw, label_col="gold_label")
    gold_clean = prepare_gold(load_csv(gold_clean_p), label_col="gold_label")

    # train+eval
    res_raw = train_and_eval(silver_raw, gold_raw, text_condition="raw")
    res_clean = train_and_eval(silver_clean, gold_clean, text_condition="cleaned_v1")

    out = {
        "raw": res_raw,
        "cleaned_v1": res_clean,
        "notes": {
            "gold_eval": "gold only, binary ADVICE vs STORY; MIXED excluded",
            "domain_breakdown": "macro-F1 computed within each gold domain subset",
        },
    }

    out_json = data / "results_tfidf_dual_by_domain.json"
    out_txt = data / "results_tfidf_dual_by_domain.txt"

    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out_txt.write_text(report_to_text(res_raw) + "\n\n" + report_to_text(res_clean) + "\n", encoding="utf-8")

    print("Wrote:", out_json)
    print("Wrote:", out_txt)


if __name__ == "__main__":
    main()