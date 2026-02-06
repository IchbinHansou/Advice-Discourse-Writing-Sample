from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# =========================
# Paths (FROZEN inputs)
# =========================
ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

# Use FROZEN splits as the only source of truth
SILVER_RAWNORM = DATA / "silver_train_raw_norm_FROZEN.csv"
SILVER_CLEAN   = DATA / "silver_train_clean_v1_FROZEN.csv"

GOLD_RAWNORM   = DATA / "gold_raw_norm_FROZEN.csv"
GOLD_CLEAN     = DATA / "gold_clean_v1_FROZEN.csv"

# Outputs (tagged)
OUT_JSON = DATA / "results_tfidf_dual_FROZEN.json"
OUT_TXT  = DATA / "results_tfidf_dual_FROZEN.txt"


# =========================
# Marker audit (optional)
# =========================
MARKERS = {
    "aita_acronym": r"\b(?:AITA|WIBTA|AITAH)\b",
    "verdicts": r"\b(?:YTA|NTA|ESH|NAH)\b",
    "tldr_edit_update": r"\b(?:TL;?DR|EDIT|UPDATE)\b",
    # covers both [25F] and plain 25f / 25 f / 25f,
    # because you said you saw many 25f not removed before
    "age_gender_any": r"(?:\[\s*(?:1[3-9]|[2-7]\d|80)\s*[MF]\s*\]|(?:1[3-9]|[2-7]\d|80)\s*[MF]\b)",
}


def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{where}] Missing columns: {missing}. Found={list(df.columns)[:30]} ...")


def load_csv(path: Path, where: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, na_filter=False)
    _require_cols(df, ["doc_id", "text"], where)
    # label column can be either "silver_label"/"y" or "gold_label"
    return df


def subset_gold_binary(gold: pd.DataFrame) -> pd.DataFrame:
    _require_cols(gold, ["gold_label"], "gold")
    out = gold[gold["gold_label"].isin(["ADVICE", "STORY"])].copy()
    return out


def marker_rates(text_series: pd.Series) -> dict[str, float]:
    txt = text_series.astype(str)
    rates = {}
    for k, pat in MARKERS.items():
        # IMPORTANT: na=False to avoid NaN propagation
        rates[k] = float(txt.str.contains(pat, case=False, regex=True, na=False).mean())
    return rates


def run_one(train_path: Path, gold_path: Path, tag: str) -> dict:
    train = load_csv(train_path, where=f"train:{tag}")
    gold = load_csv(gold_path, where=f"gold:{tag}")

    # infer label col name in silver
    if "silver_label" in train.columns:
        y_train = train["silver_label"].astype(str)
    elif "y" in train.columns:
        y_train = train["y"].astype(str)
    else:
        raise RuntimeError(f"[train:{tag}] expected label col silver_label or y, got cols={list(train.columns)[:30]}")

    X_train = train["text"].astype(str)

    # gold binary subset
    _require_cols(gold, ["gold_label"], f"gold:{tag}")
    gold_bin = subset_gold_binary(gold)
    X_test = gold_bin["text"].astype(str)
    y_test = gold_bin["gold_label"].astype(str)

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.95,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    macro = float(f1_score(y_test, pred, average="macro"))
    report = classification_report(y_test, pred, digits=4)
    cm = confusion_matrix(y_test, pred, labels=["ADVICE", "STORY"]).tolist()

    return {
        "tag": tag,
        "train_path": str(train_path),
        "gold_path": str(gold_path),
        "train_n": int(len(train)),
        "test_n_binary": int(len(gold_bin)),
        "label_counts_test": gold_bin["gold_label"].value_counts().to_dict(),
        "macro_f1": macro,
        "confusion_matrix(labels=['ADVICE','STORY'])": cm,
        "report": report,
        "marker_rates_test_text": marker_rates(gold_bin["text"]),
    }


def main():
    r_raw = run_one(SILVER_RAWNORM, GOLD_RAWNORM, tag="RAW_NORM_FROZEN")
    r_cln = run_one(SILVER_CLEAN,   GOLD_CLEAN,   tag="CLEAN_V1_FROZEN")

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "TFIDF(word 1-3) + LogisticRegression",
        "results": {
            "RAW_NORM_FROZEN": {k: v for k, v in r_raw.items() if k != "report"},
            "CLEAN_V1_FROZEN": {k: v for k, v in r_cln.items() if k != "report"},
        },
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    txt = []
    txt.append("=== RAW_NORM_FROZEN ===\n" + r_raw["report"] + f"\nmacro_f1={r_raw['macro_f1']:.4f}\n")
    txt.append("confusion_matrix(['ADVICE','STORY'])=" + str(r_raw["confusion_matrix(labels=['ADVICE','STORY'])"]) + "\n")
    txt.append("marker_rates_test_text=" + json.dumps(r_raw["marker_rates_test_text"], ensure_ascii=False) + "\n")

    txt.append("\n=== CLEAN_V1_FROZEN ===\n" + r_cln["report"] + f"\nmacro_f1={r_cln['macro_f1']:.4f}\n")
    txt.append("confusion_matrix(['ADVICE','STORY'])=" + str(r_cln["confusion_matrix(labels=['ADVICE','STORY'])"]) + "\n")
    txt.append("marker_rates_test_text=" + json.dumps(r_cln["marker_rates_test_text"], ensure_ascii=False) + "\n")

    OUT_TXT.write_text("\n".join(txt), encoding="utf-8")

    print("\n".join(txt))
    print("Wrote:", OUT_JSON)
    print("Wrote:", OUT_TXT)


if __name__ == "__main__":
    main()