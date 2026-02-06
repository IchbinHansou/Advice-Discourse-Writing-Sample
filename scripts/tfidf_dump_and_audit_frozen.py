from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline


LABELS = ["ADVICE", "STORY"]  # binary eval; MIXED excluded


def project_root() -> Path:
    # .../Saarland Writing Sample/scripts/xxx.py -> root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def ensure_upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def train_pipeline(seed: int = 1004) -> Pipeline:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=3,
        max_features=200_000,
        lowercase=True,
    )
    clf = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=seed,
        class_weight="balanced",   # align with your baseline scripts
        solver="liblinear",        # stable for binary
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def top_features(pipeline: Pipeline, topk: int = 50) -> dict:
    vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    feats = vec.get_feature_names_out()
    classes = list(clf.classes_)  # e.g., ['ADVICE','STORY']
    coef = clf.coef_[0]           # toward classes[1] in binary

    pos_class = classes[1]
    neg_class = classes[0]

    idx_pos = np.argsort(coef)[::-1][:topk]
    idx_neg = np.argsort(coef)[:topk]

    def pack(idxs, toward):
        return [{"feature": str(feats[i]), "weight": float(coef[i]), "toward": toward} for i in idxs]

    return {
        "classes": classes,
        "positive_class": pos_class,
        "negative_class": neg_class,
        "top_toward_positive": pack(idx_pos, pos_class),
        "top_toward_negative": pack(idx_neg, neg_class),
    }


def artifact_audit_on_features(features: list[str], regex_path: Path) -> dict:
    """
    Heuristic: count how many top features contain obvious artifact tokens.
    """
    txt = regex_path.read_text(encoding="utf-8", errors="replace").lower()

    tokens = []
    for w in [
        "aita", "wibta", "aitah", "amItheasshole".lower(),
        "yta", "nta", "esh", "nah", "info",
        "throwaway", "tldr", "tl;dr", "edit", "update"
    ]:
        if w in txt:
            tokens.append(w.replace("tl;dr", "tldr"))
    tokens = sorted(set(tokens))

    feats_l = [f.lower() for f in features]
    hit = {tok: 0 for tok in tokens}

    def norm(x: str) -> str:
        return x.replace(";", "").replace(":", "")

    any_hit = 0
    for f in feats_l:
        f2 = norm(f)
        matched = False
        for tok in tokens:
            if tok in f2:
                hit[tok] += 1
                matched = True
        if matched:
            any_hit += 1

    total = len(features)
    return {
        "tokens_checked": tokens,
        "total_features_checked": total,
        "features_with_any_artifact_token": any_hit,
        "ratio_with_any_artifact_token": (any_hit / total) if total else 0.0,
        "hits_by_token": hit,
    }


def eval_one(
    *,
    condition: str,                 # "raw" or "clean"
    silver_path: Path,
    gold_path: Path,
    outdir: Path,
    regex_path: Path,
    seed: int = 1004,
) -> dict:
    silver = pd.read_csv(silver_path, low_memory=False)
    gold = pd.read_csv(gold_path, low_memory=False)

    # required columns sanity
    for c in ["text", "silver_label"]:
        if c not in silver.columns:
            raise ValueError(f"Silver missing column: {c} in {silver_path.name}")
    for c in ["doc_id", "domain", "text", "gold_label"]:
        if c not in gold.columns:
            raise ValueError(f"Gold missing column: {c} in {gold_path.name}")

    # binary only
    gold = gold.copy()
    gold["gold_label"] = ensure_upper(gold["gold_label"])
    gold_bin = gold[gold["gold_label"].isin(LABELS)].copy()

    # train
    X_train = silver["text"].astype(str)
    y_train = ensure_upper(silver["silver_label"])

    pipe = train_pipeline(seed=seed)
    pipe.fit(X_train, y_train)

    # eval
    X_gold = gold_bin["text"].astype(str)
    y_true = gold_bin["gold_label"].to_numpy()
    y_pred = pipe.predict(X_gold)

    macro = float(f1_score(y_true, y_pred, labels=LABELS, average="macro"))
    rep = classification_report(y_true, y_pred, labels=LABELS, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    # per-domain macro-F1
    by_domain = {}
    for dom in sorted(gold_bin["domain"].astype(str).unique().tolist()):
        sub = gold_bin[gold_bin["domain"].astype(str) == dom].copy()
        yp = pipe.predict(sub["text"].astype(str))
        md = float(f1_score(sub["gold_label"].to_numpy(), yp, labels=LABELS, average="macro"))
        by_domain[dom] = {
            "n": int(len(sub)),
            "macro_f1": md,
            "label_counts": sub["gold_label"].value_counts().to_dict(),
        }

    # write preds (minimal columns to match your R)
    pred_path = outdir / f"preds_tfidf_{condition}_FROZEN.csv"
    pred_df = gold_bin[["doc_id", "domain", "gold_label"]].copy()
    pred_df["pred_label"] = y_pred
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    # report
    report_path = outdir / f"tfidf_{condition}_report_FROZEN.txt"
    report_path.write_text(
        f"condition={condition}\n"
        f"silver={silver_path.name}\n"
        f"gold={gold_path.name}\n"
        f"N_gold_binary={len(gold_bin)}\n"
        f"macro_f1={macro:.6f}\n"
        f"confusion_matrix(labels={LABELS})={cm}\n\n"
        + rep
        + "\n",
        encoding="utf-8",
        errors="replace",
    )

    # save model
    model_path = outdir / f"model_tfidf_{condition}_FROZEN.joblib"
    joblib.dump(pipe, model_path)

    # top features + heuristic audit
    tf = top_features(pipe, topk=50)
    top_list = [d["feature"] for d in tf["top_toward_positive"]] + [d["feature"] for d in tf["top_toward_negative"]]
    audit = artifact_audit_on_features(top_list, regex_path)

    feat_json = outdir / f"tfidf_top_features_{condition}_FROZEN.json"
    feat_csv = outdir / f"tfidf_top_features_{condition}_FROZEN.csv"
    feat_json.write_text(json.dumps(tf, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(tf["top_toward_positive"] + tf["top_toward_negative"]).to_csv(feat_csv, index=False, encoding="utf-8")

    return {
        "condition": condition,
        "train": {"silver_file": str(silver_path), "n": int(len(silver))},
        "test": {"gold_file": str(gold_path), "n_binary": int(len(gold_bin))},
        "overall": {"macro_f1": macro, "cm": cm},
        "by_domain": by_domain,
        "files": {
            "preds_csv": str(pred_path),
            "report_txt": str(report_path),
            "model_joblib": str(model_path),
            "top_features_json": str(feat_json),
            "top_features_csv": str(feat_csv),
        },
        "artifact_audit_top100": audit,
    }


def main():
    root = project_root()
    data = root / "data_new"
    spec = root / "spec"

    outdir = data
    outdir.mkdir(parents=True, exist_ok=True)

    # frozen inputs (authoritative)
    silver_raw = data / "silver_train_raw_norm_FROZEN.csv"
    silver_cln = data / "silver_train_clean_v1_FROZEN.csv"
    gold_raw = data / "gold_raw_norm_FROZEN.csv"
    gold_cln = data / "gold_clean_v1_FROZEN.csv"

    regex_path = spec / "regex_v1.txt"

    for p in [silver_raw, silver_cln, gold_raw, gold_cln, regex_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    seed = 1004

    summary = {
        "seed": seed,
        "root": str(root),
        "inputs": {
            "silver_raw": str(silver_raw),
            "silver_clean": str(silver_cln),
            "gold_raw": str(gold_raw),
            "gold_clean": str(gold_cln),
            "regex": str(regex_path),
        },
    }

    summary["raw"] = eval_one(
        condition="raw",
        silver_path=silver_raw,
        gold_path=gold_raw,
        outdir=outdir,
        regex_path=regex_path,
        seed=seed,
    )
    summary["clean"] = eval_one(
        condition="clean",
        silver_path=silver_cln,
        gold_path=gold_cln,
        outdir=outdir,
        regex_path=regex_path,
        seed=seed,
    )

    out_summary = outdir / "tfidf_dump_and_audit_summary_FROZEN.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_summary)


if __name__ == "__main__":
    main()