from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS_BINARY = ["ADVICE", "STORY"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().upper()


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_WS_RE = re.compile(r"\s+")


def simple_tokens(text: str) -> List[str]:
    text = text.lower()
    return _WORD_RE.findall(text)


def safe_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def last_segment(text: str, *, max_tokens: int = 200, max_chars: int = 1200) -> str:
    t = safe_text(text)
    if len(t) > max_chars:
        t = t[-max_chars:]
    toks = simple_tokens(t)
    if len(toks) > max_tokens:
        toks = toks[-max_tokens:]
        t = " ".join(toks)
    return t.lower()


def compile_patterns() -> List[re.Pattern]:
    pats = [
        r"\bany\s+advice\b",
        r"\bany\s+suggestions\b",
        r"\bneed\s+advice\b",
        r"\bneed\s+help\b",
        r"\bplease\s+help\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bhow\s+do\s+i\b",
        r"\bshould\s+i\b",
        r"\bcould\s+you\b",
        r"\bcan\s+you\b",
        r"\bwhat\s+do\s+you\s+think\b",
        r"\bdo\s+you\s+think\b",
        r"\blooking\s+for\s+advice\b",
        r"\blooking\s+for\s+help\b",
    ]
    return [re.compile(p, flags=re.IGNORECASE) for p in pats]


REQ_PATS = compile_patterns()

MODALS = {
    "should", "could", "would", "might", "may", "can", "cannot", "can't",
    "must", "need", "needs", "needed", "have", "has", "had",
}
HEDGES = {
    "maybe", "perhaps", "probably", "possibly", "kinda", "sorta",
    "somewhat", "guess",
}
HEDGE_PHRASES = [
    re.compile(r"\bi\s+think\b", re.I),
    re.compile(r"\bi\s+feel\b", re.I),
    re.compile(r"\bi\s+guess\b", re.I),
]


def count_regex(pats: List[re.Pattern], text: str) -> int:
    t = safe_text(text)
    return int(sum(len(p.findall(t)) for p in pats))


def extract_features_one(text: str) -> Dict[str, float]:
    t = safe_text(text)
    t_norm = _WS_RE.sub(" ", t).strip()
    toks = simple_tokens(t_norm)
    n_tok = len(toks)
    n_char = len(t_norm)

    # --- q_ group ---
    q_marks = t_norm.count("?")
    q_ratio = q_marks / max(1, n_tok)

    # --- req_ group ---
    req_count = count_regex(REQ_PATS, t_norm)
    has_advice_word = 1.0 if re.search(r"\badvice\b", t_norm, flags=re.I) else 0.0
    has_help_word = 1.0 if re.search(r"\bhelp\b", t_norm, flags=re.I) else 0.0

    # --- mod_ group ---
    modal_count = sum(1 for w in toks if w in MODALS)
    hedge_count = sum(1 for w in toks if w in HEDGES) + sum(1 for p in HEDGE_PHRASES if p.search(t_norm))

    modal_ratio = modal_count / max(1, n_tok)
    hedge_ratio = hedge_count / max(1, n_tok)

    # --- end_ group (ending segment cues) ---
    end = last_segment(t_norm)
    end_q = 1.0 if "?" in end else 0.0
    end_req = float(count_regex(REQ_PATS, end))
    end_modal = float(sum(1 for w in simple_tokens(end) if w in MODALS))

    # --- optional basic length covariates (len_) ---
    n_par = max(1, t_norm.count("\n\n") + 1)
    avg_tok_per_par = n_tok / max(1, n_par)

    return {
        "q_qmarks": float(q_marks),
        "q_qratio": float(q_ratio),

        "req_count": float(req_count),
        "req_has_advice": float(has_advice_word),
        "req_has_help": float(has_help_word),

        "mod_modal_count": float(modal_count),
        "mod_modal_ratio": float(modal_ratio),
        "mod_hedge_count": float(hedge_count),
        "mod_hedge_ratio": float(hedge_ratio),

        "end_has_qmark": float(end_q),
        "end_req_count": float(end_req),
        "end_modal_count": float(end_modal),

        "len_tok": float(n_tok),
        "len_char": float(n_char),
        "len_paragraphs": float(n_par),
        "len_avg_tok_per_par": float(avg_tok_per_par),
    }


def build_feature_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column '{text_col}'. Columns: {list(df.columns)[:30]}")
    feats = df[text_col].fillna("").astype(str).map(extract_features_one).tolist()
    X = pd.DataFrame(feats)
    X = X.fillna(0.0)
    return X


def train_eval(
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    silver_text_col: str,
    gold_text_col: str,
    name: str,
    seed: int = 1004,
    use_cols: List[str] | None = None,
) -> Dict[str, Any]:
    # normalize labels
    silver = silver.copy()
    gold = gold.copy()
    silver["silver_label"] = silver["silver_label"].map(normalize_label)
    gold["gold_label"] = gold["gold_label"].map(normalize_label)

    silver = silver[silver["silver_label"].isin(LABELS_BINARY)]
    gold = gold[gold["gold_label"].isin(LABELS_BINARY)]

    # build features
    Xtr = build_feature_df(silver, silver_text_col)
    Xte = build_feature_df(gold, gold_text_col)

    if use_cols is None:
        use_cols = list(Xtr.columns)

    Xtr = Xtr[use_cols]
    Xte = Xte[use_cols]

    ytr = silver["silver_label"].tolist()
    yte = gold["gold_label"].tolist()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
                solver="liblinear",
            )),
        ]
    )
    pipe.fit(Xtr.values, ytr)
    ypred = pipe.predict(Xte.values)

    macro_f1 = float(f1_score(yte, ypred, average="macro", labels=LABELS_BINARY))
    rep = classification_report(yte, ypred, labels=LABELS_BINARY, output_dict=True, digits=4, zero_division=0)

    # by domain
    by_domain: Dict[str, Any] = {}
    if "domain" in gold.columns:
        for dom in sorted(gold["domain"].fillna("").astype(str).unique().tolist()):
            sub = gold[gold["domain"].astype(str) == dom]
            if len(sub) == 0:
                continue
            Xsub = build_feature_df(sub, gold_text_col)[use_cols].values
            ysub = sub["gold_label"].tolist()
            ysub_pred = pipe.predict(Xsub)
            mf1 = float(f1_score(ysub, ysub_pred, average="macro", labels=LABELS_BINARY))
            by_domain[dom] = {"n": int(len(sub)), "macro_f1": mf1}

    # weights (interpretability)
    clf = pipe.named_steps["clf"]
    feats = use_cols
    coef = clf.coef_[0]  # binary
    classes = list(clf.classes_)  # order is important
    # We'll expose both sides explicitly:
    # positive weights toward classes[1], negative toward classes[0]
    pos_lab = classes[1]
    neg_lab = classes[0]

    w = pd.DataFrame({"feature": feats, "weight_toward_pos": coef})
    w = w.sort_values("weight_toward_pos", ascending=False)

    top_pos = w.head(25).to_dict(orient="records")
    top_neg = w.tail(25).sort_values("weight_toward_pos", ascending=True).to_dict(orient="records")

    return {
        "name": name,
        "config": {"model": "interpretable-features+LogReg", "seed": int(seed), "n_features": int(len(use_cols))},
        "overall": {"n": int(len(gold)), "macro_f1": macro_f1, "report": rep},
        "by_domain": by_domain,
        "weights": {
            "toward_positive_class": pos_lab,
            "toward_negative_class": neg_lab,
            "top_positive": top_pos,
            "top_negative": top_neg,
        },
        "feature_columns": use_cols,
    }


def ablation_runs(
    base_res: Dict[str, Any],
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    silver_text_col: str,
    gold_text_col: str,
) -> Dict[str, Any]:
    cols = base_res["feature_columns"]

    groups = {
        "req": [c for c in cols if c.startswith("req_")],
        "q": [c for c in cols if c.startswith("q_")],
        "mod": [c for c in cols if c.startswith("mod_")],
        "end": [c for c in cols if c.startswith("end_")],
        "len": [c for c in cols if c.startswith("len_")],
    }

    base_f1 = base_res["overall"]["macro_f1"]
    out: Dict[str, Any] = {"base_macro_f1": base_f1, "ablations": {}}

    for g, gcols in groups.items():
        if not gcols:
            continue
        kept = [c for c in cols if c not in set(gcols)]
        res = train_eval(
            silver,
            gold,
            silver_text_col=silver_text_col,
            gold_text_col=gold_text_col,
            name=f"{base_res['name']}_no_{g}",
            use_cols=kept,
        )
        out["ablations"][g] = {
            "removed": gcols,
            "macro_f1": res["overall"]["macro_f1"],
            "delta": float(res["overall"]["macro_f1"] - base_f1),
        }
    return out


def md_table_by_domain(results: Dict[str, Any]) -> str:
    # no tabulate dependency
    rows = []
    for dom, d in results.get("by_domain", {}).items():
        rows.append((dom, d["n"], d["macro_f1"]))
    rows.sort(key=lambda x: x[0])

    lines = []
    lines.append("| domain | n | macro_f1 |")
    lines.append("|---|---:|---:|")
    for dom, n, f1 in rows:
        lines.append(f"| {dom} | {n} | {f1:.6f} |")
    return "\n".join(lines) + "\n"


def main():
    root = repo_root()
    data = root / "data_new"
    print("ROOT:", root)
    print("DATA:", data)

    silver_raw = pd.read_csv(data / "silver_train_raw_norm_FROZEN.csv", low_memory=False)
    silver_clean = pd.read_csv(data / "silver_train_clean_v1_FROZEN.csv", low_memory=False)

    gold_raw = pd.read_csv(data / "gold_raw_norm_FROZEN.csv", low_memory=False)
    gold_clean_full = pd.read_csv(data / "gold_clean_v1_FROZEN.csv", low_memory=False)

    # build aligned clean binary gold
    gold_clean = gold_clean_full.copy()
    gold_clean["gold_label"] = gold_clean["gold_label"].map(normalize_label)
    gold_clean = gold_clean[gold_clean["gold_label"].isin(LABELS_BINARY)]
    # ensure clean text col exists
    if "text_clean_v1" not in gold_clean.columns:
        raise KeyError("gold_clean_v1_FROZEN.csv missing 'text_clean_v1'")
    # align by doc_id intersection with gold_raw
    common = sorted(set(gold_raw["doc_id"].astype(str)) & set(gold_clean["doc_id"].astype(str)))
    gold_raw = gold_raw[gold_raw["doc_id"].astype(str).isin(common)].copy()
    gold_clean = gold_clean[gold_clean["doc_id"].astype(str).isin(common)].copy()
    gold_raw = gold_raw.sort_values("doc_id")
    gold_clean = gold_clean.sort_values("doc_id")

    print("gold_raw rows:", len(gold_raw))
    print("gold_clean rows:", len(gold_clean))

    # RAW run
    res_raw = train_eval(
        silver_raw,
        gold_raw,
        silver_text_col="text",
        gold_text_col="text",
        name="features_raw_aligned",
    )

    # CLEAN run
    res_clean = train_eval(
        silver_clean,
        gold_clean,
        silver_text_col="text_clean_v1",
        gold_text_col="text_clean_v1",
        name="features_clean_v1_aligned",
    )

    out = {"features_raw_aligned": res_raw, "features_clean_v1_aligned": res_clean}

    (data / "results_features_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", data / "results_features_v1.json")

    # weights CSV (both conditions)
    weights_rows = []
    for k, res in out.items():
        pos = res["weights"]["toward_positive_class"]
        neg = res["weights"]["toward_negative_class"]
        for r in res["weights"]["top_positive"]:
            weights_rows.append({"run": k, "direction": f"toward_{pos}", **r})
        for r in res["weights"]["top_negative"]:
            weights_rows.append({"run": k, "direction": f"toward_{neg}", **r})
    pd.DataFrame(weights_rows).to_csv(data / "feature_weights_v1.csv", index=False, encoding="utf-8")
    print("Saved:", data / "feature_weights_v1.csv")

    # ablations (do for CLEAN by default）
    ab = ablation_runs(
        res_clean,
        silver_clean,
        gold_clean,
        silver_text_col="text_clean_v1",
        gold_text_col="text_clean_v1",
    )
    (data / "ablation_features_v1.json").write_text(json.dumps(ab, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", data / "ablation_features_v1.json")

    # markdown tables
    md_raw = md_table_by_domain(res_raw)
    md_clean = md_table_by_domain(res_clean)
    md = []
    md.append("## features_raw_aligned\n")
    md.append(md_raw)
    md.append("\n## features_clean_v1_aligned\n")
    md.append(md_clean)
    (data / "table_features_by_domain.md").write_text("".join(md), encoding="utf-8")
    print("Saved:", data / "table_features_by_domain.md")

    print("\n=== SUMMARY ===")
    print(f"RAW   macroF1={res_raw['overall']['macro_f1']:.4f}  n={res_raw['overall']['n']}")
    print(f"CLEAN macroF1={res_clean['overall']['macro_f1']:.4f}  n={res_clean['overall']['n']}")


if __name__ == "__main__":
    main()