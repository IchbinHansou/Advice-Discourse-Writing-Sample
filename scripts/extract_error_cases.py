from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["ADVICE", "STORY"]

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def normalize_label(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().upper()

# -----------------------------
# Feature extraction utilities
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_WS_RE = re.compile(r"\s+")

def simple_tokens(text: str):
    return _WORD_RE.findall(text.lower())

def safe_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def last_segment(text: str, max_tokens: int = 200, max_chars: int = 1200) -> str:
    t = safe_text(text)
    if len(t) > max_chars:
        t = t[-max_chars:]
    toks = simple_tokens(t)
    if len(toks) > max_tokens:
        toks = toks[-max_tokens:]
        t = " ".join(toks)
    return t.lower()

REQ_PATS = [re.compile(p, re.I) for p in [
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
]]

MODALS = {"should", "could", "would", "might", "may", "can", "cannot", "can't", "must",
          "need", "needs", "needed", "have", "has", "had"}

def count_regex(pats, text: str) -> int:
    return int(sum(len(p.findall(text)) for p in pats))

def extract_features_one(text: str):
    t = _WS_RE.sub(" ", safe_text(text)).strip()
    toks = simple_tokens(t)
    n_tok = len(toks)
    q_marks = t.count("?")
    req_count = count_regex(REQ_PATS, t)
    end = last_segment(t)
    return {
        "req_count": float(req_count),
        "req_has_advice": float(bool(re.search(r"\badvice\b", t, re.I))),
        "q_qmarks": float(q_marks),
        "q_qratio": float(q_marks / max(1, n_tok)),
        "end_has_qmark": float("?" in end),
        "end_req_count": float(count_regex(REQ_PATS, end)),
        "end_modal_count": float(sum(1 for w in simple_tokens(end) if w in MODALS)),
        "len_tok": float(n_tok),
        "len_char": float(len(t)),
    }

def build_feature_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    feats = df[text_col].fillna("").astype(str).map(extract_features_one).tolist()
    return pd.DataFrame(feats).fillna(0.0)

# -----------------------------
# AITA error typology (optional)
# -----------------------------
RE_REQ_SIMPLE = re.compile(
    r"\b(any advice|any suggestions|need advice|need help|please help|what should i do|"
    r"looking for advice|looking for help|should i|could you|can you)\b",
    re.I,
)
RE_VERDICT = re.compile(r"\b(aita|asshole|yta|nta|esh|nah|info)\b", re.I)
RE_VALIDATION = re.compile(r"\b(am i wrong|was i wrong|overreact|did i mess up|my fault)\b", re.I)

def categorize_aita_error(text: str) -> str:
    t = (text or "").lower()
    ttail = t[-1200:]
    has_req = bool(RE_REQ_SIMPLE.search(t))
    has_req_end = bool(RE_REQ_SIMPLE.search(ttail))
    has_verdict = bool(RE_VERDICT.search(t))
    has_validation = bool(RE_VALIDATION.search(t))
    qmarks = t.count("?")

    if has_verdict and (has_req or has_req_end):
        return "D_mixed_verdict+advice"
    if has_verdict or has_validation:
        return "A_judgment_or_validation"
    if qmarks >= 3 and not has_req:
        return "B_question_heavy_rhetorical"
    if (not has_req) and (qmarks >= 1):
        return "C_implicit_help_seeking"
    return "E_other"

def main():
    root = repo_root()
    data = root / "data_new"

    silver = pd.read_csv(data / "silver_train_clean_v1_FROZEN.csv", low_memory=False)
    gold = pd.read_csv(data / "gold_clean_v1_FROZEN.csv", low_memory=False)

    silver["silver_label"] = silver["silver_label"].map(normalize_label)
    gold["gold_label"] = gold["gold_label"].map(normalize_label)

    gold = gold[gold["gold_label"].isin(LABELS)].copy()

    # align doc_id with gold_binary if you want strict consistency; optional
    gb_path = data / "gold_raw_norm_FROZEN.csv"
    if gb_path.exists():
        gb = pd.read_csv(gb_path, low_memory=False)
        common = set(gb["doc_id"].astype(str)) & set(gold["doc_id"].astype(str))
        gold = gold[gold["doc_id"].astype(str).isin(common)].copy()

    # Train on silver, test on gold
    Xtr = build_feature_df(silver, "text_clean_v1")
    ytr = silver["silver_label"].tolist()

    Xte = build_feature_df(gold, "text_clean_v1")
    yte = gold["gold_label"].tolist()  # unused here but kept for clarity

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=1004
        )),
    ])
    pipe.fit(Xtr.values, ytr)
    pred = pipe.predict(Xte.values)

    # --- Save predictions for all gold ---
    out = gold[["doc_id", "domain", "gold_label", "text_clean_v1"]].copy()
    out["pred_label"] = pred
    out["is_error"] = out["gold_label"] != out["pred_label"]
    out["text_tail"] = out["text_clean_v1"].astype(str).str.replace("\n", " ", regex=False).str[-500:]

    save_all = data / "preds_features_clean_v1_aligned.csv"
    out.to_csv(save_all, index=False, encoding="utf-8")
    print("Saved:", save_all)

    # --- Focus on AITA errors ---
    aita_mask = out["domain"].astype(str) == "aita"
    aita_total = int(aita_mask.sum())
    aita_err = out[aita_mask & out["is_error"]].copy()

    # Optional: add auto-typology labels
    aita_err["category"] = aita_err["text_clean_v1"].astype(str).map(categorize_aita_error)

    # Save AITA error cases
    save_aita = data / "error_cases_aita.csv"
    aita_err.to_csv(save_aita, index=False, encoding="utf-8")
    print("Saved:", save_aita)
    print(f"AITA errors: {len(aita_err)}/{aita_total}")

    # Save typology counts (handy for paper)
    counts = aita_err["category"].value_counts().rename_axis("category").reset_index(name="n")
    save_counts = data / "aita_error_typology_counts.csv"
    counts.to_csv(save_counts, index=False, encoding="utf-8")
    print("Saved:", save_counts)
    print("=== AITA typology counts ===")
    print(counts.to_string(index=False))

if __name__ == "__main__":
    main()