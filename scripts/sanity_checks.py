# scripts/sanity_checks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# Paths (ONLY use FROZEN)
# =========================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_new"

GOLD_RAW = DATA / "gold_raw_norm_FROZEN.csv"
GOLD_CLN = DATA / "gold_clean_v1_FROZEN.csv"
SILV_RAW = DATA / "silver_train_raw_norm_FROZEN.csv"
SILV_CLN = DATA / "silver_train_clean_v1_FROZEN.csv"

OUT_JSON = DATA / "sanity_report_FROZEN.json"


# =========================
# Expected columns
# =========================
REQ_GOLD_COLS = ["doc_id", "domain", "gold_label", "text"]
REQ_SILV_COLS = ["doc_id", "domain", "silver_label", "text"]


# =========================
# Marker regex (artifact presence)
# 注意：这里的 marker 用于“检测伪影是否仍在文本里”
# 你的 clean_v1 目标是：AITA/裁决/TLDR-EDIT-UPDATE/年龄性别(任意形式)尽量为 0
# =========================
MARKERS = {
    "aita_acronym": re.compile(r"\b(?:AITA|WIBTA|AITAH)\b", re.I),
    "verdicts": re.compile(r"\b(?:YTA|NTA|ESH|NAH)\b", re.I),
    "tldr_edit_update": re.compile(r"\b(?:TL;?DR|EDIT|UPDATE)\b", re.I),
    "throwaway": re.compile(r"\bthrowaway\b", re.I),
    # 两类年龄性别：方括号/圆括号/紧贴形式
    "age_gender_bracket": re.compile(r"[\[\(]\s*(?:1[3-9]|[2-7]\d|80)\s*[MF]\s*[\]\)]", re.I),
    "age_gender_plain": re.compile(r"\b(?:1[3-9]|[2-7]\d|80)\s*[MF]\b", re.I),
}
# 方便汇总一个“任何年龄性别模式”
AGE_ANY = re.compile(r"(" + MARKERS["age_gender_bracket"].pattern + r")|(" + MARKERS["age_gender_plain"].pattern + r")", re.I)

# 你 clean 会写入占位符，这个不是“伪影残留”，而是“清洗成功的证据”
PLACEHOLDERS = {
    "has_age_gender_placeholder": re.compile(r"<AGE_GENDER>", re.I),
}


# =========================
# Helpers
# =========================
def _require_exists(p: Path):
    if not p.exists():
        raise RuntimeError(f"Missing file: {p}")

def _require_cols(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{where}] Missing columns: {missing}. Found={list(df.columns)[:30]}")

def _read_csv(p: Path) -> pd.DataFrame:
    # keep newlines in quoted text; pandas handles it fine
    return pd.read_csv(p, low_memory=False)

def _std_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out

def _norm_label(x: str) -> str:
    s = str(x).strip().upper()
    # 兼容你论文里写 Disclosure
    if s == "DISCLOSURE":
        return "STORY"
    return s

def _marker_rates(text_s: pd.Series) -> Dict[str, float]:
    txt = text_s.fillna("").astype(str)
    rates = {}
    for k, pat in MARKERS.items():
        rates[k] = float(txt.str.contains(pat, regex=True).mean())
    rates["age_gender_any"] = float(txt.str.contains(AGE_ANY, regex=True).mean())
    for k, pat in PLACEHOLDERS.items():
        rates[k] = float(txt.str.contains(pat, regex=True).mean())
    return rates

def _changed_ratio(a: pd.Series, b: pd.Series) -> float:
    aa = a.fillna("").astype(str)
    bb = b.fillna("").astype(str)
    return float((aa != bb).mean())

def _basic_dupes(df: pd.DataFrame, id_col: str = "doc_id") -> int:
    return int(df[id_col].astype(str).duplicated().sum())

def _overlap(a_ids: pd.Series, b_ids: pd.Series) -> int:
    return int(len(set(a_ids.astype(str).tolist()) & set(b_ids.astype(str).tolist())))

def _align_by_docid(a: pd.DataFrame, b: pd.DataFrame, id_col="doc_id") -> Tuple[bool, int]:
    a_ids = a[id_col].astype(str).tolist()
    b_ids = b[id_col].astype(str).tolist()
    if len(a_ids) != len(b_ids):
        return False, abs(len(a_ids) - len(b_ids))
    same = (a_ids == b_ids)
    if same:
        return True, 0
    # if not identical order, check set equality
    return (set(a_ids) == set(b_ids)), 0

def _subset_binary_gold(df_gold: pd.DataFrame) -> pd.DataFrame:
    g = df_gold.copy()
    g["gold_label"] = g["gold_label"].map(_norm_label)
    return g[g["gold_label"].isin(["ADVICE", "STORY"])].copy()

def _domain_label_crosstab(df_gold: pd.DataFrame) -> pd.DataFrame:
    g = df_gold.copy()
    g["gold_label"] = g["gold_label"].map(_norm_label)
    return pd.crosstab(g["domain"], g["gold_label"])


# -------------------------
# Optional: predictions sanity
# If we find paired preds (raw + clean), compute overall vs per-domain macro-F1
# -------------------------
def _macro_f1_from_cm(cm: List[List[int]]) -> float:
    # cm rows=true [ADVICE, STORY], cols=pred [ADVICE, STORY]
    tp_a = cm[0][0]
    fp_a = cm[1][0]
    fn_a = cm[0][1]
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else 0.0
    rec_a  = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
    f1_a = (2 * prec_a * rec_a / (prec_a + rec_a)) if (prec_a + rec_a) else 0.0

    tp_s = cm[1][1]
    fp_s = cm[0][1]
    fn_s = cm[1][0]
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
    rec_s  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
    f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) else 0.0

    return 0.5 * (f1_a + f1_s)

def _find_pred_files() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try to find two prediction CSVs in data_new:
    - one raw-ish (has markers)
    - one clean-ish (markers near 0)
    Expected columns: doc_id, domain, gold_label, pred or pred_label
    We prefer filenames containing 'FROZEN' if present.
    """
    cands = sorted(DATA.glob("preds_*.csv"))
    if not cands:
        return None, None

    def score_file(p: Path) -> Tuple[int, int]:
        # higher score for FROZEN and for "clean" keyword
        name = p.name.lower()
        s1 = 1 if "frozen" in name else 0
        s2 = 1 if "clean" in name else 0
        return (s1, s2)

    # load minimal to compute marker rate on pred file's text if present; otherwise use name heuristic
    usable = []
    for p in cands:
        try:
            df = pd.read_csv(p, nrows=2000)
            if "doc_id" not in df.columns:
                continue
            if "gold_label" not in df.columns:
                continue
            pred_col = "pred_label" if "pred_label" in df.columns else ("pred" if "pred" in df.columns else None)
            if pred_col is None:
                continue
            # domain may be absent in some; tolerate but prefer having it
            usable.append(p)
        except Exception:
            continue

    if len(usable) < 2:
        return None, None

    # heuristic: pick one "raw" and one "clean" by filename keywords first
    usable_sorted = sorted(usable, key=score_file, reverse=True)
    raw = None
    clean = None
    for p in usable_sorted:
        n = p.name.lower()
        if clean is None and ("clean" in n or "artifact" in n):
            clean = p
        elif raw is None and ("raw" in n or "norm" in n):
            raw = p

    # fallback: just pick first two
    if raw is None or clean is None or raw == clean:
        raw, clean = usable_sorted[0], usable_sorted[1]

    return raw, clean

def _load_preds(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, low_memory=False)
    pred_col = "pred_label" if "pred_label" in df.columns else ("pred" if "pred" in df.columns else None)
    if pred_col is None:
        raise RuntimeError(f"[preds] no pred column in {p.name}")
    # standardize col names
    if pred_col != "pred_label":
        df = df.rename(columns={pred_col: "pred_label"})
    # domain optional; if missing, fill unknown
    if "domain" not in df.columns:
        df["domain"] = "UNKNOWN"
    df["gold_label"] = df["gold_label"].map(_norm_label)
    df["pred_label"] = df["pred_label"].map(_norm_label)
    df["doc_id"] = df["doc_id"].astype(str)
    df["domain"] = df["domain"].astype(str)
    return df[["doc_id", "domain", "gold_label", "pred_label"]].copy()

def _cm_binary(gold: pd.Series, pred: pd.Series) -> List[List[int]]:
    # order: ADVICE, STORY
    labs = ["ADVICE", "STORY"]
    g = gold.tolist()
    p = pred.tolist()
    cm = [[0, 0], [0, 0]]
    for gi, pi in zip(g, p):
        if gi not in labs or pi not in labs:
            continue
        r = 0 if gi == "ADVICE" else 1
        c = 0 if pi == "ADVICE" else 1
        cm[r][c] += 1
    return cm

def _preds_sanity(raw_pred_path: Optional[Path], clean_pred_path: Optional[Path]) -> Dict:
    if raw_pred_path is None or clean_pred_path is None:
        return {"found": False, "note": "No preds_*.csv pair found. Run tfidf_preds.py if you need paired-bootstrap plots."}

    raw = _load_preds(raw_pred_path)
    cln = _load_preds(clean_pred_path)

    # pair by doc_id
    df = raw.rename(columns={"pred_label": "pred_raw"}).merge(
        cln[["doc_id", "pred_label"]].rename(columns={"pred_label": "pred_clean"}),
        on="doc_id",
        how="inner"
    )

    # binary only
    df = df[df["gold_label"].isin(["ADVICE", "STORY"])].copy()

    # overall
    cm_over_raw = _cm_binary(df["gold_label"], df["pred_raw"])
    cm_over_cln = _cm_binary(df["gold_label"], df["pred_clean"])
    m_over_raw = _macro_f1_from_cm(cm_over_raw)
    m_over_cln = _macro_f1_from_cm(cm_over_cln)

    # per-domain macro f1
    per_domain = {}
    for dom in sorted(df["domain"].unique()):
        d = df[df["domain"] == dom]
        cm_r = _cm_binary(d["gold_label"], d["pred_raw"])
        cm_c = _cm_binary(d["gold_label"], d["pred_clean"])
        per_domain[dom] = {
            "n": int(len(d)),
            "label_counts": d["gold_label"].value_counts().to_dict(),
            "raw": {"macro_f1": _macro_f1_from_cm(cm_r), "cm": cm_r},
            "clean": {"macro_f1": _macro_f1_from_cm(cm_c), "cm": cm_c},
        }

    # show the “danger confusion” explanation numbers
    dom_raw = [v["raw"]["macro_f1"] for v in per_domain.values()]
    dom_cln = [v["clean"]["macro_f1"] for v in per_domain.values()]
    weighted_raw = sum(v["n"] * v["raw"]["macro_f1"] for v in per_domain.values()) / max(1, sum(v["n"] for v in per_domain.values()))
    weighted_cln = sum(v["n"] * v["clean"]["macro_f1"] for v in per_domain.values()) / max(1, sum(v["n"] for v in per_domain.values()))

    return {
        "found": True,
        "raw_pred_file": raw_pred_path.name,
        "clean_pred_file": clean_pred_path.name,
        "overall": {
            "raw_macro_f1": m_over_raw,
            "clean_macro_f1": m_over_cln,
            "raw_cm": cm_over_raw,
            "clean_cm": cm_over_cln,
        },
        "per_domain": per_domain,
        "intuition_only_averages": {
            "unweighted_avg_domain_raw": float(sum(dom_raw) / max(1, len(dom_raw))),
            "unweighted_avg_domain_clean": float(sum(dom_cln) / max(1, len(dom_cln))),
            "weighted_avg_domain_raw": float(weighted_raw),
            "weighted_avg_domain_clean": float(weighted_cln),
            "note": (
                "These averages are NOT equal to pooled overall macro-F1. "
                "Domain-slice macro-F1 can be much lower when a domain slice is highly imbalanced."
            )
        }
    }


# =========================
# Main
# =========================
def main():
    # 1) require
    for p in [GOLD_RAW, GOLD_CLN, SILV_RAW, SILV_CLN]:
        _require_exists(p)

    # 2) load
    gold_raw = _read_csv(GOLD_RAW)
    gold_cln = _read_csv(GOLD_CLN)
    silv_raw = _read_csv(SILV_RAW)
    silv_cln = _read_csv(SILV_CLN)

    _require_cols(gold_raw, REQ_GOLD_COLS, "gold_raw_norm_FROZEN")
    _require_cols(gold_cln, REQ_GOLD_COLS, "gold_clean_v1_FROZEN")
    _require_cols(silv_raw, REQ_SILV_COLS, "silver_raw_norm_FROZEN")
    _require_cols(silv_cln, REQ_SILV_COLS, "silver_clean_v1_FROZEN")

    # standardize string cols
    gold_raw = _std_str_cols(gold_raw, ["doc_id", "domain", "gold_label", "text"])
    gold_cln = _std_str_cols(gold_cln, ["doc_id", "domain", "gold_label", "text"])
    silv_raw = _std_str_cols(silv_raw, ["doc_id", "domain", "silver_label", "text"])
    silv_cln = _std_str_cols(silv_cln, ["doc_id", "domain", "silver_label", "text"])

    # normalize gold labels (Disclosure -> STORY)
    gold_raw["gold_label"] = gold_raw["gold_label"].map(_norm_label)
    gold_cln["gold_label"] = gold_cln["gold_label"].map(_norm_label)

    # 3) row counts + dupes
    print("==== FILES (FROZEN) ====")
    print("GOLD_RAW :", GOLD_RAW)
    print("GOLD_CLN :", GOLD_CLN)
    print("SILV_RAW :", SILV_RAW)
    print("SILV_CLN :", SILV_CLN)

    print("\n==== ROW COUNTS ====")
    print("gold_raw rows:", len(gold_raw))
    print("gold_cln rows:", len(gold_cln))
    print("silv_raw rows:", len(silv_raw))
    print("silv_cln rows:", len(silv_cln))

    print("\n==== DUPLICATES (doc_id) ====")
    print("gold_raw dup:", _basic_dupes(gold_raw))
    print("gold_cln dup:", _basic_dupes(gold_cln))
    print("silv_raw dup:", _basic_dupes(silv_raw))
    print("silv_cln dup:", _basic_dupes(silv_cln))

    # 4) alignment raw vs clean
    print("\n==== ALIGNMENT raw_norm vs clean_v1 (doc_id) ====")
    ok_gold, _ = _align_by_docid(gold_raw, gold_cln)
    ok_silv, _ = _align_by_docid(silv_raw, silv_cln)
    print("gold aligned (same order OR same set):", ok_gold)
    print("silv aligned (same order OR same set):", ok_silv)

    # 5) leakage (gold ids must not overlap silver ids)
    overlap_raw = _overlap(gold_raw["doc_id"], silv_raw["doc_id"])
    overlap_cln = _overlap(gold_cln["doc_id"], silv_cln["doc_id"])
    print("\n==== LEAKAGE (gold vs silver overlap) ====")
    print("overlap(raw_norm):", overlap_raw)
    print("overlap(clean_v1):", overlap_cln)

    # 6) label counts
    print("\n==== GOLD LABEL COUNTS (overall) ====")
    print(gold_raw["gold_label"].value_counts(dropna=False).to_string())
    print("\n==== GOLD LABEL COUNTS by domain ====")
    print(_domain_label_crosstab(gold_raw).to_string())

    gold_bin = _subset_binary_gold(gold_raw)
    print("\n==== GOLD BINARY SUBSET (ADVICE/STORY only) ====")
    print("binary_n:", len(gold_bin))
    print(gold_bin["gold_label"].value_counts().to_string())

    # 7) changed ratio (raw_norm -> clean_v1)
    # IMPORTANT: raw_norm is already whitespace-normalized baseline;
    # changed_ratio here should reflect artifact removal + placeholder substitution mainly.
    gold_change = _changed_ratio(gold_raw["text"], gold_cln["text"])
    silv_change = _changed_ratio(silv_raw["text"], silv_cln["text"])
    print("\n==== CHANGED RATIO (raw_norm -> clean_v1) ====")
    print(f"gold changed_ratio: {gold_change:.4f}")
    print(f"silv changed_ratio: {silv_change:.4f}")

    # 8) marker rates
    print("\n==== MARKER RATES (gold) ====")
    mr_g_raw = _marker_rates(gold_raw["text"])
    mr_g_cln = _marker_rates(gold_cln["text"])
    print("[gold raw_norm]", json.dumps(mr_g_raw, indent=2))
    print("[gold clean_v1]", json.dumps(mr_g_cln, indent=2))

    print("\n==== MARKER RATES (silver) ====")
    mr_s_raw = _marker_rates(silv_raw["text"])
    mr_s_cln = _marker_rates(silv_cln["text"])
    print("[silver raw_norm]", json.dumps(mr_s_raw, indent=2))
    print("[silver clean_v1]", json.dumps(mr_s_cln, indent=2))

    # 9) optional preds sanity (if preds_*.csv exist)
    raw_pred, clean_pred = _find_pred_files()
    preds_block = _preds_sanity(raw_pred, clean_pred)
    print("\n==== PREDS SANITY (optional) ====")
    print(json.dumps(preds_block, indent=2)[:4000])
    if preds_block.get("found") is False:
        print("NOTE:", preds_block.get("note"))

    # 10) write report json (for reproducibility)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "gold_raw_norm": str(GOLD_RAW),
            "gold_clean_v1": str(GOLD_CLN),
            "silver_raw_norm": str(SILV_RAW),
            "silver_clean_v1": str(SILV_CLN),
        },
        "row_counts": {
            "gold_raw": int(len(gold_raw)),
            "gold_clean": int(len(gold_cln)),
            "silver_raw": int(len(silv_raw)),
            "silver_clean": int(len(silv_cln)),
        },
        "dupes_doc_id": {
            "gold_raw": _basic_dupes(gold_raw),
            "gold_clean": _basic_dupes(gold_cln),
            "silver_raw": _basic_dupes(silv_raw),
            "silver_clean": _basic_dupes(silv_cln),
        },
        "alignment": {
            "gold_raw_vs_clean": bool(ok_gold),
            "silver_raw_vs_clean": bool(ok_silv),
        },
        "leakage_overlap_doc_id": {
            "raw_norm": int(overlap_raw),
            "clean_v1": int(overlap_cln),
        },
        "gold_label_counts_overall": gold_raw["gold_label"].value_counts(dropna=False).to_dict(),
        "gold_label_counts_by_domain": _domain_label_crosstab(gold_raw).to_dict(),
        "gold_binary_n": int(len(gold_bin)),
        "gold_binary_label_counts": gold_bin["gold_label"].value_counts().to_dict(),
        "changed_ratio_rawnorm_to_clean": {
            "gold": float(gold_change),
            "silver": float(silv_change),
        },
        "marker_rates": {
            "gold_raw_norm": mr_g_raw,
            "gold_clean_v1": mr_g_cln,
            "silver_raw_norm": mr_s_raw,
            "silver_clean_v1": mr_s_cln,
        },
        "preds_sanity_optional": preds_block,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote:", OUT_JSON)


if __name__ == "__main__":
    main()