import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


# =========================
# 0) Paths (ONLY TWO INPUTS)
# =========================
ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

# (A) Your final labeled gold file (xlsx or csv). THIS is source of truth.
# Recommend: gold_to_annotate_V5_final.xlsx (the one you actually labeled)
GOLD_LABELED_IN = DATA / "gold_to_annotate_V5_final.xlsx"   # <-- 改成你的真实文件名

# (B) Your final silver train file (csv) that you will use for training
SILVER_TRAIN_IN = DATA / "silver_train.csv"                # <-- 如果你最终用的是别的就改这里


# =========================
# Outputs (your requested names)
# =========================
OUT_GOLD_IDS = DATA / "gold_ids_FROZEN.csv"
OUT_SILVER_IDS = DATA / "silver_ids_FROZEN.csv"

OUT_GOLD_RAWNORM = DATA / "gold_raw_norm_FROZEN.csv"
OUT_GOLD_CLEAN   = DATA / "gold_clean_v1_FROZEN.csv"

OUT_SILVER_RAWNORM = DATA / "silver_train_raw_norm_FROZEN.csv"
OUT_SILVER_CLEAN   = DATA / "silver_train_clean_v1_FROZEN.csv"

OUT_LOG = DATA / "freeze_log.json"


# =========================
# 1) Cleaning definitions
# =========================
# IMPORTANT: we separate RAW_NORM vs CLEAN(marker-only on raw_norm).
# RAW_NORM is minimal normalization; CLEAN removes markers but doesn't aggressively squash all whitespace.

RE_AITA = re.compile(r"\b(?:AITA|WIBTA|AITAH)\b", re.I)
RE_VERDICT = re.compile(r"\b(?:YTA|NTA|ESH|NAH)\b", re.I)
RE_EDIT = re.compile(r"\b(?:TL;?DR|EDIT|UPDATE)\b", re.I)

# Age/Gender patterns:
# [25F], [ 25 f ], (25F), 25F, 25 f, 25/f, 25-f  (restrict age range to reduce false hits)
RE_AGEG_BRACKET = re.compile(r"[\[\(\{]\s*(1[3-9]|[2-7]\d|80)\s*([mf])\s*[\]\)\}]", re.I)
RE_AGEG_PLAIN_1 = re.compile(r"\b(1[3-9]|[2-7]\d|80)\s*([mf])\b", re.I)          # 25f / 25 f
RE_AGEG_PLAIN_2 = re.compile(r"\b([mf])\s*(1[3-9]|[2-7]\d|80)\b", re.I)          # f25 / f 25
RE_AGEG_SLASH   = re.compile(r"\b(1[3-9]|[2-7]\d|80)\s*[/\-]\s*([mf])\b", re.I)  # 25/f, 25-f

MARKERS = {
    "aita_acronym": RE_AITA,
    "verdicts": RE_VERDICT,
    "tldr_edit_update": RE_EDIT,
    "age_gender_any": re.compile(
        r"(" +
        RE_AGEG_BRACKET.pattern + r"|" +
        RE_AGEG_PLAIN_1.pattern + r"|" +
        RE_AGEG_PLAIN_2.pattern + r"|" +
        RE_AGEG_SLASH.pattern +
        r")",
        re.I
    ),
}

def raw_norm(text: str) -> str:
    """Minimal normalization: keep paragraph boundaries, avoid collapsing everything into one line."""
    if text is None:
        return ""
    t = str(text)

    # unify newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # normalize tabs
    t = t.replace("\t", " ")

    # collapse repeated spaces but keep newlines
    t = re.sub(r"[ \u00A0]+", " ", t)

    # collapse too many blank lines (3+ -> 2)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def clean_v1_marker_only(text_raw_norm: str) -> str:
    """Remove template markers on top of RAW_NORM; do NOT aggressively squash all whitespace."""
    t = text_raw_norm

    t = RE_AITA.sub(" ", t)
    t = RE_VERDICT.sub(" ", t)
    t = RE_EDIT.sub(" ", t)

    # age/gender replacements
    t = RE_AGEG_BRACKET.sub(" <AGE_GENDER> ", t)
    t = RE_AGEG_SLASH.sub(" <AGE_GENDER> ", t)
    t = RE_AGEG_PLAIN_1.sub(" <AGE_GENDER> ", t)
    t = RE_AGEG_PLAIN_2.sub(" <AGE_GENDER> ", t)

    # light whitespace cleanup (not \s+)
    t = re.sub(r"[ \u00A0]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def marker_rates(series: pd.Series) -> dict:
    txt = series.fillna("").astype(str)
    out = {}
    for k, pat in MARKERS.items():
        # avoid pandas warning about capture groups by using .search on compiled regex ourselves
        out[k] = float(txt.map(lambda x: pat.search(x) is not None).mean())
    return out


def infer_domain_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    if s.startswith("ra_"):
        return "ra"
    if s.startswith("aita_"):
        return "aita"
    if s.startswith("conf_"):
        return "confession"
    # fallback: if your doc_id already contains domain
    return ""


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def pick_col(df: pd.DataFrame, candidates: list[str], where: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[{where}] None of these columns exist: {candidates}. Found={list(df.columns)[:40]} ...")


# =========================
# 2) Main freeze
# =========================
def main():
    DATA.mkdir(parents=True, exist_ok=True)

    gold = load_any(GOLD_LABELED_IN).copy()
    silver = load_any(SILVER_TRAIN_IN).copy()

    # ----- gold columns -----
    gold_id_col = pick_col(gold, ["doc_id", "id", "document_id"], "gold")
    gold_dom_col = "domain" if "domain" in gold.columns else None
    gold_text_col = pick_col(gold, ["text", "post_text", "body", "selftext"], "gold")
    gold_label_col = pick_col(gold, ["gold_label", "label", "y"], "gold")

    gold[gold_id_col] = gold[gold_id_col].astype(str)

    if gold_dom_col is None:
        gold["domain"] = gold[gold_id_col].map(infer_domain_from_doc_id)
        gold_dom_col = "domain"

    # ----- silver columns -----
    sil_id_col = pick_col(silver, ["doc_id", "id", "document_id"], "silver")
    sil_dom_col = "domain" if "domain" in silver.columns else None
    sil_text_col = pick_col(silver, ["text", "post_text", "body", "selftext"], "silver")
    sil_label_col = pick_col(silver, ["silver_label", "y", "split_label"], "silver")

    silver[sil_id_col] = silver[sil_id_col].astype(str)

    if sil_dom_col is None:
        silver["domain"] = silver[sil_id_col].map(infer_domain_from_doc_id)
        sil_dom_col = "domain"

    # ----- leakage check -----
    gold_ids = set(gold[gold_id_col].tolist())
    silver_ids = set(silver[sil_id_col].tolist())
    overlap = gold_ids & silver_ids
    if len(overlap) > 0:
        some = list(sorted(overlap))[:20]
        raise RuntimeError(f"[LEAKAGE] gold and silver overlap: n={len(overlap)} examples={some}")

    # ----- build RAW_NORM + CLEAN for both -----
    gold["text_raw_norm"] = gold[gold_text_col].map(raw_norm)
    gold["text_clean_v1"] = gold["text_raw_norm"].map(clean_v1_marker_only)

    silver["text_raw_norm"] = silver[sil_text_col].map(raw_norm)
    silver["text_clean_v1"] = silver["text_raw_norm"].map(clean_v1_marker_only)

    # ----- build frozen ids -----
    gold_ids_out = gold[[gold_id_col, gold_dom_col]].copy()
    gold_ids_out.columns = ["doc_id", "domain"]

    silver_ids_out = silver[[sil_id_col, sil_dom_col, sil_label_col]].copy()
    silver_ids_out.columns = ["doc_id", "domain", "silver_label"]

    # ----- write frozen datasets -----
    gold_rawnorm_out = gold[[gold_id_col, gold_dom_col, gold_label_col, "text_raw_norm"]].copy()
    gold_rawnorm_out.columns = ["doc_id", "domain", "gold_label", "text"]

    gold_clean_out = gold[[gold_id_col, gold_dom_col, gold_label_col, "text_clean_v1"]].copy()
    gold_clean_out.columns = ["doc_id", "domain", "gold_label", "text"]

    silver_rawnorm_out = silver[[sil_id_col, sil_dom_col, sil_label_col, "text_raw_norm"]].copy()
    silver_rawnorm_out.columns = ["doc_id", "domain", "silver_label", "text"]

    silver_clean_out = silver[[sil_id_col, sil_dom_col, sil_label_col, "text_clean_v1"]].copy()
    silver_clean_out.columns = ["doc_id", "domain", "silver_label", "text"]

    OUT_GOLD_IDS.write_text(gold_ids_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")
    OUT_SILVER_IDS.write_text(silver_ids_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")

    OUT_GOLD_RAWNORM.write_text(gold_rawnorm_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")
    OUT_GOLD_CLEAN.write_text(gold_clean_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")

    OUT_SILVER_RAWNORM.write_text(silver_rawnorm_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")
    OUT_SILVER_CLEAN.write_text(silver_clean_out.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")

    # ----- audits -----
    gold_changed = float((gold["text_raw_norm"] != gold["text_clean_v1"]).mean())
    silver_changed = float((silver["text_raw_norm"] != silver["text_clean_v1"]).mean())

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "gold_labeled_in": str(GOLD_LABELED_IN),
            "silver_train_in": str(SILVER_TRAIN_IN),
        },
        "outputs": {
            "gold_ids": str(OUT_GOLD_IDS),
            "silver_ids": str(OUT_SILVER_IDS),
            "gold_raw_norm": str(OUT_GOLD_RAWNORM),
            "gold_clean_v1": str(OUT_GOLD_CLEAN),
            "silver_raw_norm": str(OUT_SILVER_RAWNORM),
            "silver_clean_v1": str(OUT_SILVER_CLEAN),
        },
        "counts": {
            "gold_n_total": int(len(gold)),
            "gold_label_counts": gold[gold_label_col].astype(str).value_counts(dropna=False).to_dict(),
            "silver_n_total": int(len(silver)),
            "silver_label_counts": silver[sil_label_col].astype(str).value_counts(dropna=False).to_dict(),
        },
        "leakage": {
            "overlap_n": int(len(overlap)),
        },
        "marker_rates_gold_raw_norm": marker_rates(gold["text_raw_norm"]),
        "marker_rates_gold_clean_v1": marker_rates(gold["text_clean_v1"]),
        "marker_rates_silver_raw_norm": marker_rates(silver["text_raw_norm"]),
        "marker_rates_silver_clean_v1": marker_rates(silver["text_clean_v1"]),
        "changed_ratio_rawnorm_to_clean_gold": gold_changed,
        "changed_ratio_rawnorm_to_clean_silver": silver_changed,
    }

    OUT_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("OK. Wrote frozen files:")
    print(" ", OUT_GOLD_IDS)
    print(" ", OUT_SILVER_IDS)
    print(" ", OUT_GOLD_RAWNORM)
    print(" ", OUT_GOLD_CLEAN)
    print(" ", OUT_SILVER_RAWNORM)
    print(" ", OUT_SILVER_CLEAN)
    print(" ", OUT_LOG)
    print("\nAudit snapshot:")
    print("  leakage overlap_n:", len(overlap))
    print("  gold changed_ratio(raw_norm->clean):", f"{gold_changed:.4f}")
    print("  silver changed_ratio(raw_norm->clean):", f"{silver_changed:.4f}")
    print("  gold markers raw_norm:", marker_rates(gold["text_raw_norm"]))
    print("  gold markers clean_v1:", marker_rates(gold["text_clean_v1"]))


if __name__ == "__main__":
    main()