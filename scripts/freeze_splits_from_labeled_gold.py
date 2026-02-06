import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


# =========================
# Paths (EDIT ONLY HERE)
# =========================
ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

# Prefer canonical pool if exists; otherwise fallback to the known-good pool
POOL_CANON = DATA / "POOL_CANON_safe_rawnorm_cleanv1.csv"
POOL_FALLBACK = DATA / "dataset_safe_with_RAWNORM_and_CLEANv1.csv"

# Your final labeled gold file (source of truth for gold IDs + labels)
GOLD_LABELED_XLSX = DATA / "gold_to_annotate_V5_final.xlsx"

# Your fixed silver ids file (source of truth for silver IDs)
SILVER_IDS_IN = DATA / "silver_ids_FROZEN.csv"


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
# Marker diagnostics
# =========================
MARKERS = {
    "aita_acronym": re.compile(r"\b(?:AITA|WIBTA|AITAH)\b", re.I),
    "verdicts": re.compile(r"\b(?:YTA|NTA|ESH|NAH)\b", re.I),
    "tldr_edit_update": re.compile(r"\b(?:TL;?DR|EDIT|UPDATE)\b", re.I),
    "age_gender_bracket": re.compile(r"\[\s*(?:1[3-9]|[2-7]\d|80)\s*[MF]\s*\]", re.I),
    # plain "25f" / "25 f" (no brackets)
    "age_gender_plain": re.compile(r"(?i)(?<!\w)(?:1[3-9]|[2-7]\d|80)\s*[mf](?!\w)"),
}

def marker_rates(series: pd.Series) -> dict:
    txt = series.fillna("").astype(str)
    out = {}
    for k, pat in MARKERS.items():
        out[k] = float(txt.str.contains(pat, regex=True, na=False).mean())
    return out

def must_have(df: pd.DataFrame, cols: list[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")

def file_meta(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_mb": round(st.st_size / (1024 * 1024), 4),
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }

def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # 1) Load pool (canonical first)
    pool_path = POOL_CANON if POOL_CANON.exists() else POOL_FALLBACK
    if not pool_path.exists():
        raise FileNotFoundError(
            "No pool found. Expected one of:\n"
            f"- {POOL_CANON}\n- {POOL_FALLBACK}"
        )

    pool = pd.read_csv(pool_path, low_memory=False, na_filter=False)
    must_have(pool, ["doc_id", "domain", "text_raw_norm", "text_clean_v1"], "POOL")
    pool["doc_id"] = pool["doc_id"].astype(str)
    pool["domain"] = pool["domain"].astype(str)

    if pool["doc_id"].duplicated().any():
        dup_n = int(pool["doc_id"].duplicated().sum())
        raise RuntimeError(f"POOL has duplicated doc_id rows: {dup_n}. Fix before freezing.")

    # 2) Load gold labeled xlsx (source of truth for gold IDs)
    if not GOLD_LABELED_XLSX.exists():
        raise FileNotFoundError(f"Missing labeled gold file: {GOLD_LABELED_XLSX}")

    gold = pd.read_csv(GOLD_LABELED_XLSX)
    must_have(gold, ["doc_id", "domain", "gold_label"], "GOLD_LABELED_XLSX")
    gold["doc_id"] = gold["doc_id"].astype(str)
    gold["domain"] = gold["domain"].astype(str)
    gold["gold_label"] = gold["gold_label"].astype(str)

    # Freeze gold ids list (exactly what you labeled)
    gold_ids = gold[["doc_id", "domain"]].drop_duplicates().copy()
    if len(gold_ids) != len(gold):
        # If you have duplicated doc_id rows in gold sheet, that's a serious issue.
        dup_n = len(gold) - len(gold_ids)
        raise RuntimeError(f"GOLD sheet has duplicated doc_id rows: {dup_n}. Fix gold file first.")

    # 3) Load silver ids (source of truth for silver IDs)
    if not SILVER_IDS_IN.exists():
        raise FileNotFoundError(f"Missing silver ids file: {SILVER_IDS_IN}")

    silver_ids = pd.read_csv(SILVER_IDS_IN, low_memory=False, na_filter=False)
    # accept either split_label or silver_label
    if "split_label" in silver_ids.columns and "silver_label" not in silver_ids.columns:
        silver_ids = silver_ids.rename(columns={"split_label": "silver_label"})

    must_have(silver_ids, ["doc_id", "domain", "silver_label"], "SILVER_IDS_IN")
    silver_ids["doc_id"] = silver_ids["doc_id"].astype(str)
    silver_ids["domain"] = silver_ids["domain"].astype(str)
    silver_ids["silver_label"] = silver_ids["silver_label"].astype(str)

    # ensure silver ids are unique
    silver_ids_u = silver_ids[["doc_id", "domain", "silver_label"]].drop_duplicates().copy()
    if len(silver_ids_u) != len(silver_ids):
        dup_n = len(silver_ids) - len(silver_ids_u)
        raise RuntimeError(f"SILVER_IDS has duplicated doc_id rows: {dup_n}. Fix silver_ids_FROZEN.csv first.")
    silver_ids = silver_ids_u

    # 4) Leakage check (gold vs silver)
    gold_set = set(gold_ids["doc_id"].tolist())
    silver_set = set(silver_ids["doc_id"].tolist())
    overlap = len(gold_set & silver_set)
    if overlap != 0:
        raise RuntimeError(f"LEAKAGE: gold and silver overlap on doc_id: {overlap}. Must be 0.")

    # 5) Merge texts from pool
    gold_m = gold.merge(
        pool[["doc_id", "text_raw_norm", "text_clean_v1"]],
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    if gold_m["text_raw_norm"].isna().any() or gold_m["text_clean_v1"].isna().any():
        bad = gold_m[gold_m["text_raw_norm"].isna() | gold_m["text_clean_v1"].isna()][["doc_id", "domain"]].head(30)
        raise RuntimeError("Some GOLD ids are missing in POOL. Examples:\n" + bad.to_string(index=False))

    silver_m = silver_ids.merge(
        pool[["doc_id", "text_raw_norm", "text_clean_v1"]],
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    if silver_m["text_raw_norm"].isna().any() or silver_m["text_clean_v1"].isna().any():
        bad = silver_m[silver_m["text_raw_norm"].isna() | silver_m["text_clean_v1"].isna()][["doc_id", "domain"]].head(30)
        raise RuntimeError("Some SILVER ids are missing in POOL. Examples:\n" + bad.to_string(index=False))

    # 6) Build frozen datasets with unified "text" column
    gold_rawnorm = gold_m[["doc_id", "domain", "gold_label", "text_raw_norm"]].rename(columns={"text_raw_norm": "text"})
    gold_clean   = gold_m[["doc_id", "domain", "gold_label", "text_clean_v1"]].rename(columns={"text_clean_v1": "text"})

    silver_rawnorm = silver_m[["doc_id", "domain", "silver_label", "text_raw_norm"]].rename(columns={"text_raw_norm": "text"})
    silver_clean   = silver_m[["doc_id", "domain", "silver_label", "text_clean_v1"]].rename(columns={"text_clean_v1": "text"})

    # 7) Basic sanity checks
    def empty_ratio(df):
        return float((df["text"].astype(str).str.strip() == "").mean())

    gold_change_ratio = float((gold_rawnorm["text"].astype(str).values != gold_clean["text"].astype(str).values).mean())
    silver_change_ratio = float((silver_rawnorm["text"].astype(str).values != silver_clean["text"].astype(str).values).mean())

    audits = {
        "markers_gold_rawnorm": marker_rates(gold_rawnorm["text"]),
        "markers_gold_cleanv1": marker_rates(gold_clean["text"]),
        "markers_silver_rawnorm": marker_rates(silver_rawnorm["text"]),
        "markers_silver_cleanv1": marker_rates(silver_clean["text"]),
        "empty_ratio_gold_rawnorm": empty_ratio(gold_rawnorm),
        "empty_ratio_gold_cleanv1": empty_ratio(gold_clean),
        "empty_ratio_silver_rawnorm": empty_ratio(silver_rawnorm),
        "empty_ratio_silver_cleanv1": empty_ratio(silver_clean),
        "change_ratio_gold_rawnorm_vs_cleanv1": gold_change_ratio,
        "change_ratio_silver_rawnorm_vs_cleanv1": silver_change_ratio,
        "overlap_gold_silver_doc_id": overlap,
    }

    # 8) Write outputs
    gold_ids.to_csv(OUT_GOLD_IDS, index=False, encoding="utf-8-sig")
    silver_ids.to_csv(OUT_SILVER_IDS, index=False, encoding="utf-8-sig")

    gold_rawnorm.to_csv(OUT_GOLD_RAWNORM, index=False, encoding="utf-8-sig")
    gold_clean.to_csv(OUT_GOLD_CLEAN, index=False, encoding="utf-8-sig")
    silver_rawnorm.to_csv(OUT_SILVER_RAWNORM, index=False, encoding="utf-8-sig")
    silver_clean.to_csv(OUT_SILVER_CLEAN, index=False, encoding="utf-8-sig")

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "pool": file_meta(pool_path),
            "gold_labeled_xlsx": file_meta(GOLD_LABELED_XLSX),
            "silver_ids_in": file_meta(SILVER_IDS_IN),
        },
        "outputs": {
            "gold_ids_frozen": str(OUT_GOLD_IDS),
            "silver_ids_frozen": str(OUT_SILVER_IDS),
            "gold_rawnorm_frozen": str(OUT_GOLD_RAWNORM),
            "gold_cleanv1_frozen": str(OUT_GOLD_CLEAN),
            "silver_rawnorm_frozen": str(OUT_SILVER_RAWNORM),
            "silver_cleanv1_frozen": str(OUT_SILVER_CLEAN),
        },
        "counts": {
            "gold_n": int(len(gold_ids)),
            "silver_n": int(len(silver_ids)),
            "gold_by_domain": gold["domain"].value_counts().to_dict(),
            "gold_by_label": gold["gold_label"].value_counts().to_dict(),
            "silver_by_domain": silver_ids["domain"].value_counts().to_dict(),
            "silver_by_label": silver_ids["silver_label"].value_counts().to_dict(),
        },
        "audits": audits,
    }
    OUT_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Wrote frozen files:")
    print(" ", OUT_GOLD_IDS.name)
    print(" ", OUT_SILVER_IDS.name)
    print(" ", OUT_GOLD_RAWNORM.name)
    print(" ", OUT_GOLD_CLEAN.name)
    print(" ", OUT_SILVER_RAWNORM.name)
    print(" ", OUT_SILVER_CLEAN.name)
    print(" ", OUT_LOG.name)
    print("\n[Audit summary]")
    for k, v in audits.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()