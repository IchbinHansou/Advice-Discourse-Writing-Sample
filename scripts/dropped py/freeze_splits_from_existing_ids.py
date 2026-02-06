from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd


# =========================
# Paths (edit ONLY here)
# =========================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data_new"

# This big table must contain: doc_id, domain, text_raw_norm, text_clean_v1
MASTER_WITH_BOTH = DATA / "dataset_safe_with_RAWNORM_and_CLEANv1.csv"

# Your fixed IDs (do NOT regenerate)
GOLD_IDS = DATA / "gold_ids.csv"          # 300 rows: doc_id, domain
SILVER_IDS = DATA / "silver_ids.csv"      # ~1630 rows: doc_id, domain, split_label (or silver_label)

# Your labeled gold (must contain: doc_id, gold_label)
# Use the file you are sure is the final relabel result.
GOLD_LABELED = DATA / "gold_labeled_300_clean_v1.csv"

# Output (frozen)
OUT_GOLD_IDS = DATA / "gold_ids_FROZEN.csv"
OUT_SILVER_IDS = DATA / "silver_ids_FROZEN.csv"

OUT_GOLD_RAWNORM = DATA / "gold_raw_norm_FROZEN.csv"
OUT_GOLD_CLEAN = DATA / "gold_clean_v1_FROZEN.csv"

OUT_SILVER_RAWNORM = DATA / "silver_train_raw_norm_FROZEN.csv"
OUT_SILVER_CLEAN = DATA / "silver_train_clean_v1_FROZEN.csv"

OUT_LOG = DATA / "freeze_log.json"


# =========================
# Helpers
# =========================
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, na_filter=False)

def _require_cols(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{where}] Missing columns: {missing}. Found columns={list(df.columns)[:30]} ...")

def _as_str(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

def chunk_extract_master(
    master_path: Path,
    need_ids: Set[str],
    chunksize: int = 50000
) -> pd.DataFrame:
    """
    Stream-read the big MASTER_WITH_BOTH and keep only rows whose doc_id is in need_ids.
    """
    usecols = ["doc_id", "domain", "text_raw_norm", "text_clean_v1"]
    out_parts = []
    for chunk in pd.read_csv(master_path, usecols=usecols, chunksize=chunksize, low_memory=False, na_filter=False):
        chunk["doc_id"] = chunk["doc_id"].astype(str)
        sub = chunk[chunk["doc_id"].isin(need_ids)].copy()
        if not sub.empty:
            out_parts.append(sub)
    if not out_parts:
        raise RuntimeError("Extracted 0 rows from master. Check doc_id types / file paths.")
    df = pd.concat(out_parts, ignore_index=True)
    # Deduplicate just in case
    df = df.drop_duplicates(subset=["doc_id"], keep="first")
    return df

def marker_rates(series: pd.Series, markers: Dict[str, str]) -> Dict[str, float]:
    s = series.astype(str)
    out = {}
    for name, pat in markers.items():
        out[name] = float(s.str.contains(pat, regex=True).mean())
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # 1) Load fixed IDs
    gold_ids = _read_csv(GOLD_IDS)
    silver_ids = _read_csv(SILVER_IDS)

    _require_cols(gold_ids, ["doc_id", "domain"], "gold_ids")
    _require_cols(silver_ids, ["doc_id", "domain"], "silver_ids")

    _as_str(gold_ids, ["doc_id", "domain"])
    _as_str(silver_ids, ["doc_id", "domain"])

    # 2) Load labeled gold (for gold_label)
    gold_lab = _read_csv(GOLD_LABELED)
    _require_cols(gold_lab, ["doc_id", "gold_label"], "gold_labeled")
    _as_str(gold_lab, ["doc_id", "gold_label"])

    # Sanity: gold_ids and gold_labeled must align on IDs
    gold_id_set = set(gold_ids["doc_id"].tolist())
    gold_lab_set = set(gold_lab["doc_id"].tolist())
    if gold_id_set != gold_lab_set:
        only_a = sorted(list(gold_id_set - gold_lab_set))[:10]
        only_b = sorted(list(gold_lab_set - gold_id_set))[:10]
        raise RuntimeError(
            "gold_ids and gold_labeled doc_id sets do NOT match.\n"
            f"only_in_gold_ids(sample)={only_a}\n"
            f"only_in_gold_labeled(sample)={only_b}\n"
            "Fix this first: you must freeze from the exact gold you annotated."
        )

    # 3) Decide what we need from master
    silver_id_set = set(silver_ids["doc_id"].tolist())
    need_ids = gold_id_set | silver_id_set

    # 4) Extract from master (raw_norm + clean_v1)
    master = chunk_extract_master(MASTER_WITH_BOTH, need_ids=need_ids, chunksize=50000)
    _require_cols(master, ["doc_id", "domain", "text_raw_norm", "text_clean_v1"], "master_extract")
    _as_str(master, ["doc_id", "domain", "text_raw_norm", "text_clean_v1"])

    # 5) Build GOLD frozen tables
    gold_base = master[master["doc_id"].isin(gold_id_set)].copy()
    # attach labels
    gold_base = gold_base.merge(gold_lab[["doc_id", "gold_label"]], on="doc_id", how="left")
    if gold_base["gold_label"].isna().any():
        raise RuntimeError("Some gold rows missing gold_label after merge. This should never happen.")

    gold_rawnorm = gold_base[["doc_id", "domain", "text_raw_norm", "gold_label"]].rename(
        columns={"text_raw_norm": "text"}
    )
    gold_clean = gold_base[["doc_id", "domain", "text_clean_v1", "gold_label"]].rename(
        columns={"text_clean_v1": "text"}
    )

    # 6) Build SILVER frozen tables
    # label column name might be split_label or silver_label
    label_col = "split_label" if "split_label" in silver_ids.columns else ("silver_label" if "silver_label" in silver_ids.columns else None)
    if label_col is None:
        raise RuntimeError("silver_ids must contain split_label or silver_label.")
    silver_ids2 = silver_ids[["doc_id", "domain", label_col]].copy()
    silver_ids2 = silver_ids2.rename(columns={label_col: "silver_label"})

    silver_base = master[master["doc_id"].isin(silver_id_set)].copy()
    silver_base = silver_base.merge(silver_ids2, on=["doc_id", "domain"], how="left")

    if silver_base["silver_label"].isna().any():
        # show a few to debug
        bad = silver_base[silver_base["silver_label"].isna()][["doc_id", "domain"]].head(10)
        raise RuntimeError(f"Some silver rows missing silver_label after merge. Sample:\n{bad}")

    silver_rawnorm = silver_base[["doc_id", "domain", "text_raw_norm", "silver_label"]].rename(
        columns={"text_raw_norm": "text"}
    )
    silver_clean = silver_base[["doc_id", "domain", "text_clean_v1", "silver_label"]].rename(
        columns={"text_clean_v1": "text"}
    )

    # 7) Basic leakage checks
    overlap = len(set(gold_rawnorm["doc_id"]) & set(silver_rawnorm["doc_id"]))

    # 8) Marker audit (optional but useful)
    MARKERS = {
        "aita_acronym": r"\b(?:AITA|WIBTA|AITAH)\b",
        "verdicts": r"\b(?:YTA|NTA|ESH|NAH)\b",
        "tldr_edit_update": r"\b(?:TL;?DR|EDIT|UPDATE)\b",
        "age_gender_bracket": r"\[\s*(?:1[3-9]|[2-7]\d|80)\s*[MF]\s*\]",
        "age_gender_plain": r"\b(?:1[3-9]|[2-7]\d|80)\s*[MF]\b",
    }

    audit = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "master_with_both": str(MASTER_WITH_BOTH),
            "gold_ids": str(GOLD_IDS),
            "silver_ids": str(SILVER_IDS),
            "gold_labeled": str(GOLD_LABELED),
        },
        "outputs": {
            "gold_ids_frozen": str(OUT_GOLD_IDS),
            "silver_ids_frozen": str(OUT_SILVER_IDS),
            "gold_raw_norm_frozen": str(OUT_GOLD_RAWNORM),
            "gold_clean_v1_frozen": str(OUT_GOLD_CLEAN),
            "silver_train_raw_norm_frozen": str(OUT_SILVER_RAWNORM),
            "silver_train_clean_v1_frozen": str(OUT_SILVER_CLEAN),
        },
        "counts": {
            "gold_n": int(len(gold_rawnorm)),
            "silver_n": int(len(silver_rawnorm)),
            "gold_silver_overlap_ids": int(overlap),
        },
        "marker_rates": {
            "gold_raw_norm": marker_rates(gold_rawnorm["text"], MARKERS),
            "gold_clean_v1": marker_rates(gold_clean["text"], MARKERS),
            "silver_raw_norm": marker_rates(silver_rawnorm["text"], MARKERS),
            "silver_clean_v1": marker_rates(silver_clean["text"], MARKERS),
        },
        # This is the ratio you actually care about (artifact effect, not whitespace):
        "changed_ratio_clean_vs_raw_norm": {
            "gold": float((gold_clean["text"] != gold_rawnorm["text"]).mean()),
            "silver": float((silver_clean["text"] != silver_rawnorm["text"]).mean()),
        }
    }

    if overlap != 0:
        raise RuntimeError(f"Train/Test leakage: gold overlaps silver (overlap_ids={overlap}). Stop.")

    # 9) Write frozen outputs
    gold_ids.to_csv(OUT_GOLD_IDS, index=False, encoding="utf-8-sig")
    silver_ids.to_csv(OUT_SILVER_IDS, index=False, encoding="utf-8-sig")

    gold_rawnorm.to_csv(OUT_GOLD_RAWNORM, index=False, encoding="utf-8-sig")
    gold_clean.to_csv(OUT_GOLD_CLEAN, index=False, encoding="utf-8-sig")

    silver_rawnorm.to_csv(OUT_SILVER_RAWNORM, index=False, encoding="utf-8-sig")
    silver_clean.to_csv(OUT_SILVER_CLEAN, index=False, encoding="utf-8-sig")

    OUT_LOG.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Wrote frozen files:")
    print(" ", OUT_GOLD_IDS.name)
    print(" ", OUT_SILVER_IDS.name)
    print(" ", OUT_GOLD_RAWNORM.name)
    print(" ", OUT_GOLD_CLEAN.name)
    print(" ", OUT_SILVER_RAWNORM.name)
    print(" ", OUT_SILVER_CLEAN.name)
    print(" ", OUT_LOG.name)
    print("Leakage overlap_ids=0 ✅")
    print("Changed ratio (clean vs raw_norm):", audit["changed_ratio_clean_vs_raw_norm"])
    print("Marker rates (gold_clean_v1):", audit["marker_rates"]["gold_clean_v1"])


if __name__ == "__main__":
    main()
