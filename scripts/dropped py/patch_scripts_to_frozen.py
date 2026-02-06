from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"

# ---- what you want to standardize to ----
FROZEN_FILES = {
    # gold
    "gold_to_annotate_V5_final.xlsx": "gold_raw_norm_FROZEN.csv",
    "gold_labeled_300.csv": "gold_raw_norm_FROZEN.csv",
    "gold_labeled_300_clean_v1.csv": "gold_clean_v1_FROZEN.csv",

    # common gold variants people accidentally use
    "gold_binary.csv": "gold_raw_norm_FROZEN.csv",
    "gold_binary_clean_v1.csv": "gold_clean_v1_FROZEN.csv",
    "gold_clean_binary_only.csv": "gold_clean_v1_FROZEN.csv",
    "gold_clean_binary_only_v2.csv": "gold_clean_v1_FROZEN.csv",

    # silver
    "silver_train.csv": "silver_train_raw_norm_FROZEN.csv",
    "silver_train_clean_v1.csv": "silver_train_clean_v1_FROZEN.csv",
    "silver_train_v2.csv": "silver_train_raw_norm_FROZEN.csv",

    # ids
    "gold_ids.csv": "gold_ids_FROZEN.csv",
    "silver_ids.csv": "silver_ids_FROZEN.csv",
}

# If a script refers to these big pools directly, you usually want the master with both columns.
POOL_REWRITES = {
    "dataset_raw_submissions_only_safe.csv": "dataset_safe_with_RAWNORM_and_CLEANv1.csv",
    "dataset_raw_submissions_only_safe_with_clean_v1.csv": "dataset_safe_with_RAWNORM_and_CLEANv1.csv",
    "dataset_safe_RAW_NORM.csv": "dataset_safe_with_RAWNORM_and_CLEANv1.csv",
    "dataset_safe_CLEAN_v1.csv": "dataset_safe_with_RAWNORM_and_CLEANv1.csv",
}

EXCLUDE = {
    "patch_scripts_to_frozen.py",
    "freeze_splits_from_existing_ids.py",
}

def patch_text(txt: str) -> str:
    # 1) Replace filenames (quoted strings or Path/"file" usage)
    for old, new in {**FROZEN_FILES, **POOL_REWRITES}.items():
        txt = txt.replace(old, new)

    # 2) If we replaced xlsx->csv, some scripts still call read_excel(sheet_name=0).
    #    Convert read_excel(...) to read_csv(...) if the file no longer uses .xlsx.
    if ".xlsx" not in txt and "pd.read_excel" in txt:
        # remove sheet_name=... in the call
        txt = re.sub(r"pd\.read_excel\(", "pd.read_csv(", txt)
        # remove sheet_name argument variants (simple safe patterns)
        txt = re.sub(r",\s*sheet_name\s*=\s*[^,\)]*", "", txt)
        txt = re.sub(r"sheet_name\s*=\s*[^,\)]*,\s*", "", txt)

    return txt

def main():
    if not SCRIPTS.exists():
        raise RuntimeError(f"scripts dir not found: {SCRIPTS}")

    py_files = sorted([p for p in SCRIPTS.rglob("*.py") if p.name not in EXCLUDE])

    changed = 0
    touched = []

    for p in py_files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        new = patch_text(raw)
        if new != raw:
            # backup once
            bak = p.with_suffix(p.suffix + ".bak")
            if not bak.exists():
                bak.write_text(raw, encoding="utf-8")
            p.write_text(new, encoding="utf-8")
            changed += 1
            touched.append(p.name)

    print(f"Patched {changed} files.")
    if touched:
        print("Touched:")
        for name in touched:
            print(" -", name)
    print("Backups saved as *.py.bak (only created once).")

if __name__ == "__main__":
    main()