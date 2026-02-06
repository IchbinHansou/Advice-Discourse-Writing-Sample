#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import pandas as pd

def norm(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    return str(x).strip().lower()


def resolve_from_repo(repo_root: Path, p: str) -> Path:
    """Resolve path relative to repo root unless it's already absolute."""
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def safe_read_excel(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")
    # 强制全列字符串 + 指定引擎
    return pd.read_excel(xlsx_path, engine="openpyxl", dtype=str)


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    try:
        return pd.read_csv(csv_path, low_memory=False, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig", dtype=str)


def collect_exclude_ids(dropped_xlsx: Path | None) -> set[str]:
    if dropped_xlsx is None:
        return set()

    df = safe_read_excel(dropped_xlsx)

    # 1) 列名统一转成字符串再 lower，避免 int 没有 lower()
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # 2) 如果没有 doc_id 列：尝试把第一行当表头（很多人保存 xlsx 会这样）
    if "doc_id" not in cols_lower:
        if len(df) > 0:
            first_row = [str(x).strip().lower() for x in df.iloc[0].tolist()]
            if "doc_id" in first_row:
                df.columns = first_row
                df = df.iloc[1:].copy()
                cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # 3) 还没有 doc_id：最后兜底，默认第一列就是 doc_id
    if "doc_id" in cols_lower:
        col = cols_lower["doc_id"]
    else:
        col = df.columns[0]

    s = df[col].dropna().astype(str).str.strip()
    return set(x for x in s.tolist() if x and x.lower() != "nan")


def infer_repo_root(script_is_in_scripts: bool) -> Path:
    """
    If this file is in scripts/, repo root is parent of scripts/.
    If you put it in repo root, repo root is the file's parent.
    """
    here = Path(__file__).resolve()
    if script_is_in_scripts:
        return here.parent.parent
    return here.parent


def main():
    parser = argparse.ArgumentParser(
        description="Replenish gold annotation sheet by sampling from master CSV per domain."
    )
    parser.add_argument("--gold_xlsx", required=True, help="e.g. gold_to_annotate_V2.xlsx")
    parser.add_argument("--master_csv", required=True, help="e.g. dataset_raw.csv (or cleaned)")
    parser.add_argument("--dropped_xlsx", default=None, help="e.g. dropped_gold.xlsx (optional)")
    parser.add_argument("--out_xlsx", default=None, help="output xlsx path (optional)")
    parser.add_argument("--target_per_domain", type=int, default=100, help="default: 100")
    parser.add_argument("--seed", type=int, default=13, help="random seed for sampling")
    parser.add_argument(
        "--repo_root_mode",
        choices=["scripts", "root"],
        default="scripts",
        help="scripts: assume this file is under scripts/ so repo root is parent.parent; "
             "root: assume this file is in repo root so repo root is parent.",
    )
    args = parser.parse_args()

    # ---- repo root ----
    REPO_ROOT = infer_repo_root(script_is_in_scripts=(args.repo_root_mode == "scripts"))

    gold_path = resolve_from_repo(REPO_ROOT, args.gold_xlsx)
    master_path = resolve_from_repo(REPO_ROOT, args.master_csv)
    dropped_path = resolve_from_repo(REPO_ROOT, args.dropped_xlsx) if args.dropped_xlsx else None

    gold_df = safe_read_excel(gold_path)
    master_df = safe_read_csv(master_path)

    # ---- required columns ----
    # gold must have doc_id + domain
    gold_cols_lower = {c.lower(): c for c in gold_df.columns}
    if "doc_id" not in gold_cols_lower or "domain" not in gold_cols_lower:
        raise ValueError(
            f"gold_xlsx must contain columns: doc_id, domain. Got: {list(gold_df.columns)}"
        )

    gold_doc_col = gold_cols_lower["doc_id"]
    gold_domain_col = gold_cols_lower["domain"]

    master_cols_lower = {c.lower(): c for c in master_df.columns}
    if "doc_id" not in master_cols_lower or "domain" not in master_cols_lower:
        raise ValueError(
        f"master_csv must contain columns: doc_id, domain. Got: {list(master_df.columns)}"
        )
    master_doc_col = master_cols_lower["doc_id"]
    master_domain_col = master_cols_lower["domain"]

    master_df[master_doc_col] = master_df[master_doc_col].astype(str)
    master_df[master_domain_col] = master_df[master_domain_col].astype(str)

    # Some excel empty cells become 'nan' after astype(str)
    def is_real_id(x: str) -> bool:
        x = (x or "").strip()
        return x and x.lower() != "nan"

    existing_ids = set(
        (gold_df[gold_doc_col].dropna().astype(str).str.strip())
        .tolist()
    )
    existing_ids = set(x for x in existing_ids if is_real_id(x))

    exclude_ids = set()
    if dropped_path is not None:
        exclude_ids = collect_exclude_ids(dropped_path)

    # ---- domain list from gold (so you keep same set) ----
    domains = (
        gold_df[gold_domain_col]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
        .tolist()
    )
    domains = sorted({d for d in domains if d and d.lower() != "nan"})

    # ---- replenish per domain ----
    gold_out = gold_df.copy()
    new_rows_all: list[dict] = []

    for d in domains:
        d_norm = norm(d)

        cur = gold_out[gold_out[gold_domain_col].map(norm) == d_norm]
        cur_ids = cur[gold_doc_col].astype(str).str.strip()
        cur_ids = [x for x in cur_ids.tolist() if is_real_id(x)]
        cur_count = len(cur_ids)

        need = args.target_per_domain - cur_count
        if need <= 0:
            continue

        pool = master_df[master_df[master_domain_col].map(norm) == d_norm].copy()
        pool_ids = pool[master_doc_col].astype(str).str.strip()

        # exclude current + dropped
        mask_keep = ~pool_ids.isin(existing_ids.union(exclude_ids))
        pool = pool[mask_keep]

        if len(pool) < need:
            raise RuntimeError(
                f"Not enough candidates to sample for domain={d}. "
                f"need={need}, available={len(pool)} after excluding existing/dropped."
            )

        sampled = pool.sample(n=need, random_state=args.seed)

        # build rows matching gold columns
        for _, r in sampled.iterrows():
            row = {}
            for col in gold_out.columns:
                row[col] = r[col] if col in master_df.columns else ""
            new_rows_all.append(row)

        existing_ids.update(sampled[master_doc_col].astype(str).str.strip().tolist())

    if new_rows_all:
        new_df = pd.DataFrame(new_rows_all, columns=list(gold_out.columns))
        gold_out = pd.concat([gold_out, new_df], ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=list(gold_out.columns))

    # ---- output path ----
    if args.out_xlsx:
        out_path = resolve_from_repo(REPO_ROOT, args.out_xlsx)
    else:
        # default: same folder as gold, add suffix
        out_path = gold_path.parent / f"{gold_path.stem}_replenished.xlsx"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_out.to_excel(out_path, index=False)

    # also save a small "what was added" csv for audit
    audit_path = out_path.with_suffix(".added.csv")
    if len(new_df) > 0:
        new_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    else:
        # create empty audit file so pipeline is deterministic
        pd.DataFrame(columns=list(gold_out.columns)).to_csv(audit_path, index=False, encoding="utf-8-sig")

    # ---- print summary ----
    print("=== Replenish Summary ===")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"GOLD_IN:   {gold_path}")
    print(f"MASTER:    {master_path}")
    if dropped_path:
        print(f"DROPPED:   {dropped_path} (exclude {len(exclude_ids)} ids)")
    print(f"TARGET_PER_DOMAIN: {args.target_per_domain}")
    print(f"ADDED_ROWS: {len(new_df)}")
    print(f"GOLD_OUT:  {out_path}")
    print(f"AUDIT_CSV: {audit_path}")

    # per-domain counts
    gold_domain_series = gold_out[gold_domain_col].astype(str).str.strip()
    gold_id_series = gold_out[gold_doc_col].astype(str).str.strip()
    valid_mask = gold_id_series.apply(is_real_id)

    counts = (
        gold_out.loc[valid_mask, [gold_domain_col, gold_doc_col]]
        .groupby(gold_domain_col)[gold_doc_col]
        .nunique()
        .sort_index()
    )
    print("\n=== Per-domain unique doc_id counts (after) ===")
    for k, v in counts.items():
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
