# replenish_gold.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_xlsx", required=True, help=""C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\scripts\gold_to_annotate_V2.xlsx"")
    ap.add_argument("--master_csv", required=True, help="Master RAW (or CLEANED) CSV containing all domains.")
    ap.add_argument("--out_xlsx", default=None, help="Output XLSX path. Default: <gold_stem>__replenished.xlsx")
    ap.add_argument("--target_per_domain", type=int, default=100, help="Target rows per domain (default 100).")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling.")
    ap.add_argument("--exclude_ids", default=None, help="Optional txt file with doc_id to exclude (one per line).")
    args = ap.parse_args()

    gold_path = Path(args.gold_xlsx)
    master_path = Path(args.master_csv)

    if args.out_xlsx is None:
        args.out_xlsx = str(gold_path.with_name(gold_path.stem + "__replenished.xlsx"))

    # Read files
    gold = pd.read_excel(gold_path)  # reads first sheet by default
    master = pd.read_csv(master_path)

    # Basic checks
    for col in ["doc_id", "domain"]:
        if col not in gold.columns:
            raise ValueError(f"gold_xlsx missing required column: {col}")
        if col not in master.columns:
            raise ValueError(f"master_csv missing required column: {col}")

    gold["doc_id"] = gold["doc_id"].astype(str)
    master["doc_id"] = master["doc_id"].astype(str)

    # Optional exclude list
    exclude_set = set()
    if args.exclude_ids:
        p = Path(args.exclude_ids)
        if p.exists():
            exclude_set = set(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())

    # Current IDs in gold
    gold_ids = set(gold["doc_id"].dropna().astype(str).tolist())

    # Domains to enforce targets for: infer from gold
    domains = sorted([d for d in gold["domain"].dropna().unique().tolist()])

    # Count & compute needs
    counts_before = gold.groupby("domain")["doc_id"].count().to_dict()
    needs = {d: max(0, args.target_per_domain - counts_before.get(d, 0)) for d in domains}

    print("=== Counts BEFORE ===")
    for d in domains:
        print(f"{d}: {counts_before.get(d, 0)} / {args.target_per_domain} (need {needs[d]})")

    # Prepare rows to add
    to_add = []
    for d in domains:
        k = needs[d]
        if k <= 0:
            continue

        pool = master[master["domain"] == d].copy()
        # drop already-in-gold and excluded
        pool = pool[~pool["doc_id"].isin(gold_ids)]
        if exclude_set:
            pool = pool[~pool["doc_id"].isin(exclude_set)]

        if len(pool) < k:
            raise RuntimeError(
                f"Not enough candidates to sample for domain={d}. "
                f"Need {k}, but only {len(pool)} available after exclusions."
            )

        sampled = pool.sample(n=k, random_state=args.seed)
        to_add.append(sampled)

    if to_add:
        add_df = pd.concat(to_add, ignore_index=True)
    else:
        add_df = master.iloc[0:0].copy()  # empty

    # Align columns: keep gold’s columns order; fill missing columns with NA
    for c in gold.columns:
        if c not in add_df.columns:
            add_df[c] = pd.NA
    add_df = add_df[gold.columns]

    # Append and sanity checks
    out = pd.concat([gold, add_df], ignore_index=True)

    if out["doc_id"].duplicated().any():
        dups = out.loc[out["doc_id"].duplicated(), "doc_id"].astype(str).unique().tolist()[:20]
        raise RuntimeError(f"Duplicate doc_id detected after merge. Examples: {dups}")

    counts_after = out.groupby("domain")["doc_id"].count().to_dict()

    print("\n=== Counts AFTER ===")
    for d in domains:
        print(f"{d}: {counts_after.get(d, 0)} / {args.target_per_domain}")

    # Write
    out.to_excel(args.out_xlsx, index=False)
    print(f"\nSaved: {args.out_xlsx}")

if __name__ == "__main__":
    main()