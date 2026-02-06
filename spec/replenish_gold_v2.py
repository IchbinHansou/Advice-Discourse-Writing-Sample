import pandas as pd
from pathlib import Path

GOLD_PATH    = Path("gold_to_annotate_V5_final.xlsx")
DROPPED_PATH = Path("dropped_gold.xlsx")
POOL_PATH    = Path("dataset_raw_submissions_only_safe.csv")

OUT_GOLD_PATH = Path("gold_to_annotate_V5_replenished.xlsx")
OUT_LOG_PATH  = Path("gold_replenishment_log_V5.csv")

RANDOM_SEED = 42

# ---------- Load ----------
gold = pd.read_excel(GOLD_PATH)
dropped = pd.read_excel(DROPPED_PATH)
pool = pd.read_csv(POOL_PATH)

# ---------- Sanity: required cols ----------
need_cols = ["doc_id", "domain", "source_subreddit", "text"]
for df_name, df in [("gold", gold), ("dropped", dropped), ("pool", pool)]:
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")

# ---------- Compute stratified quotas from dropped ----------
quota = (
    dropped.groupby(["domain", "source_subreddit"])
           .size()
           .reset_index(name="n_to_replace")
)

# ---------- Exclusion set: avoid reusing anything already in gold or dropped ----------
used_ids = set(gold["doc_id"].astype(str)) | set(dropped["doc_id"].astype(str))

# ---------- Helper: sample with fallback ----------
def sample_from_pool(domain, subreddit, n):
    # primary: match both domain & subreddit
    cand = pool[
        (pool["domain"] == domain) &
        (pool["source_subreddit"] == subreddit) &
        (~pool["doc_id"].astype(str).isin(used_ids))
    ].copy()

    if len(cand) >= n:
        return cand.sample(n=n, random_state=RANDOM_SEED)

    # fallback 1: match domain only
    cand2 = pool[
        (pool["domain"] == domain) &
        (~pool["doc_id"].astype(str).isin(used_ids))
    ].copy()

    if len(cand2) >= n:
        return cand2.sample(n=n, random_state=RANDOM_SEED)

    # fallback 2: any remaining (last resort)
    cand3 = pool[~pool["doc_id"].astype(str).isin(used_ids)].copy()
    if len(cand3) >= n:
        return cand3.sample(n=n, random_state=RANDOM_SEED)

    raise RuntimeError(f"Not enough candidates to replace {n} rows for {domain}/{subreddit}")

# ---------- Draw replenishment ----------
picked_rows = []
log_rows = []

for _, r in quota.iterrows():
    domain = r["domain"]
    subreddit = r["source_subreddit"]
    n = int(r["n_to_replace"])

    sampled = sample_from_pool(domain, subreddit, n)

    # update used_ids to prevent duplicates across strata sampling
    for did in sampled["doc_id"].astype(str).tolist():
        used_ids.add(did)

    picked_rows.append(sampled)

    for did in sampled["doc_id"].astype(str).tolist():
        log_rows.append({
            "picked_doc_id": did,
            "target_domain": domain,
            "target_source_subreddit": subreddit,
            "reason": "replenish_dropped_gold",
        })

picked = pd.concat(picked_rows, ignore_index=True)

# ---------- Align columns with gold ----------
# Keep gold columns; fill missing cols from picked with NA; ignore extra cols in picked.
for c in gold.columns:
    if c not in picked.columns:
        picked[c] = pd.NA
picked = picked[gold.columns]

# If you want the new ones to be un-labeled:
if "gold_label" in picked.columns:
    picked["gold_label"] = pd.NA
if "notes" in picked.columns:
    picked["notes"] = "NEW_REPLENISHED_SAMPLE"

# ---------- Combine ----------
gold_replenished = pd.concat([gold, picked], ignore_index=True)

# ---------- Write ----------
gold_replenished.to_excel(OUT_GOLD_PATH, index=False)
pd.DataFrame(log_rows).to_csv(OUT_LOG_PATH, index=False)

print("Done.")
print("New gold:", OUT_GOLD_PATH)
print("Log:", OUT_LOG_PATH)
print("Old gold rows:", len(gold), "Picked:", len(picked), "New total:", len(gold_replenished))