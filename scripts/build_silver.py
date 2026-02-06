import pandas as pd

POOL = r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new\dataset_raw_submissions_only.csv"
GOLD_IDS = "gold_ids_FROZEN.csv"
OUT = "silver_train_V2.csv"
SEED = 1004

pool = pd.read_csv(POOL, low_memory=False)
gold_ids = pd.read_csv(GOLD_IDS)

pool["doc_id"] = pool["doc_id"].astype(str)
gold_ids["doc_id"] = gold_ids["doc_id"].astype(str)

gold_set = set(gold_ids["doc_id"].tolist())

# 方案A：只训练 ra + confession，排除 gold
train_pool = pool[
    (pool["domain"].isin(["ra", "confession"])) &
    (~pool["doc_id"].isin(gold_set))
].copy()

counts = train_pool["domain"].value_counts()
print("available after excluding gold:")
print(counts)

# 抽样规模：取两个域里较小的那个（自动适配ra变少的问题）
n = int(min(counts.get("ra", 0), counts.get("confession", 0)))

# 给自己留buffer（可选）：比如少抽50条
# n = max(0, n - 50)

if n <= 0:
    raise ValueError("No data available to sample. Check domain names and doc_id fields.")

silver = (
    train_pool.groupby("domain", group_keys=False)
    .sample(n=n, random_state=SEED)
    .reset_index(drop=True)
)

silver.to_csv(OUT, index=False)

print("silver rows:", len(silver))
print(silver["domain"].value_counts())

# 最重要：检查是否与gold重叠（必须为0）
overlap = set(silver["doc_id"].astype(str)) & gold_set
print("overlap with gold:", len(overlap))