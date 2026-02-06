# import pandas as pd

# POOL = "dataset_raw_submissions_only.csv"
# GOLD_IDS = "gold_ids_FROZEN.csv"
# OUT = "silver_train_raw_norm_FROZEN.csv"
# SEED = 1004

# pool = pd.read_csv(POOL, low_memory=False)
# gold_ids = pd.read_csv(GOLD_IDS)

# pool["doc_id"] = pool["doc_id"].astype(str)
# gold_ids["doc_id"] = gold_ids["doc_id"].astype(str)

# gold_set = set(gold_ids["doc_id"].tolist())

# # 方案A：只用 ra + confession 训练，AITA不进silver
# train_pool = pool[(pool["domain"].isin(["ra","confession"])) & (~pool["doc_id"].isin(gold_set))].copy()

# # 每域抽多少
# counts = train_pool["domain"].value_counts().to_dict()
# n = min(counts.get("ra",0), counts.get("confession",0))
# # 你也可以手动设 n=900 留buffer
# # n = 900

# silver = (
#     train_pool.groupby("domain", group_keys=False)
#     .apply(lambda x: x.sample(n=n, random_state=SEED))
#     .reset_index(drop=True)
# )

# silver.to_csv(OUT, index=False)
# print("silver rows:", len(silver))
# print(silver["domain"].value_counts())

import pandas as pd

POOL = "dataset_raw_submissions_only.csv"
GOLD = "gold_to_annotate_V3.xlsx"
OUT  = "gold_to_annotate_V4_final.xlsx"
SEED = 1004
TARGET_PER_DOMAIN = 100

pool = pd.read_csv(POOL, low_memory=False)
gold = pd.read_excel(GOLD)

pool["doc_id"] = pool["doc_id"].astype(str)
gold["doc_id"] = gold["doc_id"].astype(str)

taken = set(gold["doc_id"].tolist())

# 现在每个域有多少
cur = gold["domain"].value_counts().to_dict()
need = {d: TARGET_PER_DOMAIN - cur.get(d, 0) for d in ["ra","confession","aita"]}

print("current:", cur)
print("need:", need)

new_parts = []
for dom, k in need.items():
    if k <= 0:
        continue
    cand = pool[(pool["domain"] == dom) & (~pool["doc_id"].isin(taken))].copy()
    if len(cand) < k:
        raise ValueError(f"Not enough candidates for domain={dom}. Need {k}, have {len(cand)}")
    add = cand.sample(n=k, random_state=SEED)
    taken.update(add["doc_id"].tolist())

    # 对齐到 gold 的列（保留 gold 现有列，缺的就填空）
    add_aligned = pd.DataFrame({c: add[c] if c in add.columns else "" for c in gold.columns})
    new_parts.append(add_aligned)

if new_parts:
    gold_final = pd.concat([gold] + new_parts, ignore_index=True)
else:
    gold_final = gold.copy()

gold_final.to_excel(OUT, index=False)
print("gold_final rows:", len(gold_final))
print(gold_final["domain"].value_counts())