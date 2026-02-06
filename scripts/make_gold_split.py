import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_new"

GOLD_XLSX = DATA / "gold_raw_norm_FROZEN.csv"
OUT_JSON = DATA / "splits_gold_v1.json"

SEED = 1004
TEST_PER_DOMAIN = 30   # 每个 domain 留 30 条进 test

def main():
    df = pd.read_csv(GOLD_XLSX)
    # 只做二分类评估：排除 MIXED
    df_bin = df[df["gold_label"].isin(["ADVICE","STORY"])].copy()

    test_ids = []
    for d, sub in df_bin.groupby("domain"):
        sub = sub.sample(frac=1, random_state=SEED)
        test_ids += sub.head(TEST_PER_DOMAIN)["doc_id"].tolist()

    test_ids = set(test_ids)
    train_ids = [x for x in df_bin["doc_id"].tolist() if x not in test_ids]

    out = {
        "seed": SEED,
        "gold_xlsx": str(GOLD_XLSX.name),
        "rule": f"binary only; TEST_PER_DOMAIN={TEST_PER_DOMAIN} per domain",
        "gold_train_ids": train_ids,
        "gold_test_ids": sorted(list(test_ids)),
        "counts": {
            "train": len(train_ids),
            "test": len(test_ids),
        }
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:", OUT_JSON)

if __name__ == "__main__":
    main()