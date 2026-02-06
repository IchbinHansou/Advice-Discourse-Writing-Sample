import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data_new" / "dataset_raw.csv"
CLN_PATH = ROOT / "data_new" / "dataset_cleaned.csv"
CFG_PATH = ROOT / "data_new" / "run_config.json"

OUT_GOLD_IDS = ROOT / "data_new" / "gold_ids_FROZEN.csv"
OUT_GOLD_ANN = ROOT / "data_new" / "gold_to_annotate.csv"
OUT_SILVER_IDS = ROOT / "data_new" / "silver_ids_FROZEN.csv"
OUT_SILVER_TRN = ROOT / "data_new" / "silver_train_raw_norm_FROZEN.csv"

SEED_FALLBACK = 1004
GOLD_PER_DOMAIN = 100

# Silver 上限（每类），避免 silver 巨大
SILVER_PER_CLASS_FALLBACK = 1800

ID_COLS = ["doc_id", "domain"]
RAW_ANN_COLS = ["doc_id", "domain", "source_subreddit", "text"]
CLN_TRN_COLS = ["doc_id", "domain", "text"]

# 简单的 advice 请求短语（用于定向抽样，不是用于标注！）
ADVICE_PAT = re.compile(
    r"\b(any advice|need advice|what should i do|how do i|should i|help me|"
    r"am i wrong|am i the asshole|aita|wibta)\b",
    flags=re.IGNORECASE
)

def load_cfg():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", SEED_FALLBACK))
    silver_per_class = int(cfg.get("silver_per_class", SILVER_PER_CLASS_FALLBACK))
    return cfg, seed, silver_per_class

def iter_raw_text_chunks(usecols, chunksize=50000):
    dtype = {c: "string" for c in usecols}
    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
        na_filter=False
    ):
        yield chunk

def collect_candidates(dom: str, predicate, max_pool: int, seed: int) -> list[str]:
    """
    从 RAW 里按 domain 过滤，再用 predicate(text) 筛 candidate doc_id。
    max_pool: 收集到这么多就停，避免扫完整个 1.3GB 还不停。
    """
    got = []
    for chunk in iter_raw_text_chunks(["doc_id", "domain", "text"]):
        sub = chunk[chunk["domain"] == dom]
        if sub.empty:
            continue

        # predicate 是对 text 的判断
        mask = sub["text"].astype(str).map(predicate)
        if mask.any():
            got.extend(sub.loc[mask, "doc_id"].astype(str).tolist())

        if len(got) >= max_pool:
            break

    # 为了可复现：打乱一下顺序（但最终 sample 仍由 random_state 控制）
    rs = pd.Series(got).sample(frac=1.0, random_state=seed).tolist() if got else []
    return rs

def sample_gold_ids(seed: int):
    # 先读 ID（快）
    df_ids = pd.read_csv(
        RAW_PATH,
        usecols=ID_COLS,
        dtype={"doc_id": "string", "domain": "string"},
        low_memory=False,
        na_filter=False
    )

    gold_parts = []
    half = GOLD_PER_DOMAIN // 2  # 50

    for dom in ["ra", "confession", "aita"]:
        dom_ids = df_ids[df_ids["domain"] == dom].copy()
        if len(dom_ids) < GOLD_PER_DOMAIN:
            raise ValueError(f"Domain {dom} has only {len(dom_ids)} rows, < {GOLD_PER_DOMAIN}")

        # 1) 先抽 50 随机
        rand = dom_ids.sample(n=half, random_state=seed).copy()
        picked = set(rand["doc_id"].astype(str).tolist())

        # 2) 再抽 50 “定向反直觉”
        # ra: 抽 story-like（少问句、少 advice phrase）
        # confession: 抽 advice-like（有 advice phrase 或问句）
        # aita: 先不做定向，直接再随机补足
        need = GOLD_PER_DOMAIN - len(picked)

        targeted_ids = []
        if dom == "ra":
            def is_storylike_in_ra(t: str) -> bool:
                t = t or ""
                # 没有问号，且不含明显求建议短语
                return ("?" not in t) and (ADVICE_PAT.search(t) is None)
            pool = collect_candidates(dom, is_storylike_in_ra, max_pool=8000, seed=seed + 11)
            pool = [x for x in pool if x not in picked]
            if len(pool) > 0:
                targeted_ids = pd.Series(pool).sample(
                    n=min(need, len(pool)),
                    random_state=seed + 101
                ).tolist()

        elif dom == "confession":
            def is_advicelike_in_conf(t: str) -> bool:
                t = t or ""
                if ADVICE_PAT.search(t) is not None:
                    return True
                # 或者至少有一个问号（更宽松一点，保证能抽到）
                return ("?" in t)
            pool = collect_candidates(dom, is_advicelike_in_conf, max_pool=8000, seed=seed + 22)
            pool = [x for x in pool if x not in picked]
            if len(pool) > 0:
                targeted_ids = pd.Series(pool).sample(
                    n=min(need, len(pool)),
                    random_state=seed + 202
                ).tolist()

        # 3) 如果 targeted 不够，就用随机补足（排除已选）
        picked.update(targeted_ids)
        need = GOLD_PER_DOMAIN - len(picked)
        if need > 0:
            remaining = dom_ids[~dom_ids["doc_id"].astype(str).isin(picked)]
            fill = remaining.sample(n=need, random_state=seed + 303)
            picked.update(fill["doc_id"].astype(str).tolist())

        gold = dom_ids[dom_ids["doc_id"].astype(str).isin(picked)][["doc_id", "domain"]].copy()
        # 保证每域正好 100
        gold = gold.sample(n=GOLD_PER_DOMAIN, random_state=seed + 404)
        gold_parts.append(gold)

    gold_ids = pd.concat(gold_parts, ignore_index=True)
    return gold_ids

def build_silver_ids(seed: int, gold_ids: pd.DataFrame, silver_per_class: int):
    gold_set = set(gold_ids["doc_id"].astype(str).tolist())

    df_ids = pd.read_csv(
        CLN_PATH,
        usecols=ID_COLS,
        dtype={"doc_id": "string", "domain": "string"},
        low_memory=False,
        na_filter=False
    )

    ra_pool = df_ids[df_ids["domain"] == "ra"].copy()
    ra_pool = ra_pool[~ra_pool["doc_id"].astype(str).isin(gold_set)]

    conf_pool = df_ids[df_ids["domain"] == "confession"].copy()
    conf_pool = conf_pool[~conf_pool["doc_id"].astype(str).isin(gold_set)]

    # 自动下调到可用最小值，保证永远可抽
    cap = min(silver_per_class, len(ra_pool), len(conf_pool))
    if cap < 200:
        raise ValueError(f"silver pool too small after filtering: cap={cap} (ra={len(ra_pool)}, conf={len(conf_pool)})")

    ra_ids = ra_pool.sample(n=cap, random_state=seed).copy()
    ra_ids["split_label"] = "ADVICE"

    conf_ids = conf_pool.sample(n=cap, random_state=seed + 1).copy()
    conf_ids["split_label"] = "STORY"

    silver_ids = pd.concat([ra_ids, conf_ids], ignore_index=True)
    return silver_ids

def extract_rows(in_path: Path, out_path: Path, keep_ids: set, usecols: list, add_empty_cols: dict | None = None):
    first = True
    for chunk in pd.read_csv(in_path, usecols=usecols, chunksize=50000, low_memory=False, na_filter=False):
        chunk["doc_id"] = chunk["doc_id"].astype(str)
        sub = chunk[chunk["doc_id"].isin(keep_ids)].copy()
        if sub.empty:
            continue
        if add_empty_cols:
            for k, v in add_empty_cols.items():
                sub[k] = v
        sub.to_csv(
            out_path,
            index=False,
            mode="w" if first else "a",
            header=first,
            encoding="utf-8-sig"
        )
        first = False

def main():
    cfg, seed, silver_per_class = load_cfg()

    gold_ids = sample_gold_ids(seed)
    gold_ids.to_csv(OUT_GOLD_IDS, index=False, encoding="utf-8-sig")

    gold_set = set(gold_ids["doc_id"].astype(str).tolist())
    extract_rows(
        RAW_PATH, OUT_GOLD_ANN, gold_set,
        usecols=RAW_ANN_COLS,
        add_empty_cols={"gold_label": "", "notes": ""}
    )

    silver_ids = build_silver_ids(seed, gold_ids, silver_per_class=silver_per_class)
    silver_ids.to_csv(OUT_SILVER_IDS, index=False, encoding="utf-8-sig")

    silver_set = set(silver_ids["doc_id"].astype(str).tolist())
    tmp_path = ROOT / "data_new" / "_tmp_silver_text.csv"
    extract_rows(CLN_PATH, tmp_path, silver_set, usecols=CLN_TRN_COLS)

    silver_text = pd.read_csv(tmp_path, low_memory=False, na_filter=False)
    silver = silver_text.merge(silver_ids[["doc_id", "split_label"]], on="doc_id", how="left")
    silver = silver.rename(columns={"split_label": "y"})
    silver.to_csv(OUT_SILVER_TRN, index=False, encoding="utf-8-sig")
    tmp_path.unlink(missing_ok=True)

    print("Wrote:")
    print(" ", OUT_GOLD_IDS)
    print(" ", OUT_GOLD_ANN)
    print(" ", OUT_SILVER_IDS)
    print(" ", OUT_SILVER_TRN)
    print("Silver counts:", silver["y"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()