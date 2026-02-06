import json
import re
import hashlib
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DELETED_MARKERS = {"[deleted]", "[removed]"}


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def normalize_text(title: str, body: str) -> str:
    title = "" if not isinstance(title, str) else title.strip()
    body = "" if not isinstance(body, str) else body.strip()
    return (title + "\n\n" + body).strip()

def filter_basic(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    body_lower = df["body"].str.strip().str.lower()
    df = df[~body_lower.isin(DELETED_MARKERS)]

    df = df[(df["title"].str.strip() != "") & (df["body"].str.strip() != "")]
    df["text"] = [normalize_text(t, b) for t, b in zip(df["title"], df["body"])]

    # vectorized word count: count tokens separated by whitespace
    wc = df["text"].astype(str).str.count(r"\S+")
    df = df[wc >= min_words]

    return df

def build_ra(ra_csv: Path, min_words: int) -> pd.DataFrame:
    ra = pd.read_csv(ra_csv)
    out = pd.DataFrame({
        "domain": "ra",
        "source_subreddit": "relationship_advice",
        "native_id": ra["id"].astype(str),
        "title": ra["title"],
        "body": ra["body"],
        "source_url": ra["url"].fillna("").astype(str),
        "created_utc": ra["timestamp"] if "timestamp" in ra.columns else ra["created"],
    })
    out = filter_basic(out, min_words)

    # doc_id: prefer native id
    out["doc_id"] = out["native_id"].apply(lambda x: f"ra_{stable_hash(x)}")
    return out[["doc_id","domain","source_subreddit","native_id", "title","body","text","source_url","created_utc"]]

def build_confessions(conf_csv: Path, min_words: int) -> pd.DataFrame:
    conf = pd.read_csv(conf_csv)
    out = pd.DataFrame({
        "domain": "confession",
        "source_subreddit": conf["subreddit.name"].fillna("").astype(str),
        "native_id": conf["id"].astype(str),
        "title": conf["title"],
        "body": conf["selftext"],
        "source_url": conf["url"].fillna("").astype(str),
        "created_utc": conf["created_utc"],
    })
    out = filter_basic(out, min_words)

    out["doc_id"] = out["native_id"].apply(lambda x: f"conf_{stable_hash(x)}")
    return out[["doc_id","domain","source_subreddit","native_id", "title","body","text","source_url","created_utc"]]

def build_aita_sqlite(aita_sqlite: Path, min_words: int) -> pd.DataFrame:
    con = sqlite3.connect(str(aita_sqlite))
    q = "SELECT id, submission_id, title, selftext, created_utc, permalink, score FROM submission;"
    sub = pd.read_sql_query(q, con)
    con.close()

    out = pd.DataFrame({
        "domain": "aita",
        "source_subreddit": "AmItheAsshole",
        "native_id": sub["submission_id"].fillna(sub["id"]).astype(str),
        "title": sub["title"],
        "body": sub["selftext"],
        "source_url": sub["permalink"].fillna("").astype(str),
        "created_utc": sub["created_utc"],
    })
    out = filter_basic(out, min_words)

    out["doc_id"] = out["native_id"].apply(lambda x: f"aita_{stable_hash(x)}")
    return out[["doc_id","domain","source_subreddit","native_id", "title","body","text","source_url","created_utc"]]

def main():
    cfg_path = ROOT / "data_new" / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    min_words = int(cfg["min_words"])

    ra_path = ROOT / cfg["paths"]["ra"]
    conf_path = ROOT / cfg["paths"]["confession"]
    aita_path = ROOT / cfg["paths"]["aita"]

    ra = build_ra(ra_path, min_words)
    conf = build_confessions(conf_path, min_words)
    aita = build_aita_sqlite(aita_path, min_words)

    df = pd.concat([ra, conf, aita], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["domain", "native_id"])
    df = df.drop_duplicates(subset=["doc_id"])
    after = len(df)

    # 1) 主表：完整留档
    out_master = ROOT / "data_new" / "dataset_raw.csv"
    df.to_csv(out_master, index=False, encoding="utf-8-sig")

    print("Wrote master:", out_master)
    print(f"Dedup: {before} -> {after}")
    
if __name__ == "__main__":
    main()