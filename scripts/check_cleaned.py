# import pandas as pd
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]

# raw = pd.read_csv(ROOT/"data_new/dataset_raw.csv", usecols=["doc_id","domain","text"])
# cln = pd.read_csv(ROOT/"data_new/dataset_cleaned.csv", usecols=["doc_id","domain","text"])

# raw_ids = set(raw["doc_id"])
# cln_ids = set(cln["doc_id"])

# print("Raw rows:", len(raw), "Cleaned rows:", len(cln))
# print("ID overlap:", len(raw_ids & cln_ids))
# print("Raw-only:", len(raw_ids - cln_ids), "Cleaned-only:", len(cln_ids - raw_ids))

# # quick artifact check (AITA only)
# aita_raw = raw[raw["domain"]=="aita"]["text"].head(200).astype(str).str.contains(r"\b(YTA|NTA|ESH|NAH|INFO|AITA|WIBTA|TL;DR|Edit:|Update:)\b", regex=True).mean()
# aita_cln = cln[cln["domain"]=="aita"]["text"].head(200).astype(str).str.contains(r"\b(YTA|NTA|ESH|NAH|INFO|AITA|WIBTA|TL;DR|Edit:|Update:)\b", regex=True).mean()
# print("AITA artifact-hit rate (first 200): raw=", round(aita_raw,3), "cleaned=", round(aita_cln,3))

# import pandas as pd; df=pd.read_csv('data_new/dataset_raw.csv', nrows=1); print(list(df.columns))
# scripts/audit_usernames.py
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data_new" / "dataset_raw.csv"
OUT_TXT = ROOT / "data_new" / "username_audit.txt"
OUT_CSV = ROOT / "data_new" / "username_hits_sample.csv"

# Reddit username patterns (heuristic)
RE_U_SLASH = re.compile(r"(?i)\b/?u/[A-Za-z0-9_-]{3,30}\b")
RE_USER_URL = re.compile(r"(?i)\breddit\.com/(?:u|user)/[A-Za-z0-9_-]{3,30}\b")

# Optional: looks like “username: xxx” in text (rare, but check)
RE_LABEL = re.compile(r"(?i)\b(?:user(?:name)?|author)\s*[:=]\s*[A-Za-z0-9_-]{3,30}\b")

CANDIDATE_USER_COLS = {
    "author", "username", "user", "user_name", "created_by", "poster", "op",
    "author_fullname", "author_name", "account"
}

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Not found: {RAW_PATH}")

    # Read header only to inspect columns
    cols = pd.read_csv(RAW_PATH, nrows=0).columns.tolist()
    col_lower = {c.lower(): c for c in cols}
    found_user_cols = [col_lower[c] for c in col_lower.keys() if c in CANDIDATE_USER_COLS]

    # Scan text column for username-like patterns
    hits = []
    total_rows = 0
    total_text_nonempty = 0
    count_u_slash = 0
    count_user_url = 0
    count_label = 0

    usecols = cols
    if "text" not in cols:
        # fallback: try title/body if text missing
        pass

    for chunk in pd.read_csv(RAW_PATH, chunksize=20000, low_memory=False, na_filter=False):
        total_rows += len(chunk)

        if "text" in chunk.columns:
            text_series = chunk["text"].fillna("").astype(str)
        else:
            # fallback: build text from title/body
            title = chunk["title"].fillna("").astype(str) if "title" in chunk.columns else ""
            body = chunk["body"].fillna("").astype(str) if "body" in chunk.columns else ""
            text_series = (title + "\n\n" + body).astype(str)

        nonempty_mask = text_series.str.len() > 0
        total_text_nonempty += int(nonempty_mask.sum())

        # count pattern hits (document-level: whether a post contains >=1 match)
        u_mask = text_series.str.contains(RE_U_SLASH)
        url_mask = text_series.str.contains(RE_USER_URL)
        label_mask = text_series.str.contains(RE_LABEL)

        count_u_slash += int(u_mask.sum())
        count_user_url += int(url_mask.sum())
        count_label += int(label_mask.sum())

        # collect a small sample of rows that contain hits
        hit_mask = u_mask | url_mask | label_mask
        if hit_mask.any():
            sub = chunk.loc[hit_mask, ["doc_id", "domain", "source_subreddit"]].copy() if set(["doc_id","domain","source_subreddit"]).issubset(chunk.columns) else chunk.loc[hit_mask].copy()
            sub["hit_u_slash"] = u_mask[hit_mask].values
            sub["hit_user_url"] = url_mask[hit_mask].values
            sub["hit_label"] = label_mask[hit_mask].values

            # add a short snippet for manual inspection
            sub_text = text_series[hit_mask].str.replace("\n", " ", regex=False).str.slice(0, 220)
            sub["text_snippet"] = sub_text.values

            hits.append(sub.head(200))  # cap per chunk

        # hard cap total sample rows to keep file small
        if sum(len(x) for x in hits) > 2000:
            break

    sample_df = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame()

    # Write report
    lines = []
    lines.append(f"RAW_PATH: {RAW_PATH}")
    lines.append(f"Total rows scanned: {total_rows}")
    lines.append(f"Non-empty text rows: {total_text_nonempty}")
    lines.append("")
    lines.append("=== Column audit (possible username fields) ===")
    if found_user_cols:
        lines.append("Found candidate user-related columns:")
        for c in found_user_cols:
            lines.append(f"  - {c}")
    else:
        lines.append("No obvious username/author columns found.")
    lines.append("")
    lines.append("=== Text pattern audit (heuristics) ===")
    lines.append(f"Rows containing /u/ or u/: {count_u_slash}")
    lines.append(f"Rows containing reddit.com/user/: {count_user_url}")
    lines.append(f"Rows containing username/author labels: {count_label}")
    lines.append("")
    lines.append("Note: These are heuristic matches; false positives/negatives are possible.")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    if not sample_df.empty:
        sample_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"Wrote sample hits: {OUT_CSV}")
    else:
        print("No username-like patterns found in scanned portion (or dataset is clean).")

    print(f"Wrote report: {OUT_TXT}")

if __name__ == "__main__":
    main()