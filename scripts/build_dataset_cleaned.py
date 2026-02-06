import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_patterns(txt_path: Path):
    pats = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pats.append(re.compile(line, flags=re.IGNORECASE))
    return pats

def basic_markdown_cleanup(s: str) -> str:
    # minimal, safe cleanup
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)          # collapse many blank lines
    s = re.sub(r"[ \t]{2,}", " ", s)          # collapse spaces/tabs
    return s.strip()

def apply_regex_cleanup(s: str, pats) -> str:
    out = s
    for p in pats:
        out = p.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def main():
    cfg = json.loads((ROOT / "data_new" / "run_config.json").read_text(encoding="utf-8"))
    regex_path = ROOT / "spec" / "regex_v1.txt"
    pats = load_patterns(regex_path)

    in_path = ROOT / "data_new" / "dataset_raw.csv"
    out_path = ROOT / "data_new" / "dataset_cleaned.csv"

    # 指定会用到的列的 dtype
    dtype = {
        "doc_id": "string",
        "domain": "string",
        "source_subreddit": "string",
        "title": "string",
        "body": "string",
        "text": "string",
        "source_url": "string",
    }

    reader = pd.read_csv(
        in_path,
        chunksize=20000,
        dtype=dtype,
        low_memory=False,
        na_filter=False,
    )

    first = True
    total = 0

    for chunk in reader:
        # in case of 有些列可能不存在
        if "text" not in chunk.columns:
            chunk["text"] = (chunk.get("title", "") + "\n\n" + chunk.get("body", "")).astype("string")

        chunk["text"] = chunk["text"].fillna("").astype(str).map(basic_markdown_cleanup).map(lambda s: apply_regex_cleanup(s, pats))

        # 写header
        chunk.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False

        total += len(chunk)
        if total % 100000 == 0:
            print(f"Processed {total} rows...")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()