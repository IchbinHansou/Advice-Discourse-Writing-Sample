import pandas as pd
import re

df = pd.read_csv("dataset_raw.csv", low_memory=False)

if "text" not in df.columns:
    df["title"] = df.get("title", "").fillna("")
    df["body"]  = df.get("body", "").fillna("")
    df["text"]  = df["title"].astype(str) + "\n\n" + df["body"].astype(str)

text_l = df["text"].fillna("").astype(str).str.lower()

bot_patterns = [
    r"\bi am a bot\b",
    r"this action was performed automatically",
    r"contact the moderators of this subreddit",
    r"relationship advice mod team",
    r"\bmoral judgment rule\b",
    r"\bhello\s*/u/",
]
is_bot_like = text_l.str.contains("|".join(bot_patterns), regex=True)

is_comment_blob = text_l.str.match(r"^\s*comment\b")
has_title = df.get("title", "").fillna("").astype(str).str.strip().ne("")

keep = (~is_bot_like) & (~is_comment_blob) & (has_title)

# 把 author 过滤放在这里
if "author" in df.columns:
    author_l = df["author"].fillna("").astype(str).str.lower()
    is_automod = author_l.eq("automoderator") | author_l.str.contains(r"\bbot\b", regex=True)
    keep = keep & (~is_automod)

df_clean = df.loc[keep].copy()
df_clean.to_csv("dataset_raw_submissions_only.csv", index=False)

print("Before:", len(df), "After:", len(df_clean))
print(df_clean["domain"].value_counts(dropna=False).head(20))

print("removed_bot_like:", is_bot_like.sum())
print("removed_comment_blob:", is_comment_blob.sum())
print("missing_title:", (~has_title).sum())
print("removed_total:", (~keep).sum())

bad_ra = df[(df["domain"]=="ra") & (~keep)]
print("bad_ra:", len(bad_ra))
for i, t in enumerate(bad_ra["text"].fillna("").astype(str).head(20), 1):
    print("\n--- bad ra", i, "---")
    print(t[:400])