# from pathlib import Path
# import pandas as pd

# ROOT = Path(__file__).resolve().parents[1]  # 项目根目录
# inp = ROOT / "data_new" / "gold_to_annotate.csv"
# out_csv = ROOT / "data_new" / "gold_to_annotate.csv"

# df = pd.read_csv(inp, encoding="utf-8")

# # 把“ ”换成" "，避免 WPS/GBK 误解码
# REPL = {
#     "\u2019": "'",  # ’
#     "\u2018": "'",  # ‘
#     "\u201c": '"',  # “
#     "\u201d": '"',  # ”
#     "\u2014": "-",  # —
# }
# for col in ["title", "body", "text"]:
#     if col in df.columns:
#         s = df[col].astype(str)
#         for k, v in REPL.items():
#             s = s.str.replace(k, v, regex=False)
#         df[col] = s

# df.to_csv(out_csv, index=False, encoding="utf-8-sig")  # 带 BOM，给 WPS/Excel 识别 UTF-8
# print("Wrote:", out_csv)



# import pandas as pd
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# GOLD = ROOT / "data_new" / "gold_to_annotate.csv"

# df = pd.read_csv(GOLD, dtype={"doc_id":"string","domain":"string","text":"string"})

# # 1) 每域数量
# print("Counts by domain:")
# print(df["domain"].value_counts(dropna=False))
# print("Total:", len(df))

# # 2) 每域长度（字符数）分布：P50/P75/P90/P95 + max
# df["n_chars"] = df["text"].fillna("").str.len()
# q = [0.5, 0.75, 0.9, 0.95]
# stats = df.groupby("domain")["n_chars"].quantile(q).unstack()
# stats["max"] = df.groupby("domain")["n_chars"].max()
# print("\nLength quantiles by domain (chars):")
# print(stats.round(0).astype(int))

# # 3) 每域 >800 字符有多少
# print("\n>800 chars by domain:")
# print(df[df["n_chars"] > 800]["domain"].value_counts())




# import pandas as pd
# from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]
# GOLD = ROOT / "data_new" / "gold_to_annotate.csv"
# df = pd.read_csv(GOLD, dtype={"doc_id":"string","domain":"string","text":"string"})
# df["n_chars"] = df["text"].fillna("").str.len()
# ra_top = df[df["domain"]=="ra"].sort_values("n_chars", ascending=False).head(10)
# print(ra_top[["doc_id","n_chars"]])




# import pandas as pd

# gold = pd.read_csv(OUT_GOLD_ANN, dtype={"doc_id":"string","domain":"string"})
# gold["n_chars"] = gold["text"].astype(str).str.len()

# # 用你刚才打印出来的 p95（也可以直接用 quantile 现算一次）
# p95 = gold.groupby("domain")["n_chars"].quantile(0.95).to_dict()
# print("p95 by domain:", p95)

# gold["p95"] = gold["domain"].map(p95)
# outliers = gold[gold["n_chars"] > gold["p95"]].copy()

# print("\nOutliers count by domain:")
# print(outliers["domain"].value_counts())

# print("\nAll outliers (sorted):")
# print(outliers.sort_values(["domain","n_chars"], ascending=[True, False])[["doc_id","domain","n_chars"]].to_string(index=False))
# '''
# '''
# import re
# from pathlib import Path
# import pandas as pd

# ROOT = Path(__file__).resolve().parents[1]
# RAW_PATH = ROOT / "data_new" / "dataset_raw.csv"
# CLN_PATH = ROOT / "data_new" / "dataset_cleaned.csv"

# OUT_RAW_LEN = ROOT / "data_new" / "length_index_raw.csv"
# OUT_CLN_LEN = ROOT / "data_new" / "length_index_cleaned.csv"

# WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

# def count_words(s: str) -> int:
#     if not isinstance(s, str) or not s:
#         return 0
#     return len(WORD_RE.findall(s))

# def build(in_path: Path, out_path: Path, text_col="text"):
#     usecols = ["doc_id", "domain", text_col]
#     first = True
#     for chunk in pd.read_csv(in_path, usecols=usecols, chunksize=50000, dtype={"doc_id":"string","domain":"string"}):
#         t = chunk[text_col].fillna("")
#         chunk["n_chars"] = t.str.len().astype("int32")
#         chunk["n_words"] = t.map(count_words).astype("int32")
#         chunk = chunk[["doc_id","domain","n_words","n_chars"]]
#         chunk.to_csv(out_path, index=False, mode="w" if first else "a", header=first, encoding="utf-8-sig")
#         first = False
#     print("Wrote:", out_path)

# if __name__ == "__main__":
#     build(RAW_PATH, OUT_RAW_LEN, text_col="text")
#     build(CLN_PATH, OUT_CLN_LEN, text_col="text")
#     '''

# '''
# import pandas as pd

# # 1) 改成你的两个大表路径（raw / cleaned）
# RAW_PATH = r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new\dataset_raw.csv"
# CLEANED_PATH = r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new\dataset_cleaned.csv"

# def show_headers(path: str, name: str):
#     print(f"\n===== {name} =====")
#     try:
#         cols = pd.read_csv(path, nrows=0).columns.tolist()
#         print(f"n_cols = {len(cols)}")
#         print("columns:")
#         for c in cols:
#             print(" -", c)
#     except Exception as e:
#         print("ERROR:", repr(e))

# if __name__ == "__main__":
#     show_headers(RAW_PATH, "RAW")
#     show_headers(CLEANED_PATH, "CLEANED")


# import pandas as pd
# from pathlib import Path
# path = Path("C:/Users/PC/OneDrive/桌面/Saarland Writing Sample/data_new/dataset_raw.csv")
# df = pd.read_csv(r"dataset_raw.csv")
# print(df["domain"].value_counts())

# import pandas as pd

# # 读取您上传的文件 (根据您的文件名调整)
# input_file = "C:\Users\PC\OneDrive\桌面\Saarland Writing Sample\data_new\gold_to_annotate_V4_final.xlsx"
# df = pd.read_csv(input_file)

# def get_gold_label(row):
#     text = str(row['text']).lower()
#     domain = row['domain']
#     doc_id = row['doc_id']
    
#     # --- 1. 人工专家修正 (Expert Overrides) ---
#     # 这些是规则难以覆盖的特殊情况，已人工确认
#     overrides = {
#         'ra_37f643ee740b8722': 'STORY',   # Venting
#         'ra_f44dc387afb17b7a': 'STORY',   # Narrative ending
#         'ra_60fe2eb3a45d9285': 'STORY',   # Update post
#         'ra_6ba4d46ce6a0e1c2': 'STORY',   # Explicit Rant
#         'ra_776300fc6e7550b5': 'ADVICE',  # "Share thoughts" (Implicit request)
#         'ra_76d57bddc703718d': 'ADVICE',  # "Want to know what I'm doing wrong"
#         'conf_de9442cfd1d792cf': 'STORY', # Statement of desire ("I just want to be ok")
#         'conf_9e62ad519ff1b11b': 'STORY', # Narrative
#         'conf_9cfb90d370208934': 'ADVICE', # Explicit "Someone help me"
#         'conf_f835498417b2eda5': 'ADVICE', # Explicit "Help."
#         'conf_f6e6ede64cb282f7': 'ADVICE', # Explicit "Please tell me why"
#     }
    
#     if doc_id in overrides:
#         return overrides[doc_id]

#     # --- 2. AITA 域 ---
#     # 寻求评判 (Verdict) 属于 Decision Support -> ADVICE
#     if domain == 'aita':
#         return "ADVICE"
    
#     # --- 3. RA (Relationship Advice) 域 ---
#     if domain == 'ra':
#         # 优先级 A: 明确的建议信号
#         advice_kw = ["advice", "help", "what do", "how to", "how do", "thoughts", "opinion", "am i", "is it", "should i"]
#         if any(k in text for k in advice_kw): return "ADVICE"
#         if "?" in text: return "ADVICE"
        
#         # 优先级 B: 明确的故事/宣泄信号 (如果没求助的话)
#         story_signals = ["vent", "rant", "just sharing", "no advice", "update", "off my chest"]
#         if any(s in text for s in story_signals): return "STORY"
            
#         # 默认回退: RA 主要是求助
#         return "ADVICE"

#     # --- 4. Confession 域 ---
#     if domain == 'confession':
#         # 优先级 A: 隐藏的求助 (ADVICE)
#         # 必须是明确的 "Help me" 或 "What should I do" 类强信号
#         advice_kw = ["what should i do", "help me", "need advice", "am i the asshole", "aita", "is it wrong", "how do i stop", "please tell me"]
#         if any(k in text for k in advice_kw): return "ADVICE"
        
#         # 上下文判断: 既有 "advice" 词汇又有问号
#         if "advice" in text and "?" in text: return "ADVICE"

#         # 默认回退: Confession 主要是故事
#         return "STORY"
    
#     return "MIXED"

# # 应用标注逻辑
# df['gold_label'] = df.apply(get_gold_label, axis=1)

# # 添加备注 (Notes)
# def add_notes(row):
#     # 标记人工修正的行，方便您复查
#     if row['doc_id'] in ['ra_37f643ee740b8722', 'ra_f44dc387afb17b7a', 'ra_60fe2eb3a45d9285']:
#         return "Expert Correction: Narrative/Vent/Update"
#     if row['doc_id'] in ['conf_9cfb90d370208934', 'conf_f835498417b2eda5']:
#         return "Expert Correction: Explicit Help Request"
#     return ""

# df['notes'] = df.apply(add_notes, axis=1)

# # 保存文件
# output_filename = "annotated_gold_full.csv"
# df.to_csv(output_filename, index=False)
# print(f"✅ 完成！文件已保存为: {output_filename}")
# print("前 5 条预览:")
# print(df[['doc_id', 'domain', 'gold_label', 'notes']].head())

import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data_new"

INP = DATA_DIR / "dataset_raw_submissions_only.csv"

df = pd.read_csv(str(INP), low_memory=False)

print("rows:", len(df))
print("cols:", list(df.columns))

def infer_is_comment(d):
    # 1) 最可靠：fullname/name 以 t1_ 开头是 comment，t3_ 是 submission
    for c in ["name", "fullname", "full_id"]:
        if c in d.columns:
            s = d[c].astype(str)
            return s.str.startswith("t1_")

    # 2) 常见 pushshift: comments 有 parent_id/link_id
    if "parent_id" in d.columns and "link_id" in d.columns:
        pid = d["parent_id"].astype(str)
        lid = d["link_id"].astype(str)
        return pid.str.startswith("t1_") | pid.str.startswith("t3_") | lid.str.startswith("t3_")

    # 3) 兜底：没有 title 且有 body 的，极大概率是 comment
    if "title" in d.columns:
        title_empty = d["title"].isna() | (d["title"].astype(str).str.strip() == "")
        # 如果你有 body/selftext，comments 更可能只有 body
        body_col = "body" if "body" in d.columns else ("selftext" if "selftext" in d.columns else None)
        if body_col:
            body_nonempty = d[body_col].notna() & (d[body_col].astype(str).str.strip() != "")
            return title_empty & body_nonempty
        return title_empty

    return pd.Series([False] * len(d))

is_comment = infer_is_comment(df)
print("comment-like rows:", int(is_comment.sum()))

if "domain" in df.columns:
    print("\ncomment-like by domain:")
    print(df[is_comment]["domain"].value_counts())

# 抽样看看 comment 行
sample = df[is_comment].head(3)
if len(sample) > 0:
    cols_show = [c for c in ["doc_id","domain","source_subreddit","name","fullname","title","body","selftext","text"] if c in df.columns]
    print("\nSAMPLE comment-like rows:")
    print(sample[cols_show].to_string(index=False))