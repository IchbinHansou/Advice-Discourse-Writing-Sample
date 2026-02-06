from __future__ import annotations

from pathlib import Path
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    root = repo_root()
    data_dir = root / "data_new"

    # choose one merged file:
    # - for proper TFIDF raw vs TFIDF cleaned, use gold267_merged_tfidf_raw_clean.csv
    # - otherwise keep using your existing gold267_merged_preds_raw_clean.csv
    merged_path = data_dir / "gold267_merged_tfidf_raw_clean.csv"
    if not merged_path.exists():
        merged_path = data_dir / "gold267_merged_preds_raw_clean.csv"

    df = pd.read_csv(merged_path, low_memory=False)

    # choose which predictions to analyze
    # change this to "pred_clean" if you want cleaned errors
    pred_col = "pred_raw"
    if pred_col not in df.columns:
        # fallback
        pred_col = "pred_label" if "pred_label" in df.columns else df.columns[df.columns.str.contains("pred")][0]

    needed = ["doc_id", "domain", "gold_label", pred_col]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {merged_path.name}. Columns={list(df.columns)}")

    err = df[df["gold_label"].astype(str) != df[pred_col].astype(str)].copy()
    print(f"errors available: {len(err)} (using {pred_col})")

    # balanced sample by domain if possible
    sample_n = 30
    parts = []
    if "domain" in err.columns and err["domain"].nunique() >= 2:
        domains = sorted(err["domain"].astype(str).unique().tolist())
        per = max(1, sample_n // len(domains))
        for dom in domains:
            sub = err[err["domain"].astype(str) == dom]
            parts.append(sub.sample(n=min(per, len(sub)), random_state=42))
        sampled = pd.concat(parts, ignore_index=True)
        # top up if we didn't reach 30
        if len(sampled) < sample_n and len(err) > len(sampled):
            remaining = err.drop(sampled.index, errors="ignore")
            add = remaining.sample(n=min(sample_n - len(sampled), len(remaining)), random_state=42)
            sampled = pd.concat([sampled, add], ignore_index=True)
    else:
        sampled = err.sample(n=min(sample_n, len(err)), random_state=42)

    keep_cols = [c for c in [
        "doc_id", "domain", "gold_label", pred_col,
        "text", "text_clean_v1", "text_tail", "category", "is_error"
    ] if c in sampled.columns]

    out = sampled[keep_cols].copy()
    out = out.rename(columns={pred_col: "pred_for_analysis"})

    out_xlsx = data_dir / "error_cases_30.xlsx"
    out_csv = data_dir / "error_cases_30.csv"

    # Try Excel first; if file is open, this often fails -> fall back to csv
    try:
        out.to_excel(out_xlsx, index=False)
        print(f"Saved: {out_xlsx}")
    except PermissionError:
        print("[warn] error_cases_30.xlsx is probably open in Excel. Close it and rerun.")
        out.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"Saved instead: {out_csv}")

    # always save csv too (handy for quick grep)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()