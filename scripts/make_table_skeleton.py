from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    root = repo_root()
    data = root / "data_new"
    inp = data / "results_tfidf_dual_by_domain.json"
    if not inp.exists():
        raise FileNotFoundError(f"Missing: {inp}. Run scripts/run_tfidf_dual_by_domain.py first.")

    obj = json.loads(inp.read_text(encoding="utf-8"))

    rows = []

    def add_rows(block_key: str):
        block = obj[block_key]
        model = block["config"]["model"]
        text_cond = block["text_condition"]
        # overall
        rows.append({
            "model": model,
            "text": text_cond,
            "eval_scope": "overall",
            "macro_f1": block["overall"]["macro_f1"],
            "n_eval": block["overall"]["n"],
        })
        # by domain
        for dom, d in block["by_domain"].items():
            rows.append({
                "model": model,
                "text": text_cond,
                "eval_scope": f"domain={dom}",
                "macro_f1": d["macro_f1"],
                "n_eval": d["n"],
            })

    add_rows("raw")
    add_rows("cleaned_v1")

    df = pd.DataFrame(rows).sort_values(["model", "text", "eval_scope"]).reset_index(drop=True)

    out_csv = data / "table_skeleton.csv"
    out_md = data / "table_skeleton.md"

    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Markdown table
    md = []
    md.append("| Model | Text | Eval scope | Macro-F1 | N |")
    md.append("|---|---|---|---:|---:|")
    for _, r in df.iterrows():
        md.append(f"| {r['model']} | {r['text']} | {r['eval_scope']} | {r['macro_f1']:.4f} | {int(r['n_eval'])} |")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()