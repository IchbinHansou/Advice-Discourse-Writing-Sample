from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_new"

IN_JSON = DATA / "results_sanity_checks_binary_aligned.json"
OUT_CSV = DATA / "sanity_by_domain_table.csv"
OUT_MD  = DATA / "sanity_by_domain_table.md"

RAW_KEY = "char_ngram_raw"
RAW_KEY_LSA = "lsa_probe_raw"
CLEAN_KEY = "char_ngram_clean_v1_aligned"
CLEAN_KEY_LSA = "lsa_probe_clean_v1_aligned"

def get_domain_rows(obj, model_name, condition_name):
    rows = []
    by_domain = obj.get("by_domain") or {}
    for dom, dom_obj in by_domain.items():
        rows.append({
            "model": model_name,
            "condition": condition_name,
            "domain": dom,
            "n": int(dom_obj["n"]),
            "macro_f1": float(dom_obj["macro_f1"]),
            "accuracy": float(dom_obj["report"]["accuracy"]),
        })
    return rows

def main():
    if not IN_JSON.exists():
        raise FileNotFoundError(f"Missing: {IN_JSON}")

    with open(IN_JSON, "r", encoding="utf-8") as f:
        js = json.load(f)

    rows = []
    rows += get_domain_rows(js[RAW_KEY],   "char_ngram", "raw")
    rows += get_domain_rows(js[CLEAN_KEY], "char_ngram", "clean_v1_aligned")
    rows += get_domain_rows(js[RAW_KEY_LSA],   "lsa_probe", "raw")
    rows += get_domain_rows(js[CLEAN_KEY_LSA], "lsa_probe", "clean_v1_aligned")

    df = pd.DataFrame(rows)

    # Pivot to a compact comparison table
    piv = (
        df.pivot_table(
            index=["domain", "n"],
            columns=["model", "condition"],
            values="macro_f1",
            aggfunc="first",
        )
        .reset_index()
    )

    # Add deltas (clean - raw) for each model
    def delta(model):
        raw = (model, "raw")
        clean = (model, "clean_v1_aligned")
        if raw in piv.columns and clean in piv.columns:
            piv[(model, "delta")] = piv[clean] - piv[raw]

    delta("char_ngram")
    delta("lsa_probe")

    # Sort domains for readability
    piv = piv.sort_values(["domain"])

    # Save CSV (with MultiIndex columns flattened)
    flat_cols = []
    for c in piv.columns:
        if isinstance(c, tuple):
            flat_cols.append("_".join([x for x in c if x]))
        else:
            flat_cols.append(c)
    piv.columns = flat_cols
    piv.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Save Markdown table
    md = piv.to_markdown(index=False)
    OUT_MD.write_text(md, encoding="utf-8")

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_MD)
    print("\nPreview:\n")
    print(md)

if __name__ == "__main__":
    main()
