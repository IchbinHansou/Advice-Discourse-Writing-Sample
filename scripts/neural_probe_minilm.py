from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

LABELS = ["ADVICE", "STORY"]

def load_gold(data_dir: Path) -> pd.DataFrame:
    # Prefer the file that has both raw + cleaned text
    p = data_dir / "gold_clean_v1_FROZEN.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Put gold_clean_v1_FROZEN.csv under {data_dir}")
    df = pd.read_csv(p)
    need = {"doc_id", "domain", "gold_label", "text", "text_clean_v1"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{p.name} missing columns: {sorted(miss)}")
    df["doc_id"] = df["doc_id"].astype(str)
    df["domain"] = df["domain"].astype(str)
    df = df[df["gold_label"].isin(LABELS)].copy()
    return df

def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                batch_size: int = 64) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing sentence-transformers. Install via:\n"
            "  pip install -U sentence-transformers torch\n"
        ) from e

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb

def leave_one_domain_out(df: pd.DataFrame, X: np.ndarray, text_tag: str) -> dict:
    results = {"text_condition": text_tag, "splits": {}, "macro_f1_avg": None}
    f1s = []

    for heldout in sorted(df["domain"].unique()):
        train_idx = df["domain"] != heldout
        test_idx = df["domain"] == heldout

        X_train, y_train = X[train_idx.values], df.loc[train_idx, "gold_label"].values
        X_test, y_test = X[test_idx.values], df.loc[test_idx, "gold_label"].values

        clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        macro = f1_score(y_test, y_pred, labels=LABELS, average="macro")
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)

        results["splits"][heldout] = {
            "n_test": int(test_idx.sum()),
            "macro_f1": float(macro),
            "acc": float(acc),
            "confusion_ADVICE_STORY": cm.tolist(),
        }
        f1s.append(macro)

    results["macro_f1_avg"] = float(np.mean(f1s)) if f1s else None
    return results

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data_new"
    if not data_dir.exists():
        data_dir = root / "data"

    df = load_gold(data_dir)

    for tag, col in [("raw", "text"), ("clean_v1", "text_clean_v1")]:
        cache = data_dir / f"emb_minilm_{tag}.npy"
        if cache.exists():
            X = np.load(cache)
        else:
            texts = df[col].fillna("").astype(str).tolist()
            X = embed_texts(texts)
            np.save(cache, X)

        res = leave_one_domain_out(df, X, tag)

        out_json = data_dir / f"results_neural_probe_{tag}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        # also save per-example preds for later error analysis
        # Here: train a single model on all data except each heldout domain and save preds on heldout
        preds_rows = []
        for heldout in sorted(df["domain"].unique()):
            train_idx = df["domain"] != heldout
            test_idx = df["domain"] == heldout

            clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
            clf.fit(X[train_idx.values], df.loc[train_idx, "gold_label"].values)
            y_pred = clf.predict(X[test_idx.values])

            part = df.loc[test_idx, ["doc_id", "domain", "gold_label"]].copy()
            part["pred_label"] = y_pred
            part["text_condition"] = tag
            preds_rows.append(part)

        preds = pd.concat(preds_rows, ignore_index=True)
        out_csv = data_dir / f"preds_neural_probe_{tag}_gold.csv"
        preds.to_csv(out_csv, index=False)

        print(f"[OK] Saved: {out_json}")
        print(f"[OK] Saved: {out_csv}")
        print(f"[OK] macro_f1_avg (LODO): {res['macro_f1_avg']:.4f}")

if __name__ == "__main__":
    main()