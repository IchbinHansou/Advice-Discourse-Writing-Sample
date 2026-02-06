from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


LABELS = ("ADVICE", "STORY")


# -----------------------------
# Utilities
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    """
    Minimal tokenizer: keep only English-ish word tokens.
    (Your dataset is Reddit English-heavy; this keeps the pipeline stable/offline.)
    """
    if text is None:
        return []
    t = str(text).lower()
    toks = _WORD_RE.findall(t)
    return toks


def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_label(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s in LABELS else None


# -----------------------------
# Interpolated n-gram LM
# -----------------------------
@dataclass
class NgramLM:
    order: int = 3
    alpha: float = 0.1
    # interpolation weights by order (index 1..N). Unigram uses 1.0.
    # Default: rely more on higher-order; still back off with some weight.
    lambdas: Tuple[float, ...] = (1.0, 0.75, 0.75, 0.75)

    vocab: set = None
    # counts[n][context_tuple] -> Counter(next_word)
    counts: Dict[int, Dict[Tuple[str, ...], Counter]] = None
    # context_totals[n][context_tuple] -> total count of next tokens after that context
    context_totals: Dict[int, Counter] = None

    def __post_init__(self):
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if len(self.lambdas) < self.order + 1:
            # pad lambdas up to needed order
            pad = [self.lambdas[-1]] * (self.order + 1 - len(self.lambdas))
            self.lambdas = tuple(list(self.lambdas) + pad)
        self.vocab = set()
        self.counts = {n: defaultdict(Counter) for n in range(1, self.order + 1)}
        self.context_totals = {n: Counter() for n in range(2, self.order + 1)}  # totals for contexts length n-1

        # reserve special tokens
        self.vocab.update({"<s>", "</s>", "<unk>"})

    def fit_texts(self, texts: Iterable[str]):
        for text in texts:
            toks = tokenize(safe_text(text))
            self._update_one(toks)

        # Finalize: make sure <unk> is in vocab even if never seen
        self.vocab.add("<unk>")
        return self

    def _update_one(self, toks: List[str]):
        # add boundaries
        bos = ["<s>"] * (self.order - 1)
        seq = bos + toks + ["</s>"]

        # update vocab
        self.vocab.update(toks)

        # update n-gram counts for n=1..order
        for i in range(self.order - 1, len(seq)):
            for n in range(1, self.order + 1):
                ctx_len = n - 1
                if i - ctx_len < 0:
                    continue
                context = tuple(seq[i - ctx_len:i]) if ctx_len > 0 else tuple()
                w = seq[i]
                self.counts[n][context][w] += 1
                if n >= 2:
                    self.context_totals[n][context] += 1

    def _map_unk(self, tok: str) -> str:
        return tok if tok in self.vocab else "<unk>"

    def prob(self, tok: str, context: Tuple[str, ...]) -> float:
        """
        Interpolated add-alpha with recursive backoff:
          p_n = λ * p_ml_n + (1-λ) * p_{n-1}
        where p_ml_n uses add-alpha smoothing.
        """
        tok = self._map_unk(tok)

        n = len(context) + 1
        if n <= 1:
            # unigram
            c = self.counts[1][tuple()].get(tok, 0)
            total = sum(self.counts[1][tuple()].values())
            V = max(1, len(self.vocab))
            return (c + self.alpha) / (total + self.alpha * V)

        # truncate context to max order-1
        if len(context) > self.order - 1:
            context = context[-(self.order - 1):]
        n = len(context) + 1

        # MLE with add-alpha
        ctx = tuple(self._map_unk(t) for t in context)
        c_ctx_tok = self.counts[n][ctx].get(tok, 0)
        c_ctx = self.context_totals[n][ctx]
        V = max(1, len(self.vocab))
        p_ml = (c_ctx_tok + self.alpha) / (c_ctx + self.alpha * V)

        # backoff to shorter context
        lam = float(self.lambdas[n]) if n < len(self.lambdas) else 0.75
        if n == 2:
            p_bo = self.prob(tok, tuple())
        else:
            p_bo = self.prob(tok, ctx[1:])

        return lam * p_ml + (1.0 - lam) * p_bo

    def cross_entropy_bits_per_tok(self, text: str) -> float:
        toks = tokenize(safe_text(text))
        if not toks:
            return float("nan")

        bos = ["<s>"] * (self.order - 1)
        seq = bos + toks + ["</s>"]

        nll = 0.0
        count = 0
        for i in range(self.order - 1, len(seq)):
            ctx = tuple(seq[i - (self.order - 1):i])
            w = seq[i]
            p = self.prob(w, ctx)
            # guard (should not happen because smoothing)
            p = max(p, 1e-12)
            nll += -math.log(p, 2)
            count += 1

        return nll / max(1, count)


# -----------------------------
# Data loading
# -----------------------------
def load_gold(data: Path, condition: str) -> pd.DataFrame:
    """
    condition:
      - raw  -> gold_raw_norm_FROZEN.csv expects columns: doc_id, domain, text, gold_label
      - clean-> gold_clean_v1_FROZEN.csv (binary only) expects: doc_id, domain, text_clean_v1, gold_label
    """
    if condition == "raw":
        p = data / "gold_raw_norm_FROZEN.csv"
        df = pd.read_csv(p, low_memory=False)
        text_col = "text"
    else:
        p = data / "gold_clean_v1_FROZEN.csv"
        if not p.exists():
            # try to create it by merging raw ids + clean text file (if present)
            raw = pd.read_csv(data / "gold_raw_norm_FROZEN.csv", low_memory=False)
            clean = pd.read_csv(data / "gold_clean_v1_FROZEN.csv", low_memory=False)
            clean = clean[clean["gold_label"].isin(list(LABELS))].copy()
            merged = raw[["doc_id", "domain", "gold_label"]].merge(
                clean[["doc_id", "text_clean_v1"]],
                on="doc_id",
                how="left",
            )
            if merged["text_clean_v1"].isna().any():
                miss = int(merged["text_clean_v1"].isna().sum())
                raise RuntimeError(
                    f"Missing cleaned text for {miss} gold rows. "
                    f"Create gold_clean_v1_FROZEN.csv first or fix doc_id alignment."
                )
            merged.to_csv(p, index=False)
            df = merged
        else:
            df = pd.read_csv(p, low_memory=False)
        text_col = "text_clean_v1"

    df["gold_label"] = df["gold_label"].map(normalize_label)
    df = df[df["gold_label"].isin(LABELS)].copy()
    df["domain"] = df["domain"].astype(str).str.lower()
    df["doc_id"] = df["doc_id"].astype(str)

    return df, text_col


def load_silver(data: Path, condition: str) -> pd.DataFrame:
    """
    condition:
      - raw  -> silver_train_raw_norm_FROZEN.csv expects: doc_id, domain, text, silver_label
      - clean-> silver_train_clean_v1_FROZEN.csv expects: doc_id, domain, text_clean_v1, silver_label
    """
    if condition == "raw":
        p = data / "silver_train_raw_norm_FROZEN.csv"
        df = pd.read_csv(p, low_memory=False)
        text_col = "text"
    else:
        p = data / "silver_train_clean_v1_FROZEN.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. If you don't have cleaned silver yet, run scripts/make_clean_v1_for_splits.py "
                f"or switch to --condition raw."
            )
        df = pd.read_csv(p, low_memory=False)
        text_col = "text_clean_v1"

    df["silver_label"] = df["silver_label"].map(normalize_label)
    df = df[df["silver_label"].isin(LABELS)].copy()
    df["domain"] = df["domain"].astype(str).str.lower()
    df["doc_id"] = df["doc_id"].astype(str)

    return df, text_col


# -----------------------------
# Training + scoring
# -----------------------------
def train_lms(train_df: pd.DataFrame, text_col: str, label_col: str, order: int, alpha: float) -> Dict[str, NgramLM]:
    lms: Dict[str, NgramLM] = {}
    for lab in LABELS:
        sub = train_df[train_df[label_col] == lab].copy()
        lm = NgramLM(order=order, alpha=alpha)
        lm.fit_texts(sub[text_col].astype(str).tolist())
        lms[lab] = lm
    return lms


def score_df(df: pd.DataFrame, text_col: str, lms: Dict[str, NgramLM], label_col: str | None = None) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        text = safe_text(r.get(text_col, ""))
        h_adv = lms["ADVICE"].cross_entropy_bits_per_tok(text)
        h_sty = lms["STORY"].cross_entropy_bits_per_tok(text)
        delta = h_sty - h_adv  # >0 => advice LM fits better
        pred = "ADVICE" if h_adv < h_sty else "STORY"

        out = {
            "doc_id": str(r.get("doc_id", "")),
            "domain": str(r.get("domain", "")),
            "H_advice": h_adv,
            "H_story": h_sty,
            "delta_H": delta,
            "pred_by_minH": pred,
        }
        if label_col is not None and label_col in r:
            out["label"] = normalize_label(r[label_col])
        rows.append(out)

    scored = pd.DataFrame(rows)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=3, choices=[2,3,4], help="n-gram order (default: 3)")
    ap.add_argument("--alpha", type=float, default=0.1, help="add-alpha smoothing (default: 0.1)")
    ap.add_argument("--condition", type=str, default="clean", choices=["raw", "clean"], help="text condition")
    ap.add_argument("--train_source", type=str, default="silver", choices=["silver", "gold"], help="train LMs on silver or gold")
    args = ap.parse_args()

    root = repo_root()
    data = root / "data_new"

    gold_df, gold_text_col = load_gold(data, args.condition)
    silver_df, silver_text_col = load_silver(data, args.condition)

    if args.train_source == "silver":
        train_df = silver_df
        text_col = silver_text_col
        label_col = "silver_label"
    else:
        train_df = gold_df
        text_col = gold_text_col
        label_col = "gold_label"

    # Train class-conditional LMs
    lms = train_lms(train_df, text_col=text_col, label_col=label_col, order=args.order, alpha=args.alpha)

    # Save LMs (primary names requested by your Day8 plan)
    with open(data / "lm_advice.pkl", "wb") as f:
        pickle.dump(lms["ADVICE"], f)
    with open(data / "lm_story.pkl", "wb") as f:
        pickle.dump(lms["STORY"], f)

    # Score gold
    gold_scored = score_df(gold_df, text_col=gold_text_col, lms=lms, label_col="gold_label")
    gold_scored.to_csv(data / "surprisal_scores_gold.csv", index=False)

    # Score combined (silver + gold), keeping a source tag
    sil_scored = score_df(silver_df, text_col=silver_text_col, lms=lms, label_col="silver_label")
    sil_scored["source"] = "silver"
    gold_scored2 = gold_scored.copy()
    gold_scored2["source"] = "gold"

    all_scored = pd.concat([gold_scored2, sil_scored], ignore_index=True)
    all_scored.to_csv(data / "surprisal_scores.csv", index=False)

    # Quick sanity: accuracy on gold by LM likelihood
    ok = gold_scored.dropna(subset=["label", "pred_by_minH"])
    if len(ok) > 0:
        acc = (ok["label"] == ok["pred_by_minH"]).mean()
        print(f"[sanity] gold accuracy by min cross-entropy: {acc:.3f} (n={len(ok)})")

    print("Saved:")
    print(" -", data / "lm_advice.pkl")
    print(" -", data / "lm_story.pkl")
    print(" -", data / "surprisal_scores_gold.csv")
    print(" -", data / "surprisal_scores.csv")
    print(f"Config: order={args.order} alpha={args.alpha} condition={args.condition} train_source={args.train_source}")


if __name__ == "__main__":
    main()
