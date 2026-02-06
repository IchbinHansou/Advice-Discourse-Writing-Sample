# surprisal_2_updated_fixed.py
# Efficient trigram LM + within-post token shuffles for ΔH
# Outputs: per-post H_orig, H_shuf, ΔH, and Fig1 (ADVICE vs STORY)

import os
import re
import json
import math
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Mann-Whitney U from scipy if available
try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


TOKEN_RE = re.compile(r"[A-Za-z]+")  # keep simple & stable


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


def set_seed(seed: int):
    np.random.seed(seed)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Cliff's delta = ( #x>y - #x<y ) / (nx*ny)
    # O(n log n) approach via sorting would be better, but n is small (gold), so O(n^2) is fine.
    nx, ny = len(x), len(y)
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (nx * ny)


def permutation_test_median_diff(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    # two-sided permutation test on median(x)-median(y)
    rng = np.random.default_rng(seed)
    obs = np.median(x) - np.median(y)
    pooled = np.concatenate([x, y])
    nx = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x_p = pooled[:nx]
        y_p = pooled[nx:]
        stat = np.median(x_p) - np.median(y_p)
        if abs(stat) >= abs(obs):
            count += 1
    # add-one smoothing
    return (count + 1) / (n_perm + 1)


@dataclass
class TrigramLM:
    tri: Counter
    bi: Counter
    uni: Counter
    V: int
    alpha: float
    w1: float
    w2: float
    w3: float

    @classmethod
    def train(cls, token_seqs: List[List[str]],
              alpha: float = 0.1,
              w1: float = 0.1, w2: float = 0.3, w3: float = 0.6) -> "TrigramLM":
        uni = Counter()
        bi = Counter()
        tri = Counter()

        for toks in token_seqs:
            if not toks:
                continue
            # add BOS tokens for trigram context
            seq = ["<s>", "<s>"] + toks + ["</s>"]
            uni.update(seq)
            bi.update(zip(seq[:-1], seq[1:]))
            tri.update(zip(seq[:-2], seq[1:-1], seq[2:]))

        V = len(uni)
        return cls(tri=tri, bi=bi, uni=uni, V=V, alpha=alpha, w1=w1, w2=w2, w3=w3)

    def p1(self, w: str) -> float:
        # Laplace-smoothed unigram
        return (self.uni.get(w, 0) + self.alpha) / (sum(self.uni.values()) + self.alpha * self.V)

    def p2(self, u: str, w: str) -> float:
        # bigram with Laplace smoothing; denom uses unigram count of u (O(1))
        num = self.bi.get((u, w), 0) + self.alpha
        den = self.uni.get(u, 0) + self.alpha * self.V
        return num / den

    def p3(self, v: str, u: str, w: str) -> float:
        # trigram with Laplace smoothing; denom uses bigram count of (v,u) (O(1))
        num = self.tri.get((v, u, w), 0) + self.alpha
        den = self.bi.get((v, u), 0) + self.alpha * self.V
        return num / den

    def cross_entropy_bits(self, text: str) -> float:
        toks = tokenize(text)
        if len(toks) == 0:
            return np.nan
        seq = ["<s>", "<s>"] + toks + ["</s>"]
        # cross-entropy per token (exclude the two BOS, and optionally include </s>)
        # We'll include </s> in the average to stabilize short texts.
        log2p_sum = 0.0
        count = 0

        for i in range(2, len(seq)):
            v, u, w = seq[i - 2], seq[i - 1], seq[i]
            p = self.w3 * self.p3(v, u, w) + self.w2 * self.p2(u, w) + self.w1 * self.p1(w)
            # numerical safety
            p = max(p, 1e-12)
            log2p_sum += math.log2(p)
            count += 1

        return -log2p_sum / max(count, 1)


def load_training_texts(data_dir: str, condition: str) -> List[str]:
    # You can adjust these filenames to match your project.
    # Recommended: train LM on silver (non-gold), cleaned if condition=clean.
    if condition == "clean":
        fname = "silver_train_clean_v1_FROZEN.csv"
        text_col = "text_clean_v1"
    else:
        fname = "silver_train_raw_norm_FROZEN.csv"
        text_col = "text"

    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training file not found: {path}\n"
            f"Please rename in load_training_texts() to match your actual file."
        )

    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}' in {fname}, got columns: {list(df.columns)[:20]} ...")
    return df[text_col].astype(str).tolist()


def load_gold(data_dir: str) -> pd.DataFrame:
    # Your existing script uses gold_raw_norm_FROZEN.csv; keep that convention.
    path = os.path.join(data_dir, "gold_raw_norm_FROZEN.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold file not found: {path}")
    gold = pd.read_csv(path)
    if "gold_label" not in gold.columns:
        # allow 'label' fallback
        if "label" in gold.columns:
            gold = gold.rename(columns={"label": "gold_label"})
        else:
            raise ValueError("gold_raw_norm_FROZEN.csv must have 'gold_label' (or 'label') column.")
    return gold


def compute_dh_for_gold(gold: pd.DataFrame,
                        lm: TrigramLM,
                        text_col: str,
                        k_shuffles: int = 5,
                        seed: int = 0,
                        min_tokens: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for idx, row in gold.iterrows():
        text = row.get(text_col, "")
        toks = tokenize(str(text))
        n_tokens = len(toks)
        if n_tokens < min_tokens:
            continue

        h_orig = lm.cross_entropy_bits(str(text))

        # within-post token shuffles
        h_shufs = []
        toks_arr = np.array(toks, dtype=object)
        for _ in range(k_shuffles):
            perm = toks_arr.copy()
            rng.shuffle(perm)
            shuf_text = " ".join(perm.tolist())
            h_shufs.append(lm.cross_entropy_bits(shuf_text))

        h_shuf = float(np.mean(h_shufs))
        dH = h_shuf - h_orig  # ΔH > 0 means original has lower entropy than shuffled

        rows.append({
            "post_id": row.get("post_id", idx),
            "gold_label": row["gold_label"],
            "n_tokens": n_tokens,
            "H_orig_bits": float(h_orig),
            "H_shuf_bits": float(h_shuf),
            "dH_bits": float(dH),
        })

    return pd.DataFrame(rows)


def plot_fig1(dh: pd.DataFrame, out_png: str):
    # Basic boxplot + jitter points, x labels include n
    labels = ["ADVICE", "STORY"]
    data = []
    ns = []
    for lab in labels:
        vals = dh.loc[dh["gold_label"] == lab, "dH_bits"].dropna().to_numpy()
        data.append(vals)
        ns.append(len(vals))

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)

    bp = ax.boxplot(data, widths=0.35, patch_artist=True, showfliers=False)

    # do not set explicit colors; matplotlib defaults
    # jitter points
    for i, vals in enumerate(data, start=1):
        x = np.random.normal(loc=i, scale=0.04, size=len(vals))
        ax.scatter(x, vals, s=14, alpha=0.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{labels[0]}\n(n={ns[0]})", f"{labels[1]}\n(n={ns[1]})"], fontsize=12)
    ax.set_ylabel(r"$\Delta H$ (bits/token)", fontsize=13)
    ax.set_title(r"Surprisal reduction $\Delta H = H_{\mathrm{shuf}} - H_{\mathrm{orig}}$", fontsize=13)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run_stats(dh: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> Dict:
    x = dh.loc[dh["gold_label"] == "ADVICE", "dH_bits"].dropna().to_numpy()
    y = dh.loc[dh["gold_label"] == "STORY", "dH_bits"].dropna().to_numpy()

    out = {
        "N_ADVICE": int(len(x)),
        "N_STORY": int(len(y)),
        "median_ADVICE": float(np.median(x)) if len(x) else None,
        "median_STORY": float(np.median(y)) if len(y) else None,
        "median_diff_ADVICE_minus_STORY": float(np.median(x) - np.median(y)) if (len(x) and len(y)) else None,
        "cliffs_delta": float(cliffs_delta(x, y)) if (len(x) and len(y)) else None,
        "perm_test_p_two_sided_median_diff": float(permutation_test_median_diff(x, y, n_perm=n_perm, seed=seed))
            if (len(x) and len(y)) else None,
        "mannwhitneyu_p_two_sided": None,
        "mannwhitneyu_U": None,
        "n_perm": int(n_perm),
        "seed": int(seed),
    }

    if _HAVE_SCIPY and len(x) and len(y):
        res = mannwhitneyu(x, y, alternative="two-sided")
        out["mannwhitneyu_p_two_sided"] = float(res.pvalue)
        out["mannwhitneyu_U"] = float(res.statistic)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_new",
                        help="Directory containing gold_raw_norm_FROZEN.csv and silver train csvs.")
    parser.add_argument("--out_dir", type=str, default="outputs_surprisal",
                        help="Where to write csv/json/fig.")
    parser.add_argument("--condition", type=str, default="clean", choices=["clean", "raw"])
    parser.add_argument("--K", type=int, default=5, help="Number of within-post token shuffles.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_tokens", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1, help="Laplace smoothing.")
    parser.add_argument("--w1", type=float, default=0.1)
    parser.add_argument("--w2", type=float, default=0.3)
    parser.add_argument("--w3", type=float, default=0.6)
    parser.add_argument("--n_perm", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load gold
    gold = load_gold(args.data_dir)

    # Choose text column for gold
    if args.condition == "clean":
        gold_col = "text_clean_v1" if "text_clean_v1" in gold.columns else "text"
    else:
        gold_col = "text" if "text" in gold.columns else "text_clean_v1"

    if gold_col not in gold.columns:
        raise ValueError(f"Gold is missing both 'text' and 'text_clean_v1'. Columns: {list(gold.columns)[:30]} ...")

    # Load training texts (silver)
    train_texts = load_training_texts(args.data_dir, args.condition)
    train_seqs = [tokenize(t) for t in train_texts]
    lm = TrigramLM.train(train_seqs, alpha=args.alpha, w1=args.w1, w2=args.w2, w3=args.w3)

    # Compute ΔH for gold
    dh = compute_dh_for_gold(gold, lm=lm, text_col=gold_col,
                             k_shuffles=args.K, seed=args.seed, min_tokens=args.min_tokens)

    out_csv = os.path.join(args.out_dir, f"gold_surprisal_dh_{args.condition}.csv")
    dh.to_csv(out_csv, index=False)

    # Stats
    stats = run_stats(dh, n_perm=args.n_perm, seed=args.seed)
    stats["config"] = {
        "condition": args.condition,
        "gold_text_col_used": gold_col,
        "K": args.K,
        "min_tokens": args.min_tokens,
        "alpha": args.alpha,
        "w1": args.w1,
        "w2": args.w2,
        "w3": args.w3,
        "train_file_expectation": "silver_train_clean_v1_FROZEN.csv" if args.condition == "clean" else "silver_train_raw.csv",
        "tokenizer": "regex [A-Za-z]+ lowercased",
    }
    out_json = os.path.join(args.out_dir, f"stats_surprisal_tests_{args.condition}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Fig1
    out_png = os.path.join(args.out_dir, f"fig1_surprisal_distribution_{args.condition}.png")
    plot_fig1(dh, out_png)

    print("Done.")
    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()