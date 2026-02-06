from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


LABELS = ["ADVICE", "STORY"]
TOKEN_RE = re.compile(r"[A-Za-z']+")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_label(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    return s if s in {"ADVICE", "STORY"} else None


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return TOKEN_RE.findall(str(text).lower())


def split_sentences(text: str) -> List[str]:
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t.replace("\r", " "))
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    return sents


def sentence_shuffle(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) < 2:
        return str(text)
    sents2 = sents.copy()
    rng.shuffle(sents2)
    return " ".join(sents2)


class InterpTrigramLM:
    def __init__(self, alpha: float = 0.5, w1: float = 0.1, w2: float = 0.3, w3: float = 0.6):
        self.alpha = float(alpha)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.bi_ctx: Dict[str, int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.total_uni: int = 0
        self.V: int = 1

    def fit(self, texts: List[str]):
        self.uni.clear()
        self.bi.clear()
        self.bi_ctx.clear()
        self.tri.clear()
        self.total_uni = 0

        for t in texts:
            toks = tokenize(t)
            if not toks:
                continue
            seq = ["<BOS>", "<BOS>"] + toks + ["<EOS>"]
            for i in range(2, len(seq)):
                w, u, v = seq[i], seq[i - 1], seq[i - 2]
                self.uni[w] = self.uni.get(w, 0) + 1
                self.total_uni += 1
                self.bi[(u, w)] = self.bi.get((u, w), 0) + 1
                self.bi_ctx[u] = self.bi_ctx.get(u, 0) + 1
                self.tri[(v, u, w)] = self.tri.get((v, u, w), 0) + 1

        self.V = max(1, len(self.uni) + 1)

    def p1(self, w: str) -> float:
        c = self.uni.get(w, 0)
        return (c + self.alpha) / (self.total_uni + self.alpha * self.V)

    def p2(self, u: str, w: str) -> float:
        """Add-alpha bigram prob p(w|u) with O(1) denominator lookup."""
        num = self.bi.get((u, w), 0) + self.alpha
        den = self.bi_ctx.get(u, 0) + self.alpha * self.V
        return num / den if den > 0 else self.p1(w)

    def p3(self, v: str, u: str, w: str) -> float:
        """Add-alpha trigram prob p(w|v,u) with O(1) denominator via bigram count (v,u)."""
        num = self.tri.get((v, u, w), 0) + self.alpha
        den = self.bi.get((v, u), 0) + self.alpha * self.V
        return num / den if den > 0 else self.p2(u, w)

    def cross_entropy_bits(self, text: str) -> float:
        toks = tokenize(text)
        if len(toks) < 1:
            return float("nan")
        seq = ["<BOS>", "<BOS>"] + toks + ["<EOS>"]
        neglog2 = 0.0
        count = 0
        for i in range(2, len(seq)):
            w, u, v = seq[i], seq[i - 1], seq[i - 2]
            p = self.w3 * self.p3(v, u, w) + self.w2 * self.p2(u, w) + self.w1 * self.p1(w)
            p = max(1e-12, float(p))
            neglog2 += -math.log(p, 2)
            count += 1
        return neglog2 / max(1, count)


def load_gold_binary(data_dir: Path, condition: str) -> Tuple[pd.DataFrame, str]:
    if condition == "raw":
        p = data_dir / "gold_raw_norm_FROZEN.csv"
        df = pd.read_csv(p, low_memory=False)
        text_col = "text"
    else:
        p = data_dir / "gold_clean_v1_FROZEN.csv"
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            text_col = "text_clean_v1"
        else:
            alt = data_dir / "gold_clean_v1_FROZEN.csv"
            if alt.exists():
                df = pd.read_csv(alt, low_memory=False)
                text_col = "text_clean_v1" if "text_clean_v1" in df.columns else "text"
            else:
                df = pd.read_csv(data_dir / "gold_raw_norm_FROZEN.csv", low_memory=False)
                text_col = "text"

    df["gold_label"] = df["gold_label"].map(normalize_label)
    df = df[df["gold_label"].isin(LABELS)].copy()

    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str).str.lower()
    if "doc_id" in df.columns:
        df["doc_id"] = df["doc_id"].astype(str)
    else:
        df["doc_id"] = df.index.astype(str)

    return df, text_col


def find_training_corpus(data_dir: Path, condition: str) -> Tuple[pd.DataFrame, str]:
    if condition == "clean":
        candidates = [
            (data_dir / "thesis_dataset_master_cleaned.csv", "text_clean_v1"),
            (data_dir / "silver_train_clean_v1_FROZEN.csv", "text_clean_v1"),
            (data_dir / "silver_train_raw_norm_FROZEN.csv", "text"),
            (data_dir / "thesis_dataset_master_raw.csv", "text"),
        ]
    else:
        candidates = [
            (data_dir / "thesis_dataset_master_raw.csv", "text"),
            (data_dir / "silver_train_raw_norm_FROZEN.csv", "text"),
        ]

    for p, col in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            if col in df.columns:
                return df, col

    df = pd.read_csv(data_dir / "gold_raw_norm_FROZEN.csv", low_memory=False)
    return df, "text"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = len(x) * len(y)
    return (gt - lt) / denom


def stats_block(df: pd.DataFrame) -> Dict[str, float]:
    x = df[df["gold_label"] == "ADVICE"]["dH_bits"].dropna().values
    y = df[df["gold_label"] == "STORY"]["dH_bits"].dropna().values

    out: Dict[str, float] = {
        "n_advice": int(len(x)),
        "n_story": int(len(y)),
        "median_advice": float(np.median(x)) if len(x) else float("nan"),
        "median_story": float(np.median(y)) if len(y) else float("nan"),
        "mean_advice": float(np.mean(x)) if len(x) else float("nan"),
        "mean_story": float(np.mean(y)) if len(y) else float("nan"),
        "cliffs_delta": float(cliffs_delta(x, y)),
    }

    if len(x) and len(y) and SCIPY_OK:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        out["mw_stat"] = float(stat)
        out["mw_p_value"] = float(p)

    return out


def compute_dh(
    df: pd.DataFrame,
    text_col: str,
    lm: InterpTrigramLM,
    k_shuffles: int,
    seed: int,
    condition_name: str,
) -> pd.DataFrame:
    rows = []
    rng_tokens = np.random.default_rng(seed)

    for i, r in df.reset_index(drop=True).iterrows():
        base_text = str(r[text_col])

        if condition_name == "sent_shuf":
            rng_sent = random.Random(seed + i)
            text = sentence_shuffle(base_text, rng_sent)
        else:
            text = base_text

        toks = tokenize(text)
        if len(toks) < 5:
            continue

        h_orig = lm.cross_entropy_bits(text)

        hs = []
        for _ in range(k_shuffles):
            toks2 = toks.copy()
            rng_tokens.shuffle(toks2)
            hs.append(lm.cross_entropy_bits(" ".join(toks2)))
        h_shuf = float(np.mean(hs))

        rows.append({
            "doc_id": str(r.get("doc_id", i)),
            "condition": condition_name,
            "domain": r.get("domain", None),
            "gold_label": r["gold_label"],
            "n_tokens": int(len(toks)),
            "H_orig_bits": float(h_orig),
            "H_shuf_bits": float(h_shuf),
            "dH_bits": float(h_shuf - h_orig),
        })

    return pd.DataFrame(rows)


def plot_fig2(
    df_long: pd.DataFrame,
    out_png: Path,
    seed: int,
    show_pairs: bool = True,
):
    """
    Figure 2 (shuffle control).
    For each gold label (ADVICE/STORY), compare ΔH computed on:
      - base = Original (Full sequence)
      - base = Sentence-permuted (sentence order shuffled)

    ΔH is always: H(token-shuffled) − H(base).
    """

    # Typography (fallback-safe)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 11.5
    plt.rcParams["legend.fontsize"] = 11

    labels = ["ADVICE", "STORY"]
    conds = ["orig", "sent_shuf"]

    cond_display = {
        "orig": "Original",
        "sent_shuf": "Sentence-permuted",
    }

    # Two-color scheme (close to typical Fig2 styling)
    colors = {
        "orig": "#4C72B0",      # blue
        "sent_shuf": "#DD8452", # orange
    }

    offsets = {"orig": -0.22, "sent_shuf": 0.22}

    # Prepare grouped data
    data = {
        (lab, cond): df_long[
            (df_long["gold_label"] == lab) & (df_long["condition"] == cond)
        ]["dH_bits"].dropna().values
        for lab in labels for cond in conds
    }

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    # Boxplot inputs
    positions: List[float] = []
    boxdata: List[np.ndarray] = []
    box_facecolors: List[str] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []

    for i, lab in enumerate(labels, start=1):
        for cond in conds:
            pos = i + offsets[cond]
            arr = data[(lab, cond)]
            positions.append(pos)
            boxdata.append(arr)
            box_facecolors.append(colors[cond])

            n = len(arr)
            tick_positions.append(pos)
            tick_labels.append(f"{cond_display[cond]}\n(n={n})")

    bp = ax.boxplot(
        boxdata,
        positions=positions,
        widths=0.34,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=1.2, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
    )

    for patch, fc in zip(bp["boxes"], box_facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.30)

    # Jittered points (same color as the condition)
    rng = np.random.default_rng(seed)
    for i, lab in enumerate(labels, start=1):
        for cond in conds:
            ys = data[(lab, cond)]
            if len(ys) == 0:
                continue
            x0 = i + offsets[cond]
            xs = rng.normal(loc=x0, scale=0.045, size=len(ys))
            ax.scatter(xs, ys, s=20, alpha=0.45, linewidths=0, color=colors[cond])

    # Optional: paired lines (same post under two bases)
    if show_pairs:
        # Need a stable id to pair on; fall back gracefully if missing.
        id_col = None
        for cand in ["id", "post_id", "gold_id", "uid", "example_id"]:
            if cand in df_long.columns:
                id_col = cand
                break

        if id_col is not None:
            # For each label, connect orig -> sent_shuf for the same post
            for lab in labels:
                sub = df_long[df_long["gold_label"] == lab].copy()
                pivot = sub.pivot_table(index=id_col, columns="condition", values="dH_bits", aggfunc="first")
                pivot = pivot.dropna(subset=["orig", "sent_shuf"], how="any")
                x_left = (1 if lab == "ADVICE" else 2) + offsets["orig"]
                x_right = (1 if lab == "ADVICE" else 2) + offsets["sent_shuf"]
                for _, row in pivot.iterrows():
                    ax.plot([x_left, x_right], [row["orig"], row["sent_shuf"]],
                            color="black", alpha=0.08, linewidth=0.8, zorder=0)

    # Axes
    ax.set_xlim(0.4, 2.6)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, linespacing=1.15)
    ax.set_ylabel(r"$\Delta H$ (bits per token)")

    # Group labels below the two pairs
    ax.text(1.0, -0.24, "ADVICE", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=13)
    ax.text(2.0, -0.24, "STORY", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=13)

    # Subtle separator
    ax.axvline(1.5, color="black", linewidth=0.8, alpha=0.15)

    # Light grid for readability
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)

    # Legend for conditions (no internal variable names)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=10,
               markerfacecolor=colors["orig"], markeredgecolor="black", alpha=0.35,
               label=cond_display["orig"]),
        Line2D([0], [0], marker="s", linestyle="none", markersize=10,
               markerfacecolor=colors["sent_shuf"], markeredgecolor="black", alpha=0.35,
               label=cond_display["sent_shuf"]),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    # Footnote-style note outside axes (keeps plot area clean)
    fig.text(
        0.12, 0.02,
        r"$\Delta H = H(\mathrm{token\!-\!shuffled}) - H(\mathrm{base})$; "
        r"Sentence-permuted shuffles sentence order only.",
        ha="left", va="bottom", fontsize=10
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["raw", "clean"], default="clean")
    ap.add_argument("--K", type=int, default=5, help="token-shuffle repeats per post")
    ap.add_argument("--alpha", type=float, default=0.5, help="add-alpha smoothing")
    ap.add_argument("--seed", type=int, default=1004, help="random seed")
    ap.add_argument("--n_perm", type=int, default=5000, help="permutation test repeats")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--no_pairs", action="store_true", help="disable paired lines in Figure 2")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else (repo_root() / "data_new")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_col = find_training_corpus(data_dir, args.condition)
    train_texts = train_df[train_col].astype(str).tolist()

    lm = InterpTrigramLM(alpha=args.alpha, w1=0.1, w2=0.3, w3=0.6)
    lm.fit(train_texts)

    gold, gold_col = load_gold_binary(data_dir, args.condition)

    dh_orig = compute_dh(gold, text_col=gold_col, lm=lm, k_shuffles=args.K, seed=args.seed, condition_name="orig")
    dh_sent = compute_dh(gold, text_col=gold_col, lm=lm, k_shuffles=args.K, seed=args.seed, condition_name="sent_shuf")

    dh_long = pd.concat([dh_orig, dh_sent], ignore_index=True)

    out_csv = data_dir / f"gold_surprisal_dh_sentence_control_{args.condition}.csv"
    dh_long.to_csv(out_csv, index=False, encoding="utf-8")

    out_png = data_dir / f"fig2_sentence_shuffle_control_{args.condition}.png"
    plot_fig2(dh_long, out_png, seed=args.seed, show_pairs=(not args.no_pairs))

    # stats: compare ADVICE vs STORY within each condition
    stats = {
        "definition": "Sentence-order control: base text optionally sentence-permuted, then ΔH computed vs token-shuffled baseline.",
        "condition": args.condition,
        "lm_config": {"order": 3, "alpha": args.alpha, "interp": [0.1, 0.3, 0.6], "seed": args.seed},
        "K": int(args.K),
        "by_condition": {},
    }
    for cond in ["orig", "sent_shuf"]:
        d = dh_long[dh_long["condition"] == cond]
        stats["by_condition"][cond] = stats_block(d)

    out_json = data_dir / f"stats_sentence_shuffle_control_{args.condition}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved:")
    print(" -", out_csv)
    print(" -", out_png)
    print(" -", out_json)


if __name__ == "__main__":
    main()
