from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


# -------------------------------
# Config
# -------------------------------
DOMAINS = ["aita", "confession", "ra"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_label(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s in {"ADVICE", "STORY"} else None


_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return _WORD_RE.findall(str(text).lower())


# Sentence splitter:
# - split on ., !, ? followed by whitespace OR on newlines
# - keep punctuation attached to sentence (we don't drop it; but tokenizer later drops punctuation anyway)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence segmentation (consistent + no external deps)."""
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    # Normalize spaces
    t = re.sub(r"\s+", " ", t.replace("\r", " "))
    # Split
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    return sents


def sentence_shuffle(text: str, rng: random.Random) -> str:
    """
    Shuffle sentence order; keep within-sentence token order unchanged.
    If <2 sentences, return original.
    """
    sents = split_sentences(text)
    if len(sents) < 2:
        return str(text)
    sents2 = sents.copy()
    rng.shuffle(sents2)
    # Join with a space to avoid accidental token sticking
    return " ".join(sents2)


# -------------------------------
# Trigram LM (same spirit as Day9)
# -------------------------------
@dataclass
class InterpTrigramLM:
    alpha: float = 0.5
    interp: Tuple[float, float, float] = (0.1, 0.3, 0.6)  # (uni, bi, tri)

    uni: Dict[str, int] = None
    bi: Dict[Tuple[str, str], int] = None
    tri: Dict[Tuple[str, str, str], int] = None
    total_uni: int = 0
    V: int = 1

    def __post_init__(self):
        self.uni = {}
        self.bi = {}
        self.tri = {}
        self.total_uni = 0
        self.V = 1

    def fit(self, texts: List[str]):
        for t in texts:
            toks = tokenize(t)
            if len(toks) == 0:
                continue
            seq = ["<BOS>", "<BOS>"] + toks + ["<EOS>"]
            for i in range(2, len(seq)):
                w, u, v = seq[i], seq[i - 1], seq[i - 2]
                self.uni[w] = self.uni.get(w, 0) + 1
                self.total_uni += 1
                self.bi[(u, w)] = self.bi.get((u, w), 0) + 1
                self.tri[(v, u, w)] = self.tri.get((v, u, w), 0) + 1
        self.V = max(1, len(self.uni) + 1)

    def p_uni(self, w: str) -> float:
        c = self.uni.get(w, 0)
        return (c + self.alpha) / (self.total_uni + self.alpha * self.V)

    def p_bi(self, u: str, w: str) -> float:
        c_uw = self.bi.get((u, w), 0)
        c_u = self.uni.get(u, 0)
        return (c_uw + self.alpha) / (c_u + self.alpha * self.V)

    def p_tri(self, v: str, u: str, w: str) -> float:
        c_vuw = self.tri.get((v, u, w), 0)
        c_vu = self.bi.get((v, u), 0)
        return (c_vuw + self.alpha) / (c_vu + self.alpha * self.V)

    def prob(self, v: str, u: str, w: str) -> float:
        l1, l2, l3 = self.interp
        return l1 * self.p_uni(w) + l2 * self.p_bi(u, w) + l3 * self.p_tri(v, u, w)

    def cross_entropy_bits(self, text: str) -> float:
        toks = tokenize(text)
        if len(toks) < 5:
            return float("nan")
        seq = ["<BOS>", "<BOS>"] + toks + ["<EOS>"]
        nll = 0.0
        cnt = 0
        for i in range(2, len(seq)):
            w, u, v = seq[i], seq[i - 1], seq[i - 2]
            p = max(self.prob(v, u, w), 1e-12)
            nll += -math.log(p, 2)
            cnt += 1
        return nll / max(1, cnt)


def find_training_corpus(data_dir: Path) -> Tuple[pd.DataFrame, str]:
    """
    Prefer a larger corpus for a stable LM (same idea as Day9).
    Will try (in order):
      - thesis_dataset_master_raw.csv (text)
      - silver_train_raw_norm_FROZEN.csv (text)
      - silver_train_clean_v1_FROZEN.csv (text_clean_v1)
      - gold_raw_norm_FROZEN.csv (text)  [fallback]
    """
    candidates = [
        (data_dir / "thesis_dataset_master_raw.csv", "text"),
        (data_dir / "silver_train_raw_norm_FROZEN.csv", "text"),
        (data_dir / "silver_train_clean_v1_FROZEN.csv", "text_clean_v1"),
        (data_dir / "gold_raw_norm_FROZEN.csv", "text"),
    ]
    for p, col in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            if col in df.columns:
                return df, col
    raise FileNotFoundError("No training corpus found under data_new/. Put thesis_dataset_master_raw.csv or silver_train_raw_norm_FROZEN.csv there.")


def load_gold_binary(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "gold_raw_norm_FROZEN.csv"
    df = pd.read_csv(p, low_memory=False)
    df["gold_label"] = df["gold_label"].map(normalize_label)
    df = df[df["gold_label"].isin(["ADVICE", "STORY"])].copy()
    df["domain"] = df["domain"].astype(str).str.lower()
    df = df[df["domain"].isin(DOMAINS)].copy()
    df["text"] = df["text"].astype(str)
    return df


# -------------------------------
# Effect size + tests (same as Day9)
# -------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                gt += 1
            elif xi < yj:
                lt += 1
    denom = len(x) * len(y)
    return float("nan") if denom == 0 else (gt - lt) / denom


def permutation_test_median_diff(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    obs = float(np.median(x) - np.median(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    stats = []
    for _ in range(n_perm):
        rng.shuffle(pooled)
        x2 = pooled[:n_x]
        y2 = pooled[n_x:]
        stats.append(float(np.median(x2) - np.median(y2)))
    stats = np.asarray(stats)
    p = float((np.abs(stats) >= abs(obs)).mean())
    return obs, p


def stats_block(df: pd.DataFrame, n_perm: int, seed: int) -> Dict:
    sub = df.dropna(subset=["dH_bits"]).copy()
    a = sub[sub["gold_label"] == "ADVICE"]["dH_bits"].values
    s = sub[sub["gold_label"] == "STORY"]["dH_bits"].values

    res = {
        "n_advice": int(len(a)),
        "n_story": int(len(s)),
        "mean_advice": float(np.mean(a)) if len(a) else float("nan"),
        "mean_story": float(np.mean(s)) if len(s) else float("nan"),
        "median_advice": float(np.median(a)) if len(a) else float("nan"),
        "median_story": float(np.median(s)) if len(s) else float("nan"),
    }

    if len(a) and len(s):
        res["cliffs_delta"] = float(cliffs_delta(a, s))
        if mannwhitneyu is not None:
            U, p = mannwhitneyu(a, s, alternative="two-sided")
            res["mannwhitney_U"] = float(U)
            res["mannwhitney_p"] = float(p)
        else:
            res["mannwhitney_U"] = None
            res["mannwhitney_p"] = None
        obs, pp = permutation_test_median_diff(a, s, n_perm=n_perm, seed=seed)
        res["perm_stat_median_diff"] = float(obs)
        res["perm_p"] = float(pp)

    return res


# -------------------------------
# ΔH computation for 2 conditions
# -------------------------------
def compute_dh(df: pd.DataFrame, lm: InterpTrigramLM, k_shuffles: int, seed: int, text_col: str, condition_name: str) -> pd.DataFrame:
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        base_text = r[text_col]

        # condition-specific transform
        if condition_name == "sent_shuf":
            rng = random.Random(seed + i)
            text = sentence_shuffle(base_text, rng)
        else:
            text = base_text

        toks = tokenize(text)
        if len(toks) < 5:
            continue

        h_orig = lm.cross_entropy_bits(text)

        # token-level shuffle averaged over K
        rng2 = random.Random(seed * 10 + i)
        hs = []
        for _ in range(k_shuffles):
            toks2 = toks.copy()
            rng2.shuffle(toks2)
            hs.append(lm.cross_entropy_bits(" ".join(toks2)))
        h_shuf = float(np.mean(hs))

        rows.append({
            "condition": condition_name,
            "domain": r["domain"],
            "gold_label": r["gold_label"],
            "n_tokens": len(toks),
            "H_orig_bits": float(h_orig),
            "H_shuf_bits": float(h_shuf),
            "dH_bits": float(h_shuf - h_orig),
        })
    return pd.DataFrame(rows)


def plot_fig2(df_long: pd.DataFrame, out_png: Path, seed: int):
    """Figure 2: within-label original vs sentence-permuted control.

    df_long columns expected:
      - gold_label in {ADVICE, STORY}
      - condition in {orig, sent_shuf}
      - dH_bits (float)
      - post_id (optional but recommended for paired lines)
    """
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)

    # Color palette (matches your original figure style)
    c_advice_orig = "#4C72B0"  # blue
    c_advice_sent = "#DD8452"  # orange
    c_story_orig  = "#55A868"  # green
    c_story_sent  = "#C44E52"  # red
    colors = [c_advice_orig, c_advice_sent, c_story_orig, c_story_sent]

    # Order: ADVICE/orig, ADVICE/sent_shuf, STORY/orig, STORY/sent_shuf
    order = [("ADVICE", "orig"), ("ADVICE", "sent_shuf"), ("STORY", "orig"), ("STORY", "sent_shuf")]
    boxdata = [
        df_long[(df_long["gold_label"] == lab) & (df_long["condition"] == cond)]["dH_bits"].dropna().to_numpy()
        for lab, cond in order
    ]

    # X positions: two groups centered at 1 and 2
    positions = [1 - 0.18, 1 + 0.18, 2 - 0.18, 2 + 0.18]

    # Median line: rust red, thicker
    medianprops = {"color": "#B7410E", "linewidth": 2.2}

    bp = ax.boxplot(
        boxdata,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops=medianprops,
        boxprops={"linewidth": 1.8, "edgecolor": "black"},
        whiskerprops={"linewidth": 1.8, "color": "black"},
        capprops={"linewidth": 1.8, "color": "black"},
    )

    # Keep boxes clean (white fill) like your original; color comes from points
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    # Scatter (jitter) points in the original 4 colors
    for (lab, cond), pos, col in zip(order, positions, colors):
        ys = df_long[(df_long["gold_label"] == lab) & (df_long["condition"] == cond)]["dH_bits"].dropna().to_numpy()
        xs = rng.normal(pos, 0.04, size=len(ys))
        ax.scatter(xs, ys, s=18, alpha=0.55, color=col, edgecolors="none")

    # Optional paired lines: same post_id across conditions within each label
    if "post_id" in df_long.columns:
        for lab, x0, x1 in [("ADVICE", positions[0], positions[1]), ("STORY", positions[2], positions[3])]:
            sub = df_long[df_long["gold_label"] == lab].copy()
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="post_id", columns="condition", values="dH_bits", aggfunc="mean")
            if "orig" in pivot.columns and "sent_shuf" in pivot.columns:
                for _, r in pivot.dropna().iterrows():
                    ax.plot([x0, x1], [r["orig"], r["sent_shuf"]], color="0.75", linewidth=0.6, alpha=0.6, zorder=0)

    # Two-level x-axis: major ticks are labels; minor ticks are conditions
    # n is computed from unique post_id if available, else from orig rows
    if "post_id" in df_long.columns:
        n_advice = int(df_long[df_long["gold_label"] == "ADVICE"]["post_id"].nunique())
        n_story = int(df_long[df_long["gold_label"] == "STORY"]["post_id"].nunique())
    else:
        n_advice = int((df_long["gold_label"] == "ADVICE").sum() // 2)
        n_story = int((df_long["gold_label"] == "STORY").sum() // 2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"ADVICE\n(n={n_advice})", f"STORY\n(n={n_story})"], fontsize=13)

    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels(["Full", "Sent-shuf", "Full", "Sent-shuf"], minor=True, fontsize=10)
    ax.tick_params(axis="x", which="major", pad=18)
    ax.tick_params(axis="x", which="minor", pad=2, length=0)

    ax.set_ylabel(r"$\Delta H$ (bits/token)", fontsize=13)
    ax.set_title("Sentence-order permutation control", fontsize=13)

    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5, -0.18,
        "Within each label: left = Full (original), right = Sent-shuf (sentence-permuted).",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=10
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5, help="token-shuffle repeats per post")
    ap.add_argument("--alpha", type=float, default=0.5, help="add-alpha smoothing")
    ap.add_argument("--seed", type=int, default=1004, help="random seed")
    ap.add_argument("--n_perm", type=int, default=5000, help="permutation test repeats")
    args = ap.parse_args()

    root = repo_root()
    data_dir = root / "data_new"

    # 1) train LM
    train_df, train_col = find_training_corpus(data_dir)
    train_texts = train_df[train_col].astype(str).tolist()

    lm = InterpTrigramLM(alpha=args.alpha, interp=(0.1, 0.3, 0.6))
    lm.fit(train_texts)

    # 2) load gold
    gold = load_gold_binary(data_dir)

    # 3) compute ΔH for both conditions
    dh_orig = compute_dh(gold, lm, k_shuffles=args.K, seed=args.seed, text_col="text", condition_name="orig")
    dh_shuf = compute_dh(gold, lm, k_shuffles=args.K, seed=args.seed, text_col="text", condition_name="sent_shuf")

    dh_long = pd.concat([dh_orig, dh_shuf], ignore_index=True)

    out_csv = data_dir / "surprisal_scores_shuffled.csv"
    dh_long.to_csv(out_csv, index=False)

    # 4) figure2
    out_png = data_dir / "fig2_shuffle_control.png"
    plot_fig2(dh_long, out_png, seed=args.seed)

    # 5) stats (optional, but useful to show "separation weakens")
    stats = {
        "definition": "Shuffle control: compute Day9 ΔH on original vs sentence-order shuffled texts. ΔH = H_token_shuffle - H_original.",
        "lm_config": {"order": 3, "alpha": args.alpha, "interp": [0.1, 0.3, 0.6], "seed": args.seed, "K": args.K},
        "overall": {},
        "by_domain": {},
    }

    for cond in ["orig", "sent_shuf"]:
        sub = dh_long[dh_long["condition"] == cond].copy()
        stats["overall"][cond] = stats_block(sub, n_perm=args.n_perm, seed=args.seed)

        stats["by_domain"][cond] = {}
        for dom in DOMAINS:
            sub2 = sub[sub["domain"] == dom].copy()
            stats["by_domain"][cond][dom] = stats_block(sub2, n_perm=args.n_perm, seed=args.seed)

    out_json = data_dir / "stats_shuffle_control.json"
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print("Wrote:", out_json)
    print("Done.")


if __name__ == "__main__":
    main()