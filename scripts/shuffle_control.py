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
    raise FileNotFoundError(
        "No training corpus found under data_new/. Put thesis_dataset_master_raw.csv or silver_train_raw_norm_FROZEN.csv there."
    )


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
def compute_dh(
    df: pd.DataFrame,
    lm: InterpTrigramLM,
    k_shuffles: int,
    seed: int,
    text_col: str,
    condition_name: str
) -> pd.DataFrame:
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
    """
    Figure 2 (shuffle control):
    For each label (ADVICE/STORY), show ΔH distributions under two base texts:
      - base = Original (condition = "orig")
      - base = Sentence-permuted (condition = "sent_shuf")
    ΔH is always computed as H(token-shuffled) − H(base).
    """

    # --- display names (do NOT expose internal variable names) ---
    COND_LABEL = {
        "orig": "Original",
        "sent_shuf": "Sentence-permuted",
    }

    labels = ["ADVICE", "STORY"]
    conds = ["orig", "sent_shuf"]

    # Collect arrays
    data = {
        (lab, cond): df_long[
            (df_long["gold_label"] == lab) & (df_long["condition"] == cond)
        ]["dH_bits"].dropna().values
        for lab in labels for cond in conds
    }

    # Keep default font (your “original” look), but make spacing less cramped
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    # positions: for each label i, Original at i-0.22, Sentence-permuted at i+0.22
    positions = []
    boxdata = []
    facecolors = []

    # colors close to common Fig2 palette (blue/orange/green/red)
    COLORS = {
        ("ADVICE", "orig"): "tab:blue",
        ("ADVICE", "sent_shuf"): "tab:orange",
        ("STORY", "orig"): "tab:green",
        ("STORY", "sent_shuf"): "tab:red",
    }

    # tick labels under each box
    xtick_labels = []

    for i, lab in enumerate(labels, start=1):
        for cond, dx in zip(conds, [-0.22, +0.22]):
            pos = i + dx
            arr = data[(lab, cond)]
            n = len(arr)

            positions.append(pos)
            boxdata.append(arr)
            facecolors.append(COLORS[(lab, cond)])
            xtick_labels.append(f"{COND_LABEL[cond]}\n(n={n})")

    bp = ax.boxplot(
        boxdata,
        positions=positions,
        widths=0.34,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(linewidth=1.2, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
    )
    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.28)

    # jitter points (colored to match each box)
    rng = np.random.default_rng(seed)
    for (lab, cond), ys in data.items():
        if len(ys) == 0:
            continue
        i = 1 if lab == "ADVICE" else 2
        x0 = i - 0.22 if cond == "orig" else i + 0.22
        xs = rng.normal(loc=x0, scale=0.05, size=len(ys))
        ax.scatter(xs, ys, s=18, alpha=0.40, linewidths=0, color=COLORS[(lab, cond)])

    # x-axis formatting: tick labels for each box + group labels underneath
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels, fontsize=10, linespacing=1.15)

    # group labels (ADVICE / STORY) below the ticks, to avoid “4 boxes floating”
    ax.text(1.0, -0.24, "ADVICE", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=12)
    ax.text(2.0, -0.24, "STORY", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=12)

    # subtle separator between groups
    ax.axvline(1.5, color="black", linewidth=0.8, alpha=0.12)

    # y-axis label (your preferred wording)
    ax.set_ylabel(r"$\Delta H$ (bits per token)", fontsize=12)

    # light grid for readability (not visually busy)
    ax.yaxis.grid(True, alpha=0.16)
    ax.set_axisbelow(True)

    # put the definition outside plot area (prevents crowding)
    fig.text(
        0.12, 0.02,
        r"$\Delta H = H(\mathrm{token\!-\!shuffled}) - H(\mathrm{base})$;  "
        r"base is Original or Sentence-permuted.",
        ha="left", va="bottom", fontsize=10
    )

    # tighten with extra bottom room for the group labels + formula line
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
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