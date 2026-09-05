"""
Regenerate the README figures from committed artifacts.

Run with:  python -m scripts.make_figures

Nothing here hard-codes a result. The Sharpe attribution chart is read from
outputs/walkforward_significance.csv, and the leakage control chart is
recomputed with the same helpers tests/test_no_leakage.py uses, so a figure
cannot silently drift away from the numbers it is meant to illustrate.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data_cache") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

OUTPUT_DIR = Path(config.OUTPUT_DIR)

# Muted, print-safe palette. One accent colour for the effect that is
# statistically distinguishable from zero, grey for the ones that are not.
ACCENT = "#2f6f4e"
NEUTRAL = "#9aa0a6"
LEAK = "#b4553f"
CLEAN = "#2f6f4e"
TEXT = "#222222"
SIGNIFICANCE_LEVEL = 0.05


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=TEXT, labelsize=10)


def sharpe_decomposition(output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Bar chart attributing the Sharpe difference to each layer of the strategy.

    Reads the bootstrap results directly, so the bars and the p-values shown
    beside them are the same numbers the README table quotes.
    """
    sig = pd.read_csv(output_dir / "walkforward_significance.csv", index_col="comparison")

    # (row in the CSV) -> (label for a reader who has not read the code)
    rows = [
        ("Equal-Weight (Universe) vs NIFTY 50", "Equal-weighting vs NIFTY 50"),
        ("Equal-Weight (Selected) vs Equal-Weight (Universe)", "ML stock selection"),
        ("Black-Litterman vs Equal-Weight (Selected)", "BL portfolio weighting"),
    ]
    labels = [label for _, label in rows]
    effects = [float(sig.loc[key, "observed_diff"]) for key, _ in rows]
    pvalues = [float(sig.loc[key, "p_value"]) for key, _ in rows]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    y = np.arange(len(labels))[::-1]  # first row at the top
    colors = [ACCENT if p < SIGNIFICANCE_LEVEL else NEUTRAL for p in pvalues]
    ax.barh(y, effects, height=0.55, color=colors, zorder=3)

    span = max(abs(min(effects)), abs(max(effects)))
    p_column = span * 1.45  # fixed column so the p-values line up as a table

    for yi, effect, p in zip(y, effects, pvalues):
        side = 1 if effect >= 0 else -1
        ax.text(effect + side * span * 0.045, yi, f"{effect:+.3f}",
                va="center", ha="left" if side > 0 else "right",
                fontsize=11.5, fontweight="bold", color=TEXT, zorder=4)
        ax.text(p_column, yi, f"p = {p:.3f}", va="center", ha="left", fontsize=10,
                color=TEXT if p < SIGNIFICANCE_LEVEL else "#5f6368",
                fontweight="bold" if p < SIGNIFICANCE_LEVEL else "normal", zorder=4)

    ax.axvline(0, color="#444444", linewidth=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=TEXT)
    ax.set_xlabel("Change in Sharpe ratio vs. the preceding benchmark", fontsize=10.5, color=TEXT)
    ax.set_title("Where the Sharpe improvement actually comes from",
                 fontsize=13, fontweight="bold", color=TEXT, pad=14, loc="left")
    ax.set_xlim(-span * 1.6, span * 2.05)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    _style(ax)

    fig.text(0.012, 0.02,
             "Walk-forward, 18 quarterly rebalances. Green = distinguishable from zero at the 5% level.",
             fontsize=8.5, color="#5f6368")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path = output_dir / "walkforward_sharpe_decomposition.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def leakage_random_walk_control(output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Bar chart of the random-walk leakage control.

    Recomputed here with the same helpers the regression test uses, averaged
    over the same seeds, so the figure and tests/test_no_leakage.py cannot
    disagree.
    """
    from config import FEATURE_COLUMNS, PURGE_DAYS, XGB_PARAMS
    from src.features import TARGET_COLUMN, calculate_features
    from tests.test_no_leakage import (
        OLD_FEATURE_COLUMNS, OLD_TARGET_COLUMN, OLD_XGB_PARAMS, SEEDS,
        _score, old_calculate_features, random_walk_frame,
    )

    original, regularized, corrected = [], [], []
    for seed in SEEDS:
        walk = random_walk_frame(seed=seed)
        old_frame, new_frame = old_calculate_features(walk), calculate_features(walk)
        original.append(_score(old_frame, OLD_FEATURE_COLUMNS, OLD_TARGET_COLUMN, OLD_XGB_PARAMS, purge=0))
        regularized.append(_score(old_frame, OLD_FEATURE_COLUMNS, OLD_TARGET_COLUMN, XGB_PARAMS, purge=0))
        corrected.append(_score(new_frame, FEATURE_COLUMNS, TARGET_COLUMN, XGB_PARAMS, purge=PURGE_DAYS))

    labels = [
        "Original pipeline\n(original model settings)",
        "Original pipeline\n(regularized model settings)",
        "Corrected pipeline\n(forward target + purged split)",
    ]
    series = [original, regularized, corrected]
    means = [float(np.mean(s)) for s in series]
    colors = [LEAK, LEAK, CLEAN]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.55, color=colors, zorder=3)

    # per-seed spread, offset to the right so it never sits under the label
    for xi, scores in zip(x, series):
        ax.scatter([xi + 0.20] * len(scores), scores, s=20, color="#33333366",
                   edgecolors="none", zorder=5)
    # label clears both the bar and the highest seed dot
    top = max(max(s) for s in series)
    for xi, m, scores in zip(x, means, series):
        ax.text(xi, max(m, max(scores)) + top * 0.045, f"{m:.3f}", ha="center",
                fontsize=12.5, fontweight="bold", color=TEXT, zorder=6)

    ax.axhline(0, color="#444444", linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=TEXT)
    ax.set_ylabel("|correlation| between prediction\nand realised future return",
                  fontsize=10.5, color=TEXT)
    ax.set_title("A pipeline should not be able to predict random data",
                 fontsize=13, fontweight="bold", color=TEXT, pad=14, loc="left")
    ax.set_ylim(0, top * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    _style(ax)

    fig.text(0.012, 0.055,
             f"Seeded random walks, {len(SEEDS)} seeds; grey dots show individual seeds.",
             fontsize=8.5, color="#5f6368")
    fig.text(0.012, 0.018,
             "True predictability is zero by construction, so a bar well above zero indicates information leakage.",
             fontsize=8.5, color="#5f6368")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    path = output_dir / "leakage_random_walk_control.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"  leakage control means: original={means[0]:.3f} "
          f"regularized={means[1]:.3f} corrected={means[2]:.3f}")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (sharpe_decomposition(), leakage_random_walk_control()):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
