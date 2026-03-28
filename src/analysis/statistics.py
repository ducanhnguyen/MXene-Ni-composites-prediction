# src/analysis/statistics.py
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import safe_filename
def export_csv(df: pd.DataFrame, out_dir: str, name: str):
    """
    Export CSV with sanitized filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = safe_filename(name) + ".csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    print(f"[OK] Saved CSV → {path}")


def export_figure(fig, out_dir: str, name: str):
    """
    Export figure with sanitized filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = safe_filename(name) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved FIG → {path}")


# ============================================================
# MAIN API
# ============================================================

def export_statistics_and_visualize(
    df,
    process_features,
    state_vars,
    out_dir
):
    """
    Descriptive statistics + visualization for ONE dataset.
    ALL exports are sanitized.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ======================================================
    # 1. NUMERIC STATISTICS (CSV)
    # ======================================================
    def compute_stats(cols):
        rows = []
        for c in cols:
            s = df[c]
            rows.append({
                "variable": c,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
            })
        return pd.DataFrame(rows)

    input_stats = compute_stats(process_features)
    output_stats = compute_stats(state_vars)

    export_csv(input_stats, out_dir, "input_statistics")
    export_csv(output_stats, out_dir, "output_statistics")

    # ======================================================
    # 2. BOXPLOT – INPUT
    # ======================================================
    fig = plt.figure(figsize=(max(8, 0.6 * len(process_features)), 4))
    sns.boxplot(data=df[process_features], orient="v")
    plt.xticks(rotation=45, ha="right")
    plt.title("Input variables – boxplot")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    export_figure(fig, out_dir, "input_boxplot")

    # ======================================================
    # 3. BOXPLOT – OUTPUT
    # ======================================================
    fig = plt.figure(figsize=(max(8, 0.6 * len(state_vars)), 4))
    sns.boxplot(data=df[state_vars], orient="v")
    plt.xticks(rotation=45, ha="right")
    plt.title("Output variables – boxplot")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    export_figure(fig, out_dir, "output_boxplot")

    # ======================================================
    # 4. HISTOGRAMS – INPUT / OUTPUT
    # ======================================================
    def plot_histograms(cols, tag):
        n = len(cols)
        ncols = 3
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.5 * nrows),
            squeeze=False
        )

        for i, c in enumerate(cols):
            r, k = divmod(i, ncols)
            ax = axes[r, k]
            sns.histplot(df[c], bins=40, kde=True, ax=ax)
            ax.set_title(c, fontsize=10)
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, nrows * ncols):
            r, k = divmod(j, ncols)
            axes[r, k].axis("off")

        plt.tight_layout()
        export_figure(fig, out_dir, f"{tag}_histograms")

    plot_histograms(process_features, "input")
    plot_histograms(state_vars, "output")
