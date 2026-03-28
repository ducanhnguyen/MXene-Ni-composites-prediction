# src/analysis/correlation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"

def wrap_dislocation_label(text):
    if "<" not in text:
        return wrap_label(text, max_len=14, max_lines=3)

    # Split at '<'
    main, suffix = text.split("<", 1)
    suffix = "<" + suffix  # add back '<'

    # Further split main into 2 lines
    parts = main.replace(".", " ").split()
    mid = len(parts) // 2

    line1 = " ".join(parts[:mid])
    line2 = " ".join(parts[mid:])

    return "\n".join([line1, line2, suffix])


def wrap_label(text, max_len=14, max_lines=3):
    """
    Insert line breaks into long labels.
    """
    if len(text) <= max_len:
        return text

    words = text.split(" ")
    lines = []
    cur = ""

    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w

        # Stop if reaching max_lines
        if len(lines) == max_lines - 1:
            break

    # Remaining words → last line
    remaining = " ".join(words[len(" ".join(lines + [cur]).split(" ")):])
    if remaining:
        cur = f"{cur} {remaining}".strip()

    lines.append(cur)

    return "\n".join(lines[:max_lines])


def export_and_plot_input_output_correlation(
    df,
    process_features,
    state_vars,
    label_map=None,
    out_dir="out",
    method="spearman",     # spearman, pearson
    p_thresholds=(0.05, 0.01),
    min_samples=3
):
    os.makedirs(out_dir, exist_ok=True)

    corr_mat = pd.DataFrame(
        index=process_features,
        columns=state_vars,
        dtype=float
    )

    pval_mat = pd.DataFrame(
        index=process_features,
        columns=state_vars,
        dtype=float
    )

    # ======================================================
    # Compute correlation + p-value (SAFE VERSION)
    # ======================================================
    for x in process_features:
        for y in state_vars:
            x_val = df[x].values
            y_val = df[y].values

            # Pairwise finite mask (remove NaN/Inf)
            mask = np.isfinite(x_val) & np.isfinite(y_val)
            x_clean = x_val[mask]
            y_clean = y_val[mask]

            # Check 1: Too few samples
            if len(x_clean) < min_samples:
                r, p = np.nan, np.nan

            # Check 2: Constant feature (use small tolerance)
            elif np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                r, p = np.nan, np.nan

            # Check 3: All values identical (extra safety)
            elif len(np.unique(x_clean)) == 1 or len(np.unique(y_clean)) == 1:
                r, p = np.nan, np.nan

            else:
                try:
                    if method == "pearson":
                        r, p = pearsonr(x_clean, y_clean)
                    elif method == "spearman":
                        r, p = spearmanr(x_clean, y_clean)
                    else:
                        raise ValueError("method must be 'pearson' or 'spearman'")
                except Exception as e:
                    print(f"Warning: Failed to compute {method} for {x} vs {y}: {e}")
                    r, p = np.nan, np.nan

            corr_mat.loc[x, y] = r
            pval_mat.loc[x, y] = p

    # ======================================================
    # Save numeric results
    # ======================================================
    corr_csv = os.path.join(out_dir, f"correlation_{method}.csv")
    pval_csv = os.path.join(out_dir, f"pvalue_{method}.csv")

    corr_mat.to_csv(corr_csv)
    pval_mat.to_csv(pval_csv)

    print(f"[OK] Saved correlation → {corr_csv}")
    print(f"[OK] Saved p-values   → {pval_csv}")

    # ======================================================
    # Build annotation with significance stars
    # ======================================================
    def p_to_star(p):
        if np.isnan(p):
            return ""
        if p < p_thresholds[1]:      # p < 0.01
            return "**"
        elif p < p_thresholds[0]:    # p < 0.05
            return "*"
        else:
            return ""

    annot = pd.DataFrame(
        "",
        index=corr_mat.index,
        columns=corr_mat.columns
    )

    for i in corr_mat.index:
        for j in corr_mat.columns:
            r = corr_mat.loc[i, j]
            p = pval_mat.loc[i, j]

            if np.isnan(r):
                annot.loc[i, j] = ""
            else:
                annot.loc[i, j] = f"{r:.2f}{p_to_star(p)}"

    # ======================================================
    # Plot heatmap (FIXED – no cropping)
    # ======================================================
    fig, ax = plt.subplots(
        figsize=(1.2 * len(state_vars), 0.9 * len(process_features))
    )

    sns.heatmap(
        corr_mat.astype(float),
        annot=annot,
        fmt="",
        cmap="coolwarm_r",     # -1 đỏ, +1 xanh đậm
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": f"{method.title()} correlation"},
        ax=ax
    )

    # Build display labels (X: wrap tối đa 3 dòng)
    x_labels = []
    for v in state_vars:
        label = label_map.get(v, v) if label_map else v

        if label.startswith("Dislocation"):
            x_labels.append(wrap_dislocation_label(label))
        else:
            x_labels.append(wrap_label(label, max_len=14, max_lines=3))

    y_labels = [
        wrap_label(label_map.get(v, v), max_len=18, max_lines=2)
        if label_map else wrap_label(v, max_len=18, max_lines=2)
        for v in process_features
    ]

    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)

    ax.set_xticklabels(
        x_labels,
        rotation=0,
        ha="center",
        fontsize=10
    )
    ax.set_yticklabels(
        y_labels,
        rotation=0,
        fontsize=10
    )

    # Caption / legend dưới hình (nhỏ, căn trái)
    fig.text(
        0.01, 0.02,  # (x, y) trong figure coordinates
        "* p < 0.05,  ** p < 0.01",
        ha="left",
        va="bottom",
        fontsize=8
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.26,  # tăng từ 0.22 → 0.26
        left=0.18,
        right=0.95
    )

    fig_path = os.path.join(out_dir, f"correlation_{method}_with_p.png")
    fig.savefig(fig_path, dpi=300, bbox_inches=None)  # Don't use tight_layout
    plt.close(fig)

    print(f"[OK] Saved correlation plot → {fig_path}")

    # ======================================================
    # Summary statistics
    # ======================================================
    n_total = len(process_features) * len(state_vars)
    n_valid = corr_mat.notna().sum().sum()
    n_sig_05 = (pval_mat < 0.05).sum().sum()
    n_sig_01 = (pval_mat < 0.01).sum().sum()

    print(f"\n[Summary]")
    print(f"  Total pairs:        {n_total}")
    print(f"  Valid correlations: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"  Significant p<0.05: {n_sig_05} ({n_sig_05/n_valid*100:.1f}% of valid)")
    print(f"  Significant p<0.01: {n_sig_01} ({n_sig_01/n_valid*100:.1f}% of valid)")