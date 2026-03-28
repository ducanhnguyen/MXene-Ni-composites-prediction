# src/utils.py
import math
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def safe_filename(name: str) -> str:
    """
    Replace all non-alphanumeric characters with underscore.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", name)


def plot_case_predictions(
        df_case,
        state_vars,
        case_id,
        label_map=None,
        ncols=3,
        figsize_per_plot=(4.5, 3),
        out_dir = None,
        save=True
):
    if label_map is None:
        label_map = {v: v for v in state_vars}

    n_vars = len(state_vars)
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols,
                 figsize_per_plot[1] * nrows),
        squeeze=False
    )

    # formatter: 1000 → 1k
    def k_formatter(x, pos):
        if x >= 1000:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    for idx, var in enumerate(state_vars):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        ax.plot(
            df_case["Timestep1and2"],
            df_case[var],
            color="black",
            linestyle="-",
            linewidth=0.9,  # ↓ mảnh hơn
            label="Ground Truth"
        )

        # Prediction: blue dashed
        ax.plot(
            df_case["Timestep1and2"],
            df_case[f"{var}_pred"],
            color="red",  # ← đổi sang đỏ
            linestyle="--",
            linewidth=0.9,  # ↓ mảnh hơn
            label="Prediction"
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel(label_map.get(var, var))
        ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_vars, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    # Global legend (bottom)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=12,
        frameon=False
    )

    fig.suptitle(
        f"Prediction vs Ground Truth — {case_id}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # ========================================================
    # SAVE FIGURE (ONE IMAGE PER CASE)
    # ========================================================
    if save:
        import os
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(
            out_dir, f"{case_id}_pred_vs_gt.png"
        )

        plt.savefig(save_path, dpi=300)
        print(f"[OK] Saved figure: {save_path}")

    # plt.show()
def plot_predictions_by_output(
    viz_df,
    output_var,
    label_map=None,
    out_dir=None,
    save=True,
    ncols=3,
    figsize_per_plot=(4.5, 3)
):
    """
    Plot prediction vs ground truth for ONE output across ALL cases,
    using grid layout similar to plot_case_predictions.

    Each subplot = one case.
    """

    if label_map is None:
        label_map = {}

    cases = viz_df["CaseID"].unique()
    n_cases = len(cases)

    nrows = math.ceil(n_cases / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_plot[0] * ncols,
            figsize_per_plot[1] * nrows
        ),
        squeeze=False
    )

    # formatter: 1000 → 1k (same as by-case)
    def k_formatter(x, pos):
        if abs(x) >= 1000:
            return f"{int(x / 1000)}k"
        return f"{int(x)}"

    for idx, case_id in enumerate(cases):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        df_case = viz_df[viz_df["CaseID"] == case_id]

        # Ground Truth — black solid
        ax.plot(
            df_case["Timestep1and2"],
            df_case[output_var],
            color="black",
            linestyle="-",
            linewidth=0.9,
            label="Ground Truth"
        )

        # Prediction — red dashed
        ax.plot(
            df_case["Timestep1and2"],
            df_case[f"{output_var}_pred"],
            color="red",
            linestyle="--",
            linewidth=0.9,
            label="Prediction"
        )

        ax.set_title(case_id, fontsize=11)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(label_map.get(output_var, output_var))
        ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_cases, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    # Global legend (same style as by-case)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=12,
        frameon=False
    )

    fig.suptitle(
        label_map.get(output_var, output_var),
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # ========================================================
    # SAVE
    # ========================================================
    if save:
        os.makedirs(out_dir, exist_ok=True)

        fname = safe_filename(output_var)
        save_path = os.path.join(
            out_dir, f"{fname}_all_cases.png"
        )

        plt.savefig(save_path, dpi=300)
        print(f"[OK] Saved by-output plot (grid): {save_path}")


import os

def export_best_param_txt(
    out_dir,
    method_name,
    grid=None
):
    """
    Export best model parameters to best_param.txt
    """
    path = os.path.join(out_dir, "best_param.txt")

    with open(path, "w") as f:
        f.write(f"Method: {method_name}\n")

        if grid is not None:
            f.write("Best params:\n")
            for k, v in grid.best_params_.items():
                f.write(f"  {k}: {v}\n")

            f.write(
                f"Best CV score: {grid.best_score_:.6f}\n"
            )
        else:
            f.write("Best params: N/A (no grid search)\n")

    print(f"[OK] Saved {path}")


def export_best_result_eval_csv(
    out_dir,
    metrics_df
):
    """
    Export per-output evaluation results to best_result_eval.csv
    """

    path = os.path.join(out_dir, "best_result_eval.csv")

    required_cols = ["State", "R2", "MAE", "MSE", "RMSE", "MAE_%"]
    missing = set(required_cols) - set(metrics_df.columns)
    if missing:
        raise ValueError(
            f"metrics_df missing columns: {missing}"
        )

    metrics_df[required_cols].to_csv(
        path,
        index=False,
        float_format="%.6f"
    )

    print(f"[OK] Saved {path}")
