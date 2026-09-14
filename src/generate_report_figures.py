"""Generate report-ready figures and tables for the GraphSAGE study."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "results" / "graph"
FIGURES_DIR = GRAPH_DIR / "figures"
TABLES_DIR = GRAPH_DIR / "tables"

COMPARISON_PATH = GRAPH_DIR / "comparison_results.csv"
COURSE_VARIATION_PATH = GRAPH_DIR / "course_variation.csv"
ABLATION_PATH = GRAPH_DIR / "ablation_results.csv"
MODULE_SUMMARY_PATH = GRAPH_DIR / "lcpo_module_summary.csv"
DIAGNOSTICS_PATH = GRAPH_DIR / "course_diagnostics.csv"

METRICS = ["auroc", "auprc", "f1", "precision", "recall", "balanced_acc"]
WEEKS = [2, 4, 6, 8]

sns.set_theme(style="whitegrid")


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def empty_series() -> pd.Series:
    return pd.Series(dtype=float)


def mean_std(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean.iloc[0]), np.nan
    return float(clean.mean()), float(clean.std(ddof=1))


def fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| |\n|---|\n"
    headers = [str(col) for col in df.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in df.fillna("").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, stem: str) -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df))


def plot_grouped_bars(
    ax: plt.Axes,
    categories: list[str],
    series_specs: list[tuple[str, list[float], list[float], str]],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(categories))
    width = 0.24 if len(series_specs) >= 3 else 0.32

    for idx, (label, means, stds, color) in enumerate(series_specs):
        offset = (idx - (len(series_specs) - 1) / 2) * width
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds, dtype=float)
        valid = ~np.isnan(means_arr)
        if not valid.any():
            continue
        ax.bar(
            x[valid] + offset,
            means_arr[valid],
            width=width,
            label=label,
            color=color,
            yerr=np.where(np.isnan(stds_arr[valid]), 0.0, stds_arr[valid]),
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True)


def make_week_performance_figure(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    weighted_means, weighted_stds = [], []
    unweighted_means, unweighted_stds = [], []
    lgbm_means, lgbm_stds = [], []

    for week in WEEKS:
        weighted = comparison_df[
            (comparison_df["week"] == week)
            & (comparison_df["split_type"] == "random_student")
            & (comparison_df["model"] == "GNN (weighted)")
        ]
        unweighted = comparison_df[
            (comparison_df["week"] == week)
            & (comparison_df["split_type"] == "random_student")
            & (comparison_df["model"] == "GNN (unweighted)")
        ]
        lgbm = comparison_df[
            (comparison_df["week"] == week)
            & (comparison_df["split_type"] == "random_student")
            & (comparison_df["model"] == "LightGBM")
        ]

        weighted_mean, weighted_std = mean_std(weighted["auroc"] if not weighted.empty and "auroc" in weighted else empty_series())
        unweighted_mean, unweighted_std = mean_std(unweighted["auroc"] if not unweighted.empty and "auroc" in unweighted else empty_series())
        lgbm_mean, lgbm_std = mean_std(lgbm["auroc"] if "auroc" in lgbm else empty_series())

        weighted_means.append(weighted_mean)
        weighted_stds.append(weighted_std)
        unweighted_means.append(unweighted_mean)
        unweighted_stds.append(unweighted_std)
        lgbm_means.append(lgbm_mean)
        lgbm_stds.append(lgbm_std)

        rows.append(
            {
                "week": week,
                "GNN weighted": fmt_mean_std(weighted_mean, weighted_std),
                "GNN unweighted": fmt_mean_std(unweighted_mean, unweighted_std),
                "LightGBM": fmt_mean_std(lgbm_mean, lgbm_std),
            }
        )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_grouped_bars(
        ax,
        [str(week) for week in WEEKS],
        [
            ("GNN weighted", weighted_means, weighted_stds, "#3b82d4"),
            ("GNN unweighted", unweighted_means, unweighted_stds, "#7c5cd8"),
            ("LightGBM", lgbm_means, lgbm_stds, "#57606a"),
        ],
        ylabel="AUROC",
        title="Early-prediction performance by week",
    )
    ax.set_xlabel("Prediction Week")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_week_performance.png", dpi=150)
    plt.close(fig)

    return pd.DataFrame(rows)


def make_random_vs_lcpo_figure(comparison_df: pd.DataFrame) -> None:
    series = []
    for model, label, color in [("GNN", "GNN", "#3b82d4"), ("LightGBM", "LightGBM", "#57606a")]:
        means = []
        stds = []
        for split in ["random_student", "lcpo"]:
            if split == "random_student" and model == "GNN":
                subset = comparison_df[
                    (comparison_df["week"] == 8)
                    & (comparison_df["split_type"] == split)
                    & (comparison_df["model"] == "GNN (weighted)")
                ]
            else:
                subset = comparison_df[
                    (comparison_df["week"] == 8)
                    & (comparison_df["split_type"] == split)
                    & (comparison_df["model"] == model)
                ]
            mean, std = mean_std(subset["auroc"] if "auroc" in subset else empty_series())
            means.append(mean)
            stds.append(std)
        series.append((label, means, stds, color))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    plot_grouped_bars(
        ax,
        ["Random", "LCPO"],
        series,
        ylabel="AUROC",
        title="Random split vs. LCPO at week 8",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_random_vs_lcpo.png", dpi=150)
    plt.close(fig)


def make_course_variation_figure(course_df: pd.DataFrame) -> pd.DataFrame:
    if course_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Per-course AUROC at week 8 (LCPO)")
        ax.set_xlabel("AUROC")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig_course_variation.png", dpi=150)
        plt.close(fig)
        return course_df

    ordered = course_df.sort_values("lgbm_auroc", ascending=True).copy()
    ordered["course"] = ordered["held_out_module"] + "-" + ordered["held_out_presentation"]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(ordered))
    height = 0.38
    ax.barh(y - height / 2, ordered["gnn_auroc"], height=height, color="#3b82d4", label="GNN")
    ax.barh(y + height / 2, ordered["lgbm_auroc"], height=height, color="#57606a", label="LightGBM")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["course"])
    ax.set_xlabel("AUROC")
    ax.set_title("Per-course AUROC at week 8 (LCPO)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_course_variation.png", dpi=150)
    plt.close(fig)

    top_wins = course_df.head(5)
    top_losses = course_df.tail(5).sort_values("auroc_delta", ascending=True)
    return pd.concat([top_wins, top_losses], ignore_index=True)


def make_ablation_figure(ablation_df: pd.DataFrame) -> pd.DataFrame:
    if ablation_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.set_xlabel("Condition")
        ax.set_ylabel("AUROC")
        ax.set_title("Ablation AUROC by condition")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig_ablation.png", dpi=150)
        plt.close(fig)
        return pd.DataFrame(columns=["condition", "auroc", "auprc", "f1"])

    summary = (
        ablation_df.groupby("condition")[["auroc", "auprc", "f1"]]
        .mean()
        .reset_index()
        .sort_values("auroc", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=summary, x="condition", y="auroc", color="#3b82d4", ax=ax)
    ax.set_xlabel("Condition")
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation AUROC by condition")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ablation.png", dpi=150)
    plt.close(fig)

    return summary


def make_module_summary_figure(module_df: pd.DataFrame) -> None:
    """Horizontal grouped bar chart: one bar pair (GNN vs LightGBM) per module.

    Bars are sorted by GNN AUROC descending.  Error bars show ± 1 std.
    Saves to results/graph/figures/fig_module_summary.png.

    Parameters
    ----------
    module_df : pd.DataFrame
        Output of summarize_lcpo_modules.build_module_summary() —
        columns: held_out_module, model, auroc_mean, auroc_std, …
    """
    if module_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("LCPO AUROC by Module (GNN vs LightGBM)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fig_module_summary.png", dpi=150)
        plt.close(fig)
        return

    # Pivot to wide format: one row per module, columns gnn_auroc / lgbm_auroc
    gnn = module_df[module_df["model"] == "GNN"][
        ["held_out_module", "auroc_mean", "auroc_std"]
    ].rename(columns={"auroc_mean": "gnn_auroc", "auroc_std": "gnn_std"})

    lgbm = module_df[module_df["model"] == "LightGBM"][
        ["held_out_module", "auroc_mean", "auroc_std"]
    ].rename(columns={"auroc_mean": "lgbm_auroc", "auroc_std": "lgbm_std"})

    wide = gnn.merge(lgbm, on="held_out_module", how="outer").sort_values(
        "gnn_auroc", ascending=True  # ascending for horizontal barh (top = highest)
    ).reset_index(drop=True)

    modules = wide["held_out_module"].tolist()
    n = len(modules)
    y = np.arange(n)
    height = 0.35

    gnn_means = wide["gnn_auroc"].fillna(0).tolist()
    gnn_stds = wide["gnn_std"].fillna(0).tolist()
    lgbm_means = wide["lgbm_auroc"].fillna(0).tolist()
    lgbm_stds = wide["lgbm_std"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, 1.0 * n)))

    bars_gnn = ax.barh(
        y + height / 2, gnn_means, height,
        xerr=gnn_stds, label="GNN", color="#3b82d4",
        error_kw={"elinewidth": 1.2, "capsize": 3},
    )
    bars_lgbm = ax.barh(
        y - height / 2, lgbm_means, height,
        xerr=lgbm_stds, label="LightGBM", color="#7c5cd8",
        error_kw={"elinewidth": 1.2, "capsize": 3},
    )

    ax.set_yticks(y)
    ax.set_yticklabels(modules)
    ax.set_xlabel("AUROC (mean ± 1 std across seeds × folds)")
    ax.set_title("LCPO AUROC by Module — GNN vs LightGBM")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_module_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def make_course_diagnostics_figure(diag_df: pd.DataFrame) -> None:
    """2×3 grid of scatter plots: each diagnostic factor vs. GNN AUROC mean.

    Points are coloured by module; a linear trendline is added per panel.
    Saves to results/graph/figures/fig_course_diagnostics.png.

    Parameters
    ----------
    diag_df : pd.DataFrame
        Output of course_diagnostics.build_course_diagnostics() — must contain
        columns: gnn_auroc_mean, held_out_module, and at least one factor column.
    """
    factors = [
        ("n_test", "Sample size (n_test)"),
        ("at_risk_rate", "At-risk rate"),
        ("n_assessments", "# assessments"),
        ("mean_clicks_per_student", "Mean VLE clicks / student"),
        ("student_overlap_rate", "Student overlap rate"),
        ("dist_shift", "Distribution shift (cosine)"),
    ]

    modules = sorted(diag_df["held_out_module"].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=max(len(modules), 1))
    mod_colour = {m: palette[i] for i, m in enumerate(modules)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()

    for ax, (col, label) in zip(axes_flat, factors):
        if col not in diag_df.columns:
            ax.set_visible(False)
            continue

        subset = diag_df[["held_out_module", col, "gnn_auroc_mean"]].dropna()

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        for mod in modules:
            m_data = subset[subset["held_out_module"] == mod]
            ax.scatter(
                m_data[col],
                m_data["gnn_auroc_mean"],
                label=mod,
                color=mod_colour[mod],
                s=60,
                alpha=0.85,
                zorder=3,
            )

        # Linear trendline across all modules
        x_vals = subset[col].values
        y_vals = subset["gnn_auroc_mean"].values
        if len(x_vals) >= 2 and np.std(x_vals) > 0:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="#57606a", lw=1.5, ls="--", zorder=2)

            # Pearson r annotation
            r = float(np.corrcoef(x_vals, y_vals)[0, 1])
            ax.annotate(
                f"r = {r:.2f}",
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                fontsize=9,
                color="#57606a",
            )

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("GNN AUROC (mean)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    # Shared legend (modules)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=mod_colour[m],
                   markersize=8, label=m)
        for m in modules
    ]
    fig.legend(handles=handles, title="Module", loc="lower center",
               ncol=len(modules), bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(
        "Course-Level Diagnostic Factors vs. GNN AUROC (LCPO)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = FIGURES_DIR / "fig_course_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def make_main_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_type, split_label, model_names in [
        ("random_student", "Random", ["GNN (weighted)", "LightGBM"]),
        ("lcpo", "LCPO", ["GNN", "LightGBM"]),
    ]:
        for model_name in model_names:
            subset = comparison_df[
                (comparison_df["week"] == 8)
                & (comparison_df["split_type"] == split_type)
                & (comparison_df["model"] == model_name)
            ]
            row = {"split": split_label, "model": model_name}
            for metric in METRICS:
                mean, std = mean_std(subset[metric] if metric in subset else empty_series())
                row[metric.upper()] = fmt_mean_std(mean, std)
            rows.append(row)
    return pd.DataFrame(rows)


def make_cross_dataset_figure(combined_df: pd.DataFrame) -> None:
    """Produce a faceted AUROC comparison figure for OULAD vs Zenodo.

    Parameters
    ----------
    combined_df : DataFrame with columns: dataset, model, split_type, auroc.
        Expected to have at least 2 unique dataset values and both
        "random_student" and "lcpo" split_types.

    Outputs
    -------
    results/graph/figures/fig_cross_dataset.png
    """
    split_types = ["random_student", "lcpo"]
    split_labels = {"random_student": "Random student", "lcpo": "LCPO"}
    datasets = sorted(combined_df["dataset"].unique())
    models = combined_df["model"].unique()

    n_splits = len(split_types)
    fig, axes = plt.subplots(1, n_splits, figsize=(10, 4.5), sharey=True)
    if n_splits == 1:
        axes = [axes]

    palette = sns.color_palette("muted", n_colors=len(models))
    model_colors = dict(zip(models, palette))

    bar_width = 0.35
    dataset_positions = {d: i for i, d in enumerate(datasets)}

    for ax, stype in zip(axes, split_types):
        subset = combined_df[combined_df["split_type"] == stype]
        for m_idx, model in enumerate(models):
            model_sub = subset[subset["model"] == model]
            means, stds, positions = [], [], []
            for ds in datasets:
                ds_sub = model_sub[model_sub["dataset"] == ds]["auroc"]
                m, s = mean_std(ds_sub)
                means.append(m if not np.isnan(m) else 0.0)
                stds.append(s if not np.isnan(s) else 0.0)
                positions.append(dataset_positions[ds] + (m_idx - 0.5 * (len(models) - 1)) * bar_width)
            ax.bar(
                positions, means, bar_width,
                yerr=stds, capsize=4,
                color=model_colors[model], label=model, alpha=0.85,
            )
        ax.set_title(split_labels.get(stype, stype), fontsize=11)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=10)
        ax.set_ylim(0.45, 0.90)
        ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", label="Random (0.5)")
        ax.set_xlabel("Dataset")
        if ax is axes[0]:
            ax.set_ylabel("AUROC")
            ax.legend(fontsize=9)

    fig.suptitle("Cross-dataset AUROC: OULAD vs Zenodo (KU Leuven)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_cross_dataset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# Zenodo results paths (populated by run_zenodo_pipeline.py / run_zenodo_lgbm_only.py)
ZENODO_RESULTS_DIR = PROJECT_ROOT / "results" / "zenodo"
ZENODO_GNN_RANDOM_PATH = ZENODO_RESULTS_DIR / "random_student_results.csv"
ZENODO_LGBM_PATH = ZENODO_RESULTS_DIR / "comparison_results.csv"


def main() -> None:
    ensure_dirs()

    if not COMPARISON_PATH.exists():
        print(
            "ERROR: comparison_results.csv not found. "
            "Run compare_gnn_lgbm.py first to generate comparison_results.csv"
        )
        sys.exit(1)

    comparison_df = load_csv(COMPARISON_PATH)
    course_df = load_csv(COURSE_VARIATION_PATH)
    ablation_df = load_csv(ABLATION_PATH)

    week_table = make_week_performance_figure(comparison_df)
    make_random_vs_lcpo_figure(comparison_df)
    course_table = make_course_variation_figure(course_df)
    ablation_table = make_ablation_figure(ablation_df)
    main_comparison_table = make_main_comparison_table(comparison_df)

    save_table(main_comparison_table, "table_main_comparison")
    save_table(week_table, "table_week_performance")
    save_table(ablation_table, "table_ablation")
    save_table(course_table, "table_course_variation")

    # Module summary figure (guard: only if the CSV exists)
    if MODULE_SUMMARY_PATH.exists():
        module_df = load_csv(MODULE_SUMMARY_PATH)
        make_module_summary_figure(module_df)

    # Course diagnostics figure (guard: only if the CSV exists)
    if DIAGNOSTICS_PATH.exists():
        diag_df = load_csv(DIAGNOSTICS_PATH)
        make_course_diagnostics_figure(diag_df)

    # Cross-dataset figure (guard: only if Zenodo results exist)
    if ZENODO_GNN_RANDOM_PATH.exists() and ZENODO_LGBM_PATH.exists():
        # Build combined DataFrame: OULAD rows from comparison_results.csv
        # (week 8 only) + Zenodo GNN random + Zenodo LightGBM
        oulad_rows = comparison_df[comparison_df["week"] == 8].copy()
        oulad_rows["dataset"] = "oulad"
        # OULAD comparison_results uses "model" and "split" columns
        if "split" in oulad_rows.columns and "split_type" not in oulad_rows.columns:
            oulad_rows = oulad_rows.rename(columns={"split": "split_type"})

        zen_gnn = load_csv(ZENODO_GNN_RANDOM_PATH)
        zen_lgbm = load_csv(ZENODO_LGBM_PATH)
        zen_combined = pd.concat([zen_gnn, zen_lgbm], ignore_index=True)

        # Normalise model column name for Zenodo LightGBM rows
        if "model" not in zen_combined.columns:
            zen_combined["model"] = "LightGBM"

        cross_df = pd.concat(
            [
                oulad_rows[["dataset", "model", "split_type", "auroc"]],
                zen_combined[["dataset", "model", "split_type", "auroc"]],
            ],
            ignore_index=True,
        )
        make_cross_dataset_figure(cross_df)
    else:
        print("  Skipping cross-dataset figure (Zenodo results not found)")

    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
