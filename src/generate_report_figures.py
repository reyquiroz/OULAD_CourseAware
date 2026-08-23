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

    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
