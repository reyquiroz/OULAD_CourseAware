"""
Helper script to load baseline results into a Jupyter notebook
Add this cell before the visualization cell in the notebook
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the baseline results we generated
try:
    results_df = pd.read_csv("baseline_results_detailed.csv")

    # Rename columns to match notebook expectations
    results_df = results_df.rename(
        columns={
            "Week": "window",
            "Model": "model",
            "AUROC_mean": "AUROC",
            "AUPRC_mean": "AUPRC",
            "F1_mean": "F1",
            "Precision_mean": "Precision",
            "Recall_mean": "Recall",
            "Balanced_Acc_mean": "Balanced Acc",
        }
    )

    # Filter for All_features only
    results_df = results_df[results_df["Features"] == "All_features"].copy()

    # Select relevant columns
    results_df = results_df[
        [
            "window",
            "model",
            "AUROC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "Balanced Acc",
        ]
    ]

    print(f"✓ Loaded {len(results_df)} baseline results")
    print(f"Windows: {sorted(results_df['window'].unique())}")
    print(f"Models: {list(results_df['model'].unique())}")

    # Make it available as Results_df for the notebook
    Results_df = results_df.copy()

except FileNotFoundError:
    print("Error: baseline_results_detailed.csv not found")
    print("Please run: python OULAD_baseline_analysis_v5.py")
    Results_df = pd.DataFrame()

# Made with Bob
