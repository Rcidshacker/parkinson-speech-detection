"""
Multi-Dataset Feature Ranking & Consensus Merge
================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Generated for User

PURPOSE:
    Dynamically analyzes unified feature CSVs containing multiple datasets.
    1. Reads the feature CSV
    2. Detects available datasets (e.g., pc_gita, italian, voice_dataset)
    3. Ranks features independently for EACH dataset using Mann-Whitney U & AUC
    4. Extracts the Top-N features from each
    5. Computes the N-way consensus (intersection/union) to find globally stable features
    6. Generates full reports and visual plots separately for each CSV input

INPUTS (default — all three feature sets run in sequence):
    - features/features_sustained_a.csv      (handcrafted, 112 features)
    - features/features_egemaps_8k.csv       (eGeMAPS, 88 features)
    - features/features_compare_8k.csv       (ComParE16, 6373 features)

OVERRIDE (single CSV via CLI):
    python pipeline/per_dataset_ranking.py --input features/features_compare_8k.csv

OUTPUTS:
    results/final/ranking_<feature_set>_<timestamp>/
    ├── <dataset>_ranking.csv           (per dataset)
    ├── ranking_comparison.csv
    ├── merged_consensus_list.csv
    └── ranking_plots.png
"""

import os
import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
LOG_DIR = os.path.join(BASE, "logs")

# Default: ComParE 8kHz only (our active scope).
# Pass --input to override and run a single CSV instead.
CSV_FILES = [
    os.path.join(BASE, "features", "opensmile", "features_compare_8k.csv"),
]

# Meta columns to strictly exclude from statistical feature ranking
META_COLS = {
    "dataset",
    "subject_id",
    "language",
    "gender",
    "speech_type",
    "disease_label",
    "label_binary",
    "multiclass_label",
    "label",
    "updrs_total",
    "moca_score",
    "meds_status",
    "file",
    "rms_original",
    "rms_after",
    "rms_target",
    "rms_clipped",
    "duration_s",
    "vowel_letter",
    "attempt_num",
    "speaker_id",
    "rpde",
    "dfa",
    "ppe",
}


# ============================================================================
# FEATURE RANKING ALGORITHM
# ============================================================================
def _score_feature(feat, pd_vals, hc_vals, all_vals, labels):
    if len(pd_vals) < 5 or len(hc_vals) < 5:
        return None
    _, p_val = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")
    direction = "PD>HC" if pd_vals.mean() > hc_vals.mean() else "PD<HC"
    signed_vals = all_vals if direction == "PD>HC" else -all_vals
    valid_mask = ~np.isnan(all_vals)
    try:
        auc = roc_auc_score(labels[valid_mask], signed_vals[valid_mask])
    except Exception:
        return None
    return {
        "feature": feat,
        "auc": round(auc, 4),
        "p_value": round(p_val, 6),
        "pd_mean": round(pd_vals.mean(), 6),
        "hc_mean": round(hc_vals.mean(), 6),
        "direction": direction,
    }


def rank_features(df, dataset_name, log):
    """Rank all features for a single dataset using Mann-Whitney U & AUC (parallelized)."""
    label_col = "label_binary" if "label_binary" in df.columns else "label"
    feat_cols = [
        c
        for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    log.info(
        f"  [{dataset_name}] {len(df)} rows | PD={df[label_col].sum()} HC={(df[label_col] == 0).sum()} | Features={len(feat_cols)}"
    )

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_score_feature)(
            feat,
            df[df[label_col] == 1][feat].dropna().values,
            df[df[label_col] == 0][feat].dropna().values,
            df[feat].values,
            df[label_col].values,
        )
        for feat in feat_cols
    )

    ranking = [r for r in results if r is not None]

    rank_df = (
        pd.DataFrame(ranking).sort_values("auc", ascending=False).reset_index(drop=True)
    )
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))

    return rank_df


# ============================================================================
# MULTI-DATASET MERGING & CONSENSUS
# ============================================================================
def build_comparison_table(rankings, n=30):
    """Build side-by-side rank comparison for top-N from each dataset."""
    cols = []
    for ds, rank_df in rankings.items():
        top_df = rank_df.head(n)[["rank", "feature", "auc"]].copy()
        top_df.columns = [f"{ds}_rank", f"{ds}_feature", f"{ds}_auc"]
        cols.append(top_df.reset_index(drop=True))
    return pd.concat(cols, axis=1)


def merge_top_n_multi(rankings, top_n):
    """Find union and intersection of Top-N features across ALL datasets."""
    datasets = list(rankings.keys())
    top_sets = {ds: set(rankings[ds].head(top_n)["feature"]) for ds in datasets}

    union_features = set.union(*top_sets.values())
    intersection_features = set.intersection(*top_sets.values())

    rows = []
    for feat in union_features:
        in_datasets = [ds for ds in datasets if feat in top_sets[ds]]

        row = {
            "feature": feat,
            "dataset_count": len(in_datasets),
            "in_datasets": " | ".join(in_datasets),
            "is_common_all": len(in_datasets) == len(datasets),
        }

        aucs = []
        for ds in datasets:
            ds_rank = rankings[ds]
            ds_row = ds_rank[ds_rank["feature"] == feat]

            if not ds_row.empty:
                auc = float(ds_row["auc"].values[0])
                row[f"{ds}_rank"] = int(ds_row["rank"].values[0])
                row[f"{ds}_auc"] = auc
                if feat in top_sets[ds]:
                    aucs.append(auc)
            else:
                row[f"{ds}_rank"] = None
                row[f"{ds}_auc"] = None

        row["mean_auc_in_top"] = round(np.mean(aucs), 4) if aucs else 0.0
        rows.append(row)

    merged_df = (
        pd.DataFrame(rows)
        .sort_values(["dataset_count", "mean_auc_in_top"], ascending=[False, False])
        .reset_index(drop=True)
    )

    summary = {
        "top_n": top_n,
        "datasets": datasets,
        "common_all": len(intersection_features),
        "merged_total": len(union_features),
    }
    return merged_df, summary


# ============================================================================
# DYNAMIC PLOTTING
# ============================================================================
def make_plots_multi(rankings, merged_df, summary, top_n, out_path, feature_set_name):
    datasets = summary["datasets"]
    n_ds = len(datasets)

    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(
        f"Consensus Feature Ranking: {feature_set_name} | Top-{top_n} per Dataset\n"
        f"{n_ds} Datasets Analyzed | {summary['common_all']} Holy Grail (Common) Features | {summary['merged_total']} Total Unique Features",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    common_set = set(merged_df[merged_df["is_common_all"] == True]["feature"])
    for i, ds in enumerate(datasets):
        ax = fig.add_subplot(2, max(3, n_ds), i + 1)
        top20 = rankings[ds].head(20)

        colors = ["#7B1FA2" if f in common_set else "#90CAF9" for f in top20["feature"]]

        ax.barh(range(len(top20)), top20["auc"], color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f[:25] for f in top20["feature"]], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{ds} — Top-20", fontweight="bold", fontsize=11)
        ax.set_xlabel("Univariate AUC")

        p1 = mpatches.Patch(color="#7B1FA2", label="Common in ALL datasets")
        p2 = mpatches.Patch(color="#90CAF9", label="Dataset specific")
        ax.legend(handles=[p1, p2], fontsize=8, loc="lower right")

    ax_venn = fig.add_subplot(2, 2, 3)
    top_sets = [set(rankings[ds].head(top_n)["feature"]) for ds in datasets]

    try:
        from matplotlib_venn import venn2, venn3

        if n_ds == 2:
            venn2(top_sets, set_labels=datasets, ax=ax_venn)
        elif n_ds == 3:
            venn3(top_sets, set_labels=datasets, ax=ax_venn)
        ax_venn.set_title(f"Top-{top_n} Overlap", fontweight="bold", fontsize=12)
    except ImportError:
        ax_venn.text(
            0.5,
            0.5,
            "matplotlib-venn not installed.\nCannot render Venn diagram.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_venn.axis("off")

    ax_sweep = fig.add_subplot(2, 2, 4)
    n_vals = [3, 5, 7, 9, 12, 15, 20, 25, 30]
    overlaps, totals = [], []
    for n in n_vals:
        t_sets = [set(rankings[ds].head(n)["feature"]) for ds in datasets]
        overlaps.append(len(set.intersection(*t_sets)))
        totals.append(len(set.union(*t_sets)))

    ax_sweep.plot(
        n_vals[: len(overlaps)],
        overlaps,
        "-o",
        color="#7B1FA2",
        linewidth=2.5,
        label="Common in ALL",
    )
    ax_sweep_twin = ax_sweep.twinx()
    ax_sweep_twin.plot(
        n_vals[: len(totals)],
        totals,
        "-s",
        color="#F57C00",
        linewidth=2.5,
        label="Total Merged Set",
    )

    ax_sweep.axvline(
        top_n, color="red", linestyle="--", label=f"Current Selection (N={top_n})"
    )
    ax_sweep.set_xlabel("Top-N Selected per Dataset", fontsize=10)
    ax_sweep.set_ylabel("Count of Universal Features", color="#7B1FA2", fontsize=10)
    ax_sweep_twin.set_ylabel(
        "Total Unique Features in Merge", color="#F57C00", fontsize=10
    )
    ax_sweep.set_title(
        "Cross-Dataset Feature Stability\n(Helps determine optimal N)",
        fontweight="bold",
        fontsize=12,
    )

    lines, labels = ax_sweep.get_legend_handles_labels()
    lines2, labels2 = ax_sweep_twin.get_legend_handles_labels()
    ax_sweep.legend(lines + lines2, labels + labels2, loc="upper left")
    ax_sweep.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# PROCESSOR ENGINE
# ============================================================================
def process_feature_set(csv_path, top_n):
    feature_set_name = (
        os.path.basename(csv_path).replace(".csv", "").replace("features_", "")
    )

    out_dir = os.path.join(
        BASE,
        "results",
        "final",
        f"ranking_{feature_set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "ranking_report.log")
    log = logging.getLogger(feature_set_name)
    log.setLevel(logging.INFO)
    if log.hasHandlers():
        log.handlers.clear()
    log.addHandler(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    log.addHandler(logging.StreamHandler(sys.stdout))

    log.info("=" * 80)
    log.info(f"PROCESSING FEATURE SET: {feature_set_name.upper()}")
    log.info(f"Target Top-N: {top_n}")
    log.info("=" * 80)

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["dataset"] != "italian"]
    datasets = df["dataset"].unique().tolist()
    log.info(f"\n[STEP 1] Detected {len(datasets)} datasets: {datasets}")

    log.info("\n[STEP 2] Independent Dataset Ranking...")
    rankings = {}
    for ds in datasets:
        sub_df = df[df["dataset"] == ds]
        if len(sub_df) < 10:
            log.warning(f"  [{ds}] Skipping - Insufficient samples.")
            continue
        rankings[ds] = rank_features(sub_df, ds, log)
        rankings[ds].to_csv(os.path.join(out_dir, f"{ds}_ranking.csv"), index=False)

    log.info("\n[STEP 3] Generating Side-by-Side Comparison...")
    comparison = build_comparison_table(rankings, n=30)
    comparison.to_csv(os.path.join(out_dir, "ranking_comparison.csv"), index=False)

    log.info(f"\n[STEP 4] N-Way Consensus Merge (Top {top_n} per dataset)...")
    merged_df, summary = merge_top_n_multi(rankings, top_n)
    merged_df.to_csv(os.path.join(out_dir, "merged_consensus_list.csv"), index=False)

    log.info(f"  Total Unique Features   : {summary['merged_total']}")
    log.info(f"  'Holy Grail' (in ALL)   : {summary['common_all']}")

    log.info("\n  Top 15 Consensus Features:")
    for _, row in merged_df.head(15).iterrows():
        star = "★ " if row["is_common_all"] else "  "
        log.info(
            f"  {star}{row['feature']:<30} | Present in {row['dataset_count']}/{len(datasets)} datasets | Mean Top-AUC: {row['mean_auc_in_top']:.4f}"
        )

    log.info("\n[STEP 5] Generating Visual Reports...")
    plot_path = os.path.join(out_dir, "ranking_plots.png")
    try:
        make_plots_multi(
            rankings, merged_df, summary, top_n, plot_path, feature_set_name
        )
        log.info(f"  Plots saved successfully.")
    except Exception as e:
        log.error(f"  Plot generation failed: {e}")

    log.info("\n" + "=" * 80)
    log.info(f"COMPLETE: {feature_set_name.upper()}")
    log.info(f"All reports saved to: {out_dir}")
    log.info("=" * 80 + "\n")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Feature ranking & consensus for PD detection CSVs."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Top-N features to extract per dataset (default: 15)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override: path to a single CSV to rank (skips default CSV_FILES list)",
    )
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # --input overrides the default list; useful for targeted runs
    files_to_run = [args.input] if args.input else CSV_FILES

    for csv_file in files_to_run:
        if not os.path.exists(csv_file):
            print(f"[ERROR] Cannot find CSV: {csv_file}")
            continue
        process_feature_set(csv_file, args.top_n)


if __name__ == "__main__":
    main()
