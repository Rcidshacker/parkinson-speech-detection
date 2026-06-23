"""
RF Feature Importance vs AUC Ranking — Full 112-Feature Analysis
=================================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE:
    Mentor directive (2026-03-23):
    "Check the RF feature importance ranking and compare with the AUC-based
     merged feature list — for PC-GITA, VOICED, and combined."

    This script covers ALL 112 features, not just top-N. The full ranking
    is needed to:
      (a) validate the AUC-based ranking criterion across the entire feature space
      (b) identify the bottom features → feeds CNN 11x10 matrix task (remove 2 lowest)
      (c) show where biomarkers land in the full importance distribution

WHAT THIS SCRIPT DOES:
    1. Trains RF on PC-GITA alone  -> full 112-feature importance ranking (Gini)
    2. Trains RF on VOICED alone   -> full 112-feature importance ranking (Gini)
    3. Trains RF on Combined       -> full 112-feature importance ranking (Gini)
    4. Loads AUC-based per-dataset rankings (all features, not just merged top-9)
    5. Builds master comparison table: all 112 features x all rank sources
    6. Computes convergence metrics at multiple cutoffs (top-9, top-20, top-30)
    7. Identifies consistently bottom features -> CNN matrix input recommendation
    8. Saves full CSVs + 3 plots

OUTPUTS:
    results/final/rf_importance_comparison_<timestamp>/
    |- rf_full_ranking_es.csv          <- All 112 features, RF rank for PC-GITA
    |- rf_full_ranking_it.csv          <- All 112 features, RF rank for VOICED
    |- rf_full_ranking_combined.csv    <- All 112 features, RF rank for combined
    |- master_comparison_all112.csv    <- MAIN FILE: all features x all ranks
    |- bottom_features.csv             <- Candidates to remove for CNN matrix
    |- plot_top30_bars.png             <- Top-30 RF importance bar chart
    |- plot_full_scatter.png           <- AUC rank vs RF rank scatter (all 112)
    `- plot_bottom_features.png        <- Bottom-20 features by RF importance

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\rf_importance_comparison.py
"""

import os
import sys
import glob
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
DATA_FILE   = os.path.join(BASE, "features", "training_top112_113features.csv")
RESULTS_DIR = os.path.join(BASE, "results", "final")
LOG_DIR     = os.path.join(BASE, "logs")
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR     = os.path.join(RESULTS_DIR, f"rf_importance_comparison_{TIMESTAMP}")

META_COLS    = {"dataset", "subject_id", "language", "label_binary"}
RANDOM_STATE = 42
BIOMARKERS   = {
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5",
    "shimmer_apq11", "shimmer_dda", "hnr", "nhr"
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"rf_importance_{TIMESTAMP}.log"),
            mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("rf_importance")

def section(title):
    log.info(""); log.info("=" * 70)
    log.info(f"  {title}"); log.info("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================
section("1. Load Data")

df = pd.read_csv(DATA_FILE)
feat_cols = [c for c in df.columns if c not in META_COLS]

log.info(f"  Rows     : {len(df)}")
log.info(f"  Features : {len(feat_cols)}")
log.info(f"  Datasets : {list(df['dataset'].unique())}")
log.info(f"  PD={df['label_binary'].sum()}  HC={(df['label_binary']==0).sum()}")


# ============================================================
# LOAD AUC-BASED RANKINGS
# ============================================================
section("2. Load AUC-Based Per-Dataset Rankings (all features)")

# Locate merged top-9 list — search ALL ranking dirs for the specific file
merged_pattern = os.path.join(RESULTS_DIR, "per_dataset_ranking_*",
                               "merged_feature_list_top9_14features.csv")
merged_matches = sorted(glob.glob(merged_pattern))

if not merged_matches:
    log.error("  merged_feature_list_top9_14features.csv not found.")
    log.error("  Run: python scripts\\per_dataset_ranking.py --top_n 9")
    sys.exit(1)

merged_path = merged_matches[-1]
latest_dir  = os.path.dirname(merged_path)
log.info(f"  Using ranking folder: {latest_dir}")

auc_es_path = os.path.join(latest_dir, "pcgita_feature_ranking.csv")
auc_it_path = os.path.join(latest_dir, "voiced_feature_ranking.csv")

auc_es_df  = pd.read_csv(auc_es_path)
auc_it_df  = pd.read_csv(auc_it_path)
auc_merged = pd.read_csv(merged_path)

log.info(f"  AUC-ranked ES  : {len(auc_es_df)} features")
log.info(f"  AUC-ranked IT  : {len(auc_it_df)} features")
log.info(f"  AUC merged list: {len(auc_merged)} features in top-9 merged set")

auc_es_rank = dict(zip(auc_es_df["feature"], auc_es_df["rank"]))
auc_it_rank = dict(zip(auc_it_df["feature"], auc_it_df["rank"]))
auc_es_auc  = dict(zip(auc_es_df["feature"], auc_es_df["auc"]))
auc_it_auc  = dict(zip(auc_it_df["feature"], auc_it_df["auc"]))
auc_set     = set(auc_merged["feature"].tolist())


# ============================================================
# RF IMPORTANCE — full ranking, nothing discarded
# ============================================================
def get_rf_importance_full(X, y, feat_names, label):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    imps = pipe.named_steps["rf"].feature_importances_

    imp_df = (pd.DataFrame({"feature": feat_names, f"rf_imp_{label}": imps})
              .sort_values(f"rf_imp_{label}", ascending=False)
              .reset_index(drop=True))
    imp_df.insert(0, f"rf_rank_{label}", range(1, len(imp_df) + 1))

    log.info(f"\n  [{label}] TOP 15:")
    log.info(f"  {'Rank':<7} {'Feature':<32} {'Importance':>10}  Bio?  In Merged?")
    log.info("  " + "-" * 66)
    for _, row in imp_df.head(15).iterrows():
        bio = "B" if row["feature"] in BIOMARKERS else " "
        inm = "Y" if row["feature"] in auc_set else " "
        log.info(f"  {int(row[f'rf_rank_{label}']):<7} {row['feature']:<32} "
                 f"{row[f'rf_imp_{label}']:>10.5f}  {bio:^5} {inm:^9}")

    log.info(f"\n  [{label}] BOTTOM 5:")
    log.info(f"  {'Rank':<7} {'Feature':<32} {'Importance':>10}")
    log.info("  " + "-" * 52)
    for _, row in imp_df.tail(5).iterrows():
        log.info(f"  {int(row[f'rf_rank_{label}']):<7} {row['feature']:<32} "
                 f"{row[f'rf_imp_{label}']:>10.5f}")

    return imp_df


section("3. RF PC-GITA (ES) — All 112 Features")
es_df = df[df["dataset"] == "pc_gita"].copy()
X_es  = es_df[feat_cols].values
y_es  = es_df["label_binary"].values
log.info(f"  Rows: {len(es_df)}  PD={y_es.sum()}  HC={(y_es==0).sum()}")
rf_es = get_rf_importance_full(X_es, y_es, feat_cols, "es")

section("4. RF VOICED (IT) — All 112 Features")
it_df = df[df["dataset"] == "voiced"].copy()
X_it  = it_df[feat_cols].values
y_it  = it_df["label_binary"].values
log.info(f"  Rows: {len(it_df)}  PD={y_it.sum()}  HC={(y_it==0).sum()}")
rf_it = get_rf_importance_full(X_it, y_it, feat_cols, "it")

section("5. RF Combined — All 112 Features")
X_all = df[feat_cols].values
y_all = df["label_binary"].values
log.info(f"  Rows: {len(df)}  PD={y_all.sum()}  HC={(y_all==0).sum()}")
rf_comb = get_rf_importance_full(X_all, y_all, feat_cols, "comb")


# ============================================================
# MASTER COMPARISON TABLE — all 112 features x all rank sources
# ============================================================
section("6. Master Comparison Table — All 112 Features")

master = (rf_es[["feature", "rf_rank_es", "rf_imp_es"]]
          .merge(rf_it[["feature", "rf_rank_it", "rf_imp_it"]], on="feature")
          .merge(rf_comb[["feature", "rf_rank_comb", "rf_imp_comb"]], on="feature"))

master["auc_rank_es"]   = master["feature"].map(auc_es_rank)
master["auc_rank_it"]   = master["feature"].map(auc_it_rank)
master["auc_val_es"]    = master["feature"].map(auc_es_auc)
master["auc_val_it"]    = master["feature"].map(auc_it_auc)
master["is_biomarker"]  = master["feature"].isin(BIOMARKERS).map({True: "Y", False: ""})
master["in_auc_merged"] = master["feature"].isin(auc_set).map({True: "Y", False: ""})
master["mean_rf_rank"]  = master[["rf_rank_es", "rf_rank_it", "rf_rank_comb"]].mean(axis=1)

master = master.sort_values("mean_rf_rank").reset_index(drop=True)
master.insert(0, "overall_rf_rank", range(1, len(master) + 1))

log.info(f"  {len(master)} rows x {len(master.columns)} columns")
log.info("")
log.info(f"  {'#':>4} {'Feature':<32} {'RF-ES':>6} {'RF-IT':>6} {'RF-Cmb':>7} "
         f"{'AUC-ES':>7} {'AUC-IT':>7} {'Bio':>4} {'Mgd':>4}")
log.info("  " + "-" * 86)
for _, r in master.head(30).iterrows():
    aes = str(int(r["auc_rank_es"])) if pd.notna(r["auc_rank_es"]) else "-"
    ait = str(int(r["auc_rank_it"])) if pd.notna(r["auc_rank_it"]) else "-"
    log.info(f"  {int(r['overall_rf_rank']):>4} {r['feature']:<32} "
             f"{int(r['rf_rank_es']):>6} {int(r['rf_rank_it']):>6} "
             f"{int(r['rf_rank_comb']):>7} {aes:>7} {ait:>7} "
             f"{r['is_biomarker']:>4} {r['in_auc_merged']:>4}")
log.info(f"  ... [{len(master) - 30} more rows — see master_comparison_all112.csv]")


# ============================================================
# CONVERGENCE — multiple cutoffs
# ============================================================
section("7. Convergence Analysis — Multiple Cutoffs")

valid_es = master.dropna(subset=["auc_rank_es"])
valid_it = master.dropna(subset=["auc_rank_it"])
rho_es, p_es = spearmanr(valid_es["rf_rank_es"],   valid_es["auc_rank_es"])
rho_it, p_it = spearmanr(valid_it["rf_rank_it"],   valid_it["auc_rank_it"])
rho_ce, _    = spearmanr(valid_es["rf_rank_comb"],  valid_es["auc_rank_es"])

log.info(f"  Spearman correlation (all {len(feat_cols)} features):")
log.info(f"    RF_ES   vs AUC_ES : rho={rho_es:.3f}  p={p_es:.4f}")
log.info(f"    RF_IT   vs AUC_IT : rho={rho_it:.3f}  p={p_it:.4f}")
log.info(f"    RF_Comb vs AUC_ES : rho={rho_ce:.3f}")
log.info("")
log.info(f"  {'Cut':>5} | {'ES: RF∩AUC':>12} {'%':>5} | {'IT: RF∩AUC':>12} {'%':>5} | "
         f"{'Bio in RF-ES':>13} {'Bio in RF-IT':>13}")
log.info("  " + "-" * 80)

for cutoff in [5, 9, 14, 20, 30, 50]:
    top_rf_es_n  = set(master.sort_values("rf_rank_es").head(cutoff)["feature"])
    top_rf_it_n  = set(master.sort_values("rf_rank_it").head(cutoff)["feature"])
    top_auc_es_n = set(auc_es_df.head(cutoff)["feature"])
    top_auc_it_n = set(auc_it_df.head(cutoff)["feature"])

    ov_es  = len(top_rf_es_n & top_auc_es_n)
    ov_it  = len(top_rf_it_n & top_auc_it_n)
    bio_es = len(top_rf_es_n & BIOMARKERS)
    bio_it = len(top_rf_it_n & BIOMARKERS)

    log.info(f"  {cutoff:>5} | {ov_es:>5}/{cutoff:<5} {ov_es/cutoff*100:>4.0f}% | "
             f"{ov_it:>5}/{cutoff:<5} {ov_it/cutoff*100:>4.0f}% | "
             f"{bio_es:>5}/{min(cutoff,11):<7} {bio_it:>5}/{min(cutoff,11):<7}")


# ============================================================
# BOTTOM FEATURES — CNN 11x10 matrix candidates
# ============================================================
section("8. Bottom Features — CNN Matrix Recommendation")

bottom20 = master.sort_values("mean_rf_rank", ascending=False).head(20)
log.info("  20 lowest-importance features (mean rank across ES + IT + Combined):")
log.info(f"  {'Rank':>5} {'Feature':<32} {'RF-ES':>6} {'RF-IT':>6} {'RF-Comb':>8} {'Mean':>7}")
log.info("  " + "-" * 64)
for _, r in bottom20.iterrows():
    log.info(f"  {int(r['overall_rf_rank']):>5} {r['feature']:<32} "
             f"{int(r['rf_rank_es']):>6} {int(r['rf_rank_it']):>6} "
             f"{int(r['rf_rank_comb']):>8} {r['mean_rf_rank']:>7.1f}")

bottom2 = master.tail(2)
log.info("")
log.info("  Recommended removal (bottom 2 consistently across all datasets):")
for _, r in bottom2.iterrows():
    log.info(f"    Remove: {r['feature']:<32}  "
             f"RF-ES={int(r['rf_rank_es'])}  "
             f"RF-IT={int(r['rf_rank_it'])}  "
             f"RF-Comb={int(r['rf_rank_comb'])}")
log.info(f"  -> Remaining {len(feat_cols)-2} features -> reshape 11x10 -> CNN input")


# ============================================================
# PLOTS
# ============================================================
section("9. Plots")

COLORS = {"es": "#1565C0", "it": "#2E7D32", "comb": "#6A1B9A"}

# -- Plot 1: Top-30 bars (3 panels) --
fig, axes = plt.subplots(1, 3, figsize=(24, 10))
fig.suptitle(
    "RF Feature Importance — Top 30 per Dataset  |  "
    "Red=AUC-merged+Biomarker  Dark=AUC-merged  Amber=Biomarker only  Grey=neither",
    fontsize=10, fontweight="bold")

for ax, (rf_df, rank_col, imp_col, label, color) in zip(axes, [
    (rf_es,   "rf_rank_es",   "rf_imp_es",   "PC-GITA (ES)",  COLORS["es"]),
    (rf_it,   "rf_rank_it",   "rf_imp_it",   "VOICED (IT)",   COLORS["it"]),
    (rf_comb, "rf_rank_comb", "rf_imp_comb", "Combined",      COLORS["comb"]),
]):
    top30 = rf_df.sort_values(rank_col).head(30).copy()
    cols = []
    for feat in top30["feature"]:
        if feat in auc_set and feat in BIOMARKERS: cols.append("#C62828")
        elif feat in auc_set:                       cols.append(color)
        elif feat in BIOMARKERS:                    cols.append("#FF8F00")
        else:                                       cols.append("#B0BEC5")

    vals = top30[imp_col].values
    lbls = top30["feature"].values
    ax.barh(range(len(top30)), vals[::-1], color=cols[::-1],
            edgecolor="white", height=0.75)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(lbls[::-1], fontsize=7.5)
    ax.set_xlabel("RF Gini Importance", fontsize=9)
    ax.set_title(f"{label}  (top-30 of {len(rf_df)})", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for i, (v, f) in enumerate(zip(vals[::-1], lbls[::-1])):
        rk = int(top30[top30["feature"] == f][rank_col].values[0])
        ax.text(v + max(vals)*0.01, i, f"#{rk}", va="center", fontsize=7, color="#455A64")

from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(facecolor="#C62828", label="AUC merged + Biomarker"),
    Patch(facecolor=COLORS["comb"], label="AUC merged only"),
    Patch(facecolor="#FF8F00", label="Biomarker not in merged"),
    Patch(facecolor="#B0BEC5", label="Neither"),
], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.07, 1, 1])
p1 = os.path.join(OUT_DIR, "plot_top30_bars.png")
plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
log.info(f"  Plot 1 -> {p1}")


# -- Plot 2: Scatter — AUC rank vs RF rank, all 112 --
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("AUC Rank vs RF Rank — All 112 Features\n"
             "Near diagonal = both methods agree  |  Size = RF importance",
             fontsize=11, fontweight="bold")

for ax, (auc_col, rf_col, imp_col, title) in zip(axes, [
    ("auc_rank_es", "rf_rank_es", "rf_imp_es", f"PC-GITA (ES)  rho={rho_es:.3f}"),
    ("auc_rank_it", "rf_rank_it", "rf_imp_it", f"VOICED (IT)   rho={rho_it:.3f}"),
]):
    sub = master.dropna(subset=[auc_col]).copy()
    c_map = {f: ("#C62828" if f in BIOMARKERS
                 else "#7B1FA2" if f.startswith("mfcc")
                 else "#0277BD" if (f.startswith("spectral") or f.startswith("log_energy") or f.startswith("zcr"))
                 else "#2E7D32" if (f.startswith("praat") or f.startswith("pyin"))
                 else "#F57C00" if f.startswith("chroma")
                 else "#90A4AE") for f in sub["feature"]}
    colors_sc = [c_map[f] for f in sub["feature"]]
    sizes     = (sub[imp_col] / sub[imp_col].max() * 120 + 20).values
    mx = max(sub[auc_col].max(), sub[rf_col].max())
    ax.scatter(sub[auc_col], sub[rf_col], c=colors_sc, s=sizes,
               alpha=0.75, edgecolors="white", linewidths=0.5)
    ax.plot([1, mx], [1, mx], "k--", lw=1, alpha=0.3)
    for _, row in sub[sub["feature"].isin(auc_set | BIOMARKERS)].iterrows():
        ax.annotate(row["feature"][:14], (row[auc_col], row[rf_col]),
                    fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("AUC-based Rank", fontsize=9); ax.set_ylabel("RF Rank", fontsize=9)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)
    ax.invert_xaxis(); ax.invert_yaxis()

fig.legend(handles=[
    Patch(facecolor="#C62828", label="Biomarker"),
    Patch(facecolor="#7B1FA2", label="MFCC"),
    Patch(facecolor="#0277BD", label="Spectral/Energy"),
    Patch(facecolor="#2E7D32", label="F0"),
    Patch(facecolor="#F57C00", label="Chroma"),
    Patch(facecolor="#90A4AE", label="Other"),
], loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout(rect=[0, 0.08, 1, 1])
p2 = os.path.join(OUT_DIR, "plot_full_scatter.png")
plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
log.info(f"  Plot 2 -> {p2}")


# -- Plot 3: Bottom-20 —-
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("Bottom 20 Features by Mean RF Rank\n"
             "Red dashed line = recommended cut (remove 2 -> 110 features -> 11x10 CNN)",
             fontsize=11, fontweight="bold")
b20 = master.tail(20).sort_values("mean_rf_rank", ascending=True)
y   = range(len(b20))
ax.barh([i-0.25 for i in y], b20["rf_imp_es"].values,   0.25, color=COLORS["es"],   label="ES",       alpha=0.85)
ax.barh([i      for i in y], b20["rf_imp_it"].values,   0.25, color=COLORS["it"],   label="IT",       alpha=0.85)
ax.barh([i+0.25 for i in y], b20["rf_imp_comb"].values, 0.25, color=COLORS["comb"], label="Combined", alpha=0.85)
ax.set_yticks(list(y)); ax.set_yticklabels(b20["feature"].values, fontsize=9)
ax.axhline(1.5, color="red", lw=2, linestyle="--", label="Recommended cut")
ax.set_xlabel("RF Gini Importance", fontsize=10)
ax.set_title(f"Remove: {b20.iloc[-2]['feature']} + {b20.iloc[-1]['feature']}", fontsize=9)
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "plot_bottom_features.png")
plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
log.info(f"  Plot 3 -> {p3}")


# ============================================================
# SAVE
# ============================================================
section("10. Save")

rf_es.rename(columns={"rf_rank_es": "rf_rank", "rf_imp_es": "rf_importance"}).to_csv(
    os.path.join(OUT_DIR, "rf_full_ranking_es.csv"), index=False)
rf_it.rename(columns={"rf_rank_it": "rf_rank", "rf_imp_it": "rf_importance"}).to_csv(
    os.path.join(OUT_DIR, "rf_full_ranking_it.csv"), index=False)
rf_comb.rename(columns={"rf_rank_comb": "rf_rank", "rf_imp_comb": "rf_importance"}).to_csv(
    os.path.join(OUT_DIR, "rf_full_ranking_combined.csv"), index=False)
master.to_csv(os.path.join(OUT_DIR, "master_comparison_all112.csv"), index=False)
master.tail(20).to_csv(os.path.join(OUT_DIR, "bottom_features.csv"), index=False)

log.info("  rf_full_ranking_es.csv")
log.info("  rf_full_ranking_it.csv")
log.info("  rf_full_ranking_combined.csv")
log.info("  master_comparison_all112.csv")
log.info("  bottom_features.csv")
log.info("")
log.info(f"  Spearman ES: rho={rho_es:.3f}  p={p_es:.4f}")
log.info(f"  Spearman IT: rho={rho_it:.3f}  p={p_it:.4f}")
log.info(f"  Remove for CNN: {list(master.tail(2)['feature'])}")
log.info(f"  Output: {OUT_DIR}")
log.info(f"  End   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
