"""
Class-wise PDF Analysis — Feature Discrimination Strength
===========================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE (mentor instruction, March 19, 2026):
    Plot class-wise probability density functions (PDF) for each feature,
    showing PD vs HC distribution separately for:
        1. Each individual dataset (PC-GITA Spanish, VOICED Italian)
        2. The combined dataset
    This reveals feature discrimination strength and dataset readiness
    to be combined.

CHANGES FROM PREVIOUS VERSION:
    CSV_PATH        : now points to features_extracted_sv.csv (v4 output)
    F0_FEATS        : was f0_mean/std — now praat_f0_mean/std (correct column names)
    BIOMARKER_FEATS : removed rpde/dfa/ppe (100% NaN, not computable from audio)
                      added shimmer_dda (was missing)
    pYIN columns    : excluded (38% NaN, unreliable — praat CC used instead)
    Datasets        : only pc_gita + voiced (UNINA dropped, MDVR-KCL dropped)
    UNINA logic     : all red-background / "inverted biomarkers" markers removed
    New plot group  : Delta MFCCs (dmfcc_*) and Delta-Delta MFCCs (d2mfcc_*)
    New plot group  : MFCC std features (mfcc_*_std)
    Spectral        : added _std variants (bandwidth_std, flux_std, etc.)
    Compatibility   : updated — only checks pc_gita and voiced now

HOW TO INTERPRET PLOTS:
    Separated peaks  → feature discriminates PD from HC well
    Overlapping peaks → feature has weak discrimination
    AUC near 0.5     → feature is not useful
    AUC near 1.0     → feature is highly discriminative
    Same direction across datasets → safe to combine

OUTPUT:
    results/pdf_analysis_<TIMESTAMP>/
        01_biomarkers_per_dataset.png     — jitter, shimmer, HNR, NHR
        02_f0_per_dataset.png             — Praat CC F0 features
        03_mfcc_mean_per_dataset.png      — MFCC mean coefficients 1-13
        04_mfcc_std_per_dataset.png       — MFCC std coefficients 1-13
        05_delta_mfcc_per_dataset.png     — Delta MFCC (velocity)
        06_d2mfcc_per_dataset.png         — Delta-Delta MFCC (acceleration)
        07_spectral_per_dataset.png       — Spectral + ZCR + Mel + Energy
        08_discrimination_summary.png     — AUC ranking bar + heatmap
        pdf_discrimination_scores.csv     — all AUC scores numerically
        run_log.txt

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\pdf_analysis.py
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from datetime import datetime

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

# Primary CSV — v4 extraction output
CSV_PATH = os.path.join(BASE, "features", "features_extracted_sv.csv")

# Fallbacks (in order) if primary not found yet
FALLBACK_CSVS = [
    os.path.join(BASE, "features", "features_extracted_sv_pc_gita.csv"),
]

BASE_OUTDIR = os.path.join(BASE, "results")

# ── Visual constants ──
PD_COLOR = "#DC2626"   # red
HC_COLOR = "#2563EB"   # blue
ALPHA    = 0.35

# Only the two active datasets
DS_COLORS = {
    "pc_gita": "#f59e0b",
    "voiced":  "#8b5cf6",
}
DS_LABELS = {
    "pc_gita": "PC-GITA (ES)",
    "voiced":  "VOICED (IT)",
}

# ── Feature groups — aligned to actual column names in features_extracted_sv.csv ──

# Praat CC F0 — correct names (not f0_mean — that was old pipeline)
F0_FEATS = [
    "praat_f0_mean", "praat_f0_std", "praat_f0_min",
    "praat_f0_max",  "praat_f0_range", "praat_f0_median",
]

# Biomarkers — rpde/dfa/ppe EXCLUDED (100% NaN, not computable from raw audio)
BIOMARKER_FEATS = [
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
    "hnr", "nhr",
]

# MFCC mean coefficients 1–13
MFCC_MEAN_FEATS = [f"mfcc_{i:02d}_mean" for i in range(1, 14)]

# MFCC std coefficients 1–13 — captures within-utterance variability
MFCC_STD_FEATS  = [f"mfcc_{i:02d}_std"  for i in range(1, 14)]

# Delta MFCCs — first derivative (velocity) of MFCC trajectory
DELTA_MFCC_FEATS = [f"dmfcc_{i:02d}_mean" for i in range(1, 14)]

# Delta-Delta MFCCs — second derivative (acceleration)
D2MFCC_FEATS = [f"d2mfcc_{i:02d}_mean" for i in range(1, 14)]

# Spectral + energy + ZCR + Mel — all _mean and _std variants
SPECTRAL_FEATS = [
    "spectral_centroid_mean",   "spectral_centroid_std",
    "spectral_bandwidth_mean",  "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_flux_mean",       "spectral_flux_std",
    "zcr_mean",                 "zcr_std",
    "log_energy_mean",          "log_energy_std",
    "mel_mean",                 "mel_std",
]


# ══════════════════════════════════════════════════════════════
# OUTPUT SETUP
# ══════════════════════════════════════════════════════════════
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = os.path.join(BASE_OUTDIR, f"pdf_analysis_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_PATH  = os.path.join(RUN_DIR, "run_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("pdf_analysis")

def log(m=""): logger.info(m)
def section(t):
    log(); log("=" * 65)
    log(f"  {t}")
    log("=" * 65)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def plot_pdf(ax, values_pd, values_hc, feat_name,
             title=None, show_legend=True, fontsize=8):
    """
    KDE smooth PDF for PD (red) and HC (blue) on one axis.
    Shows filled area, mean dashed line, and AUC in title.
    """
    values_pd = values_pd.dropna()
    values_hc = values_hc.dropna()

    if len(values_pd) < 5 or len(values_hc) < 5:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=fontsize, color="#94a3b8")
        ax.set_title(title or feat_name, fontsize=fontsize,
                     fontweight="bold", pad=4)
        return

    for vals, color, label in [
        (values_pd, PD_COLOR, "PD"),
        (values_hc, HC_COLOR, "HC"),
    ]:
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            x   = np.linspace(vals.min(), vals.max(), 200)
            y   = kde(x)
            ax.plot(x, y, color=color, lw=1.8, label=label)
            ax.fill_between(x, y, alpha=ALPHA, color=color)
            ax.axvline(vals.mean(), color=color, lw=1,
                       linestyle="--", alpha=0.8)
        except Exception:
            pass

    # AUC for discrimination measure
    try:
        y_true  = [1] * len(values_pd) + [0] * len(values_hc)
        y_score = list(values_pd) + list(values_hc)
        auc     = roc_auc_score(y_true, y_score)
        auc     = max(auc, 1 - auc)
        ax.set_title(f"{title or feat_name}\nAUC={auc:.3f}",
                     fontsize=fontsize, fontweight="bold", pad=4)
    except Exception:
        ax.set_title(title or feat_name, fontsize=fontsize,
                     fontweight="bold", pad=4)

    ax.set_xlabel("Value", fontsize=fontsize - 1)
    ax.set_ylabel("Density", fontsize=fontsize - 1)
    ax.tick_params(labelsize=fontsize - 2)
    ax.set_facecolor("#f8fafc")
    if show_legend:
        ax.legend(fontsize=fontsize - 1, framealpha=0.9)


def compute_auc(df, feat):
    """AUC between PD and HC for one feature. Always returns >= 0.5."""
    sub = df[[feat, "label"]].dropna()
    if len(sub) < 10 or sub["label"].nunique() < 2:
        return float("nan")
    try:
        auc = roc_auc_score(sub["label"], sub[feat])
        return max(auc, 1 - auc)
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
section("1. Load Data")

if not os.path.exists(CSV_PATH):
    for p in FALLBACK_CSVS:
        if os.path.exists(p):
            CSV_PATH = p
            log(f"  Primary CSV not found — using fallback: {CSV_PATH}")
            break
    else:
        log("[ERROR] No CSV found. Run extract_features_all.py first.")
        sys.exit(1)

log(f"  Loading : {CSV_PATH}")
df_all = pd.read_csv(CSV_PATH, low_memory=False)
log(f"  Shape   : {df_all.shape}")
log(f"  Datasets: {list(df_all['dataset'].unique())}")

# Normalise label column
if "label" in df_all.columns:
    df_all["label"] = df_all["label"].astype(int)
else:
    df_all["label"] = df_all["label_binary"].astype(int)

# Confirm only sustained vowel (all rows should be SV in this CSV)
if "speech_type" in df_all.columns:
    non_sv = df_all[df_all["speech_type"] != "sustained_vowel"]
    if len(non_sv) > 0:
        log(f"  WARNING: {len(non_sv)} non-sustained-vowel rows found — filtering out")
    df_sv = df_all[df_all["speech_type"] == "sustained_vowel"].copy()
else:
    df_sv = df_all.copy()

datasets = [d for d in df_sv["dataset"].unique().tolist()
            if d in DS_LABELS]  # only known/active datasets

pd_data  = df_sv[df_sv["label"] == 1]
hc_data  = df_sv[df_sv["label"] == 0]

log(f"  Rows    : {len(df_sv)}  PD={len(pd_data)}  HC={len(hc_data)}")
log(f"  Datasets in analysis: {datasets}")

# Check which expected feature columns are actually present
log()
log("  Feature column availability:")
all_expected = (F0_FEATS + BIOMARKER_FEATS + MFCC_MEAN_FEATS +
                MFCC_STD_FEATS + DELTA_MFCC_FEATS + D2MFCC_FEATS +
                SPECTRAL_FEATS)
missing_cols = [f for f in all_expected if f not in df_sv.columns]
present_cols = [f for f in all_expected if f in df_sv.columns]
log(f"    Expected : {len(all_expected)}")
log(f"    Present  : {len(present_cols)}")
if missing_cols:
    log(f"    Missing  : {missing_cols}")


# ══════════════════════════════════════════════════════════════
# CORE PLOT FUNCTION — per dataset + combined
# ══════════════════════════════════════════════════════════════
def plot_features_per_dataset(feat_list, group_name, filename, suptitle):
    """
    Rows = features, Cols = [dataset1, dataset2, ..., COMBINED]
    Each cell = KDE PDF for PD vs HC.
    """
    feats = [f for f in feat_list if f in df_sv.columns]
    if not feats:
        log(f"  [SKIP] No columns from '{group_name}' found in CSV")
        return

    all_cols = datasets + ["COMBINED"]
    nrows    = len(feats)
    ncols    = len(all_cols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.4, nrows * 2.8))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.005)

    if nrows == 1: axes = axes.reshape(1, -1)
    if ncols == 1: axes = axes.reshape(-1, 1)

    # Column header — dataset name + class counts
    for j, col in enumerate(all_cols):
        if col == "COMBINED":
            n_pd = len(pd_data)
            n_hc = len(hc_data)
            header = f"COMBINED\nPD={n_pd}  HC={n_hc}"
        else:
            sub  = df_sv[df_sv["dataset"] == col]
            n_pd = (sub["label"] == 1).sum()
            n_hc = (sub["label"] == 0).sum()
            header = f"{DS_LABELS.get(col, col)}\nPD={n_pd}  HC={n_hc}"
        axes[0, j].set_title(header, fontsize=8.5, fontweight="bold",
                              color="#1e293b", pad=6)

    for i, feat in enumerate(feats):
        for j, col in enumerate(all_cols):
            ax = axes[i, j]
            if col == "COMBINED":
                val_pd = pd_data[feat]
                val_hc = hc_data[feat]
            else:
                sub    = df_sv[df_sv["dataset"] == col]
                val_pd = sub[sub["label"] == 1][feat]
                val_hc = sub[sub["label"] == 0][feat]

            # Feature label only in leftmost column
            feat_title = feat if j == 0 else ""
            show_leg   = (i == 0 and j == 0)
            plot_pdf(ax, val_pd, val_hc, feat,
                     title=feat_title, show_legend=show_leg, fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, filename)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#f8fafc")
    plt.close()
    log(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════
# 2. PLOT 01 — BIOMARKERS
# ══════════════════════════════════════════════════════════════
section("2. Biomarker Features (jitter / shimmer / HNR / NHR)")

plot_features_per_dataset(
    feat_list  = BIOMARKER_FEATS,
    group_name = "biomarkers",
    filename   = "01_biomarkers_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — Biomarker Features\n"
        "PD (red) vs HC (blue)  |  dashed = mean  |  AUC in title\n"
        "Expected: jitter/shimmer PD > HC  |  HNR: PD < HC"
    ),
)


# ══════════════════════════════════════════════════════════════
# 3. PLOT 02 — F0 (Praat CC)
# ══════════════════════════════════════════════════════════════
section("3. F0 Features — Praat Cross-Correlation")

plot_features_per_dataset(
    feat_list  = F0_FEATS,
    group_name = "f0",
    filename   = "02_f0_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — F0 / Pitch Features (Praat CC)\n"
        "PD (red) vs HC (blue)  |  AUC in title\n"
        "Note: pYIN excluded (38% NaN rate, octave errors on pathological voices)"
    ),
)


# ══════════════════════════════════════════════════════════════
# 4. PLOT 03 — MFCC MEAN
# ══════════════════════════════════════════════════════════════
section("4. MFCC Mean Features (coefficients 1–13)")

plot_features_per_dataset(
    feat_list  = MFCC_MEAN_FEATS,
    group_name = "mfcc_mean",
    filename   = "03_mfcc_mean_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — MFCC Mean (n_mfcc=13, n_fft=256, hop=128 at 8kHz)\n"
        "PD (red) vs HC (blue)  |  AUC in title"
    ),
)


# ══════════════════════════════════════════════════════════════
# 5. PLOT 04 — MFCC STD
# ══════════════════════════════════════════════════════════════
section("5. MFCC Std Features — within-utterance variability")

plot_features_per_dataset(
    feat_list  = MFCC_STD_FEATS,
    group_name = "mfcc_std",
    filename   = "04_mfcc_std_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — MFCC Std (within-utterance variability)\n"
        "PD (red) vs HC (blue)  |  AUC in title\n"
        "Higher std = more variable pitch/spectrum within the vowel"
    ),
)


# ══════════════════════════════════════════════════════════════
# 6. PLOT 05 — DELTA MFCC
# ══════════════════════════════════════════════════════════════
section("6. Delta MFCC Features — spectral velocity")

plot_features_per_dataset(
    feat_list  = DELTA_MFCC_FEATS,
    group_name = "delta_mfcc",
    filename   = "05_delta_mfcc_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — Delta MFCC (first derivative, delta_width=9)\n"
        "PD (red) vs HC (blue)  |  AUC in title\n"
        "Captures rate of spectral change — PD shows irregular dynamics"
    ),
)


# ══════════════════════════════════════════════════════════════
# 7. PLOT 06 — DELTA-DELTA MFCC
# ══════════════════════════════════════════════════════════════
section("7. Delta-Delta MFCC Features — spectral acceleration")

plot_features_per_dataset(
    feat_list  = D2MFCC_FEATS,
    group_name = "d2mfcc",
    filename   = "06_d2mfcc_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — Delta-Delta MFCC (second derivative)\n"
        "PD (red) vs HC (blue)  |  AUC in title\n"
        "Captures spectral acceleration — complements delta features"
    ),
)


# ══════════════════════════════════════════════════════════════
# 8. PLOT 07 — SPECTRAL
# ══════════════════════════════════════════════════════════════
section("8. Spectral + ZCR + Energy + Mel Features")

plot_features_per_dataset(
    feat_list  = SPECTRAL_FEATS,
    group_name = "spectral",
    filename   = "07_spectral_per_dataset.png",
    suptitle   = (
        "Class-wise PDF — Spectral, ZCR, Log Energy, Mel\n"
        "PD (red) vs HC (blue)  |  AUC in title"
    ),
)


# ══════════════════════════════════════════════════════════════
# 9. AUC DISCRIMINATION SUMMARY
# ══════════════════════════════════════════════════════════════
section("9. Discrimination Summary — AUC Ranking")

all_feats = [f for f in (
    BIOMARKER_FEATS + F0_FEATS +
    MFCC_MEAN_FEATS + MFCC_STD_FEATS +
    DELTA_MFCC_FEATS + D2MFCC_FEATS +
    SPECTRAL_FEATS
) if f in df_sv.columns]

log(f"  Computing AUC for {len(all_feats)} features × "
    f"{len(datasets)+1} columns (datasets + combined)...")

records = []
for feat in all_feats:
    row = {"feature": feat}
    for ds in datasets:
        row[ds] = compute_auc(df_sv[df_sv["dataset"] == ds], feat)
    row["combined"] = compute_auc(df_sv, feat)
    records.append(row)

scores_df = pd.DataFrame(records).sort_values("combined", ascending=False)

scores_path = os.path.join(RUN_DIR, "pdf_discrimination_scores.csv")
scores_df.to_csv(scores_path, index=False)
log(f"  Saved → {scores_path}")

# Print top 25
log()
log("  ── Top 25 Most Discriminative Features ──")
header = f"  {'Feature':<30}  {'Combined':>9}"
for ds in datasets:
    header += f"  {DS_LABELS.get(ds,ds)[:12]:>13}"
log(header)
log("  " + "─" * 75)

for _, row in scores_df.head(25).iterrows():
    line = f"  {row['feature']:<30}  {row['combined']:>9.4f}"
    for ds in datasets:
        val = row.get(ds, float("nan"))
        line += f"  {val:>13.4f}" if not np.isnan(val) else f"  {'N/A':>13}"
    log(line)


# ── AUC bar chart + per-dataset heatmap ──
fig, axes = plt.subplots(2, 1, figsize=(22, 18))
fig.patch.set_facecolor("#f8fafc")
fig.suptitle(
    "Feature Discrimination Strength — AUC-ROC (PD vs HC)\n"
    "Higher = better  |  0.50 = random  |  1.00 = perfect",
    fontsize=13, fontweight="bold",
)

# Top: combined AUC bar — top 30
top_n   = min(30, len(scores_df))
top30   = scores_df.head(top_n)
ax1     = axes[0]
ax1.set_facecolor("#f8fafc")

# Color bars by feature group
def feat_color(feat_name):
    if any(x in feat_name for x in ["jitter","shimmer","hnr","nhr"]):
        return "#ef4444"   # biomarker — red
    if "praat_f0" in feat_name:
        return "#06b6d4"   # F0 — cyan
    if "d2mfcc" in feat_name:
        return "#f59e0b"   # delta-delta — amber
    if "dmfcc" in feat_name:
        return "#10b981"   # delta — green
    if "mfcc" in feat_name:
        return "#8b5cf6"   # MFCC — purple
    return "#64748b"       # spectral/other — slate

bar_colors = [feat_color(f) for f in top30["feature"]]
bars = ax1.barh(range(top_n), top30["combined"].values[::-1],
                color=bar_colors[::-1], edgecolor="white", height=0.72)
for bar, val in zip(bars, top30["combined"].values[::-1]):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=7.5, color="#374151")

ax1.set_yticks(range(top_n))
ax1.set_yticklabels(top30["feature"].values[::-1], fontsize=8)
ax1.set_xlabel("AUC-ROC (combined)", fontsize=10)
ax1.set_title(f"Top {top_n} Features — Combined Dataset AUC",
              fontsize=10, fontweight="bold")
ax1.axvline(0.5, color="gray", lw=1.5, linestyle="--", alpha=0.5,
            label="Random baseline (0.50)")
ax1.set_xlim([0.40, 1.02])
ax1.grid(axis="x", alpha=0.2)

legend_els = [
    Patch(facecolor="#ef4444", label="Biomarker (jitter/shimmer/HNR)"),
    Patch(facecolor="#06b6d4", label="F0 / Pitch (Praat CC)"),
    Patch(facecolor="#8b5cf6", label="MFCC mean"),
    Patch(facecolor="#f59e0b", label="Delta-Delta MFCC"),
    Patch(facecolor="#10b981", label="Delta MFCC"),
    Patch(facecolor="#64748b", label="Spectral / Energy / ZCR"),
]
ax1.legend(handles=legend_els, fontsize=8.5, framealpha=0.9,
           loc="lower right")

# Bottom: per-dataset AUC heatmap — top 20
top20        = scores_df.head(20)
ds_cols      = [c for c in scores_df.columns
                if c not in ["feature", "combined"]]
heatmap_data = top20[ds_cols].values.T  # (n_datasets, 20)

ax2 = axes[1]
ax2.set_facecolor("#f8fafc")
im  = ax2.imshow(heatmap_data, aspect="auto", cmap="RdYlGn",
                 vmin=0.40, vmax=1.00)
plt.colorbar(im, ax=ax2, label="AUC-ROC", shrink=0.8)

ax2.set_xticks(range(20))
ax2.set_xticklabels(top20["feature"].values, rotation=40,
                     ha="right", fontsize=8)
ax2.set_yticks(range(len(ds_cols)))
ax2.set_yticklabels([DS_LABELS.get(d, d) for d in ds_cols], fontsize=10)
ax2.set_title(
    "Per-Dataset AUC for Top 20 Features\n"
    "Green = strong discrimination  |  Red = weak",
    fontsize=10, fontweight="bold",
)

for i in range(len(ds_cols)):
    for j in range(20):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            txt_color = "white" if val < 0.58 else "#1e293b"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5, color=txt_color, fontweight="bold")

plt.tight_layout()
out_path = os.path.join(RUN_DIR, "08_discrimination_summary.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#f8fafc")
plt.close()
log(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════
# 10. DATASET COMPATIBILITY CHECK
# ══════════════════════════════════════════════════════════════
section("10. Dataset Compatibility Check")

log("  For each biomarker + F0 feature:")
log("  ✓ = correct direction (PD > HC for jitter/shimmer, PD < HC for HNR)")
log("  ✗ = inverted (incompatible — do NOT combine without correction)")
log()

check_feats = BIOMARKER_FEATS + F0_FEATS[:3]
header_line = f"  {'Feature':<25}"
for ds in datasets:
    header_line += f"  {DS_LABELS.get(ds, ds):<16}"
header_line += f"  {'COMBINED':<12}"
log(header_line)
log("  " + "─" * 80)

all_compatible = True
for feat in check_feats:
    if feat not in df_sv.columns:
        continue
    line = f"  {feat:<25}"
    for ds in datasets + ["__combined__"]:
        sub  = df_sv if ds == "__combined__" else df_sv[df_sv["dataset"] == ds]
        pd_m = sub[sub["label"] == 1][feat].mean()
        hc_m = sub[sub["label"] == 0][feat].mean()
        if np.isnan(pd_m) or np.isnan(hc_m):
            line += f"  {'N/A':<16}"
            continue
        if feat == "hnr":
            ok = pd_m < hc_m
        else:
            ok = pd_m > hc_m
        if not ok:
            all_compatible = False
        symbol = "✓" if ok else "✗ INVERTED"
        line += f"  {symbol:<16}"
    log(line)

log()
if all_compatible:
    log("  ✓ ALL features show correct direction across both datasets.")
    log("    Datasets are COMPATIBLE to combine.")
else:
    log("  ✗ Some features are inverted in at least one dataset.")
    log("    Investigate before combining.")


# ══════════════════════════════════════════════════════════════
# 11. DONE
# ══════════════════════════════════════════════════════════════
section("Done")

log(f"  Output folder : {RUN_DIR}")
log()
log("  Files generated:")
for fname in sorted(os.listdir(RUN_DIR)):
    fpath = os.path.join(RUN_DIR, fname)
    size  = os.path.getsize(fpath) // 1024
    log(f"    {fname:<50}  {size:>5} KB")

log()
log("  How to share with mentor:")
log("    08_discrimination_summary.png  — overall AUC ranking + heatmap")
log("    01_biomarkers_per_dataset.png  — jitter/shimmer/HNR confirmation")
log("    02_f0_per_dataset.png          — pitch feature check")
log("    pdf_discrimination_scores.csv  — full numerical table")
log()
log(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
