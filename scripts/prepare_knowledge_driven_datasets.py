"""
Knowledge-Driven Training Dataset Preparation
==============================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE:
    Generate 6 training CSVs derived from analytical findings rather than
    arbitrary AUC cutoffs. Each set is motivated by a specific result from
    RF importance analysis and feature type-wise analysis.

    Previous sets (top9/20/30/112) were built before we knew which features
    are cross-lingually invariant. These sets USE that knowledge.

FEATURE SETS:
    1. biomarkers_11f        -- 11 phonatory biomarkers (jitter, shimmer, HNR, NHR)
                                Motivation: Feature type analysis -> 0.729 symmetric
                                cross-lingual AUC, best of all categories

    2. biomarker_f0_23f      -- 11 biomarkers + 12 F0 features (Praat + pYIN)
                                Motivation: Feature type showed Phonatory+F0=0.690,
                                slightly less than biomarkers alone. Verify on full
                                unbalanced dataset.

    3. no_mfcc_60f           -- 112 features minus all 52 MFCC-family features
                                (mfcc_ + dmfcc_ + d2mfcc_)
                                Motivation: MFCC cross-lingual AUC = 0.412 (below
                                random). Direct test: does removing the poison improve
                                the full set?

    4. rf_top14_es           -- Top-14 features from RF trained on PC-GITA (ES) only
                                Motivation: Use RF-ranked list (not AUC-ranked) as
                                feature selector for ES->IT direction

    5. rf_top14_it           -- Top-14 features from RF trained on VOICED (IT) only
                                Motivation: Use RF-ranked list as feature selector
                                for IT->ES direction

    6. cnn_ready_110f        -- All 112 features minus bottom-2 RF features
                                (dmfcc_13_mean + dmfcc_07_mean)
                                Motivation: CNN 11x10 matrix prep. Removes only
                                the two features confirmed useless by RF across
                                all three training conditions.

INPUT:
    features/training_top112_113features.csv        (source of truth, 1614 rows)
    results/final/rf_importance_comparison_*/       (RF ranking files)

OUTPUT:
    features/training_biomarkers_11f.csv
    features/training_biomarker_f0_23f.csv
    features/training_no_mfcc_60f.csv
    features/training_rf_top14_es.csv
    features/training_rf_top14_it.csv
    features/training_cnn_ready_110f.csv

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\prepare_knowledge_driven_datasets.py
"""

import os
import sys
import glob
import logging
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
SOURCE_CSV  = os.path.join(BASE, "features", "training_top112_113features.csv")
FEATURES_DIR= os.path.join(BASE, "features")
RESULTS_DIR = os.path.join(BASE, "results", "final")
LOG_DIR     = os.path.join(BASE, "logs")

META_COLS   = ["dataset", "subject_id", "language", "label_binary"]

BIOMARKERS  = [
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
    "hnr", "nhr"
]

F0_FEATURES = [
    "praat_f0_mean", "praat_f0_std", "praat_f0_min", "praat_f0_max",
    "praat_f0_median", "praat_f0_range",
    "pyin_f0_mean", "pyin_f0_std", "pyin_f0_min", "pyin_f0_max",
    "pyin_f0_median", "pyin_f0_range"
]

CNN_REMOVE  = ["dmfcc_13_mean", "dmfcc_07_mean"]

os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"prepare_knowledge_driven_{TIMESTAMP}.log"),
            mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("kd_prep")

def section(t):
    log.info(""); log.info("=" * 65)
    log.info(f"  {t}"); log.info("=" * 65)

def save_csv(df, meta, features, label, out_path):
    """Filter df to meta + features, verify, save, log."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        log.warning(f"  [{label}] {len(missing)} features missing from source: {missing}")
        features = [f for f in features if f in df.columns]

    out = df[meta + features].copy()
    out.to_csv(out_path, index=False)

    pd  = (out["label_binary"] == 1).sum()
    hc  = (out["label_binary"] == 0).sum()
    log.info(f"  [{label}]")
    log.info(f"    Features : {len(features)}")
    log.info(f"    Rows     : {len(out)}  PD={pd}  HC={hc}")
    log.info(f"    Columns  : {len(out.columns)}  ({len(meta)} meta + {len(features)} features)")
    log.info(f"    Saved -> {os.path.basename(out_path)}")
    return len(features)


# ============================================================
# LOAD SOURCE
# ============================================================
section("1. Load Source Data")
df = pd.read_csv(SOURCE_CSV)
all_feat_cols = [c for c in df.columns if c not in set(META_COLS)]
log.info(f"  Source : {SOURCE_CSV}")
log.info(f"  Rows   : {len(df)}  Features: {len(all_feat_cols)}")
log.info(f"  PD={df['label_binary'].sum()}  HC={(df['label_binary']==0).sum()}")
log.info(f"  Datasets: {list(df['dataset'].unique())}")


# ============================================================
# LOAD RF RANKINGS
# ============================================================
section("2. Load RF Rankings")
rf_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "rf_importance_comparison_*")))
if not rf_dirs:
    log.error("  No RF importance comparison folder found.")
    log.error("  Run scripts/rf_importance_comparison.py first.")
    sys.exit(1)

rf_dir = rf_dirs[-1]
log.info(f"  Using: {os.path.basename(rf_dir)}")

rf_es_top14 = pd.read_csv(os.path.join(rf_dir, "rf_full_ranking_es.csv"))["feature"].head(14).tolist()
rf_it_top14 = pd.read_csv(os.path.join(rf_dir, "rf_full_ranking_it.csv"))["feature"].head(14).tolist()

log.info(f"  RF ES top-14: {rf_es_top14}")
log.info(f"  RF IT top-14: {rf_it_top14}")


# ============================================================
# DEFINE ALL FEATURE SETS
# ============================================================
section("3. Feature Set Definitions")

mfcc_family = [f for f in all_feat_cols if any(
    f.startswith(p) for p in ["mfcc_", "dmfcc_", "d2mfcc_"])]
no_mfcc     = [f for f in all_feat_cols if f not in mfcc_family]
cnn_ready   = [f for f in all_feat_cols if f not in CNN_REMOVE]
bio_f0      = list(dict.fromkeys(BIOMARKERS + F0_FEATURES))  # preserve order, no dups

log.info(f"  Biomarkers (11)         : {BIOMARKERS}")
log.info(f"  F0 features (12)        : {F0_FEATURES}")
log.info(f"  Biomarker+F0 (23)       : {len(bio_f0)} features")
log.info(f"  MFCC-family removed (52): {len(mfcc_family)} features")
log.info(f"  No-MFCC remaining (60)  : {len(no_mfcc)} features")
log.info(f"  CNN-remove (2)          : {CNN_REMOVE}")
log.info(f"  CNN-ready (110)         : {len(cnn_ready)} features")

FEATURE_SETS = [
    ("biomarkers_11f",    BIOMARKERS,   "training_biomarkers_11f.csv"),
    ("biomarker_f0_23f",  bio_f0,       "training_biomarker_f0_23f.csv"),
    ("no_mfcc_60f",       no_mfcc,      "training_no_mfcc_60f.csv"),
    ("rf_top14_es",       rf_es_top14,  "training_rf_top14_es.csv"),
    ("rf_top14_it",       rf_it_top14,  "training_rf_top14_it.csv"),
    ("cnn_ready_110f",    cnn_ready,    "training_cnn_ready_110f.csv"),
]


# ============================================================
# GENERATE CSVs
# ============================================================
section("4. Generate Training CSVs")
summary = []
for label, features, filename in FEATURE_SETS:
    out_path = os.path.join(FEATURES_DIR, filename)
    n = save_csv(df, META_COLS, features, label, out_path)
    summary.append((label, n, filename))


# ============================================================
# SUMMARY
# ============================================================
section("5. Summary")
log.info(f"  {'Label':<22} {'Features':>9}  File")
log.info("  " + "-" * 60)
for label, n, fname in summary:
    log.info(f"  {label:<22} {n:>9}  {fname}")

log.info("")
log.info("  Motivation recap:")
log.info("    biomarkers_11f    <- Feature type AUC 0.729 symmetric cross-lingual")
log.info("    biomarker_f0_23f  <- Test whether F0 adds or hurts cross-lingual")
log.info("    no_mfcc_60f       <- MFCC AUC 0.412 (below random) -- remove poison")
log.info("    rf_top14_es       <- RF-ranked (not AUC-ranked) for ES->IT direction")
log.info("    rf_top14_it       <- RF-ranked for IT->ES direction")
log.info("    cnn_ready_110f    <- 112 minus bottom-2 -> reshape 11x10 for CNN")
log.info("")
log.info("  Next: run dataset_wise_analysis_v2.py on each via --data_file arg")
log.info(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
