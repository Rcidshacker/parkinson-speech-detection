"""
Prepare ComParE16 Training Dataset (8kHz)
==========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE:
    Mirrors prepare_egemaps_training.py but for ComParE16 features.
    Splits features_compare_8k.csv (which already contains both datasets merged)
    into a clean training CSV placed in features/opensmile/ alongside the
    eGeMAPS training files — keeping the pipeline consistent across feature sets.

INPUT  : features/features_compare_8k.csv
         (both pc_gita + voice_dataset, ~6373 features after VarianceThreshold)

OUTPUT : features/opensmile/training_compare_8k_full.csv
         Same rows as input, placed in opensmile/ for pipeline consistency.

HOW TO RUN:
    venv\Scripts\python.exe scripts\pipeline\prepare_compare_training.py

THEN run evaluation:
    venv\Scripts\python.exe scripts\pipeline\dataset_wise_analysis_v2.py --data_file features\opensmile\training_compare_8k_full.csv
    venv\Scripts\python.exe scripts\pipeline\dataset_wise_analysis_v2.py --data_file features\opensmile\training_compare_8k_full.csv --kbest 50
    venv\Scripts\python.exe scripts\pipeline\dataset_wise_analysis_v2.py --data_file features\opensmile\training_compare_8k_full.csv --kbest 100
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================
BASE       = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
INPUT_CSV  = os.path.join(BASE, "features", "opensmile", "features_compare_8k.csv")
OUTPUT_DIR = os.path.join(BASE, "features", "opensmile")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "training_compare_8k_full.csv")
LOG_DIR    = os.path.join(BASE, "logs")

EXPECTED_DATASETS = {"pc_gita", "voice_dataset"}

# ============================================================================
# LOGGING
# ============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"prepare_compare_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("prepare_compare")


def main():
    log.info("=" * 70)
    log.info("  Prepare ComParE16 Training CSV (8kHz)")
    log.info("=" * 70)
    log.info(f"  Input  : {INPUT_CSV}")
    log.info(f"  Output : {OUTPUT_CSV}")

    # 1. Load
    if not os.path.exists(INPUT_CSV):
        log.error(f"  [ERROR] Input not found: {INPUT_CSV}")
        log.error("  Run extract_features_sustained_a_opensmile.py first.")
        sys.exit(1)

    log.info("  Loading features_compare_8k.csv (this may take a moment — 50MB file)...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info(f"  Loaded : {len(df)} rows x {len(df.columns)} columns")

    # 2. Validate datasets
    found_datasets = set(df["dataset"].unique())
    log.info(f"  Datasets found : {sorted(found_datasets)}")
    missing = EXPECTED_DATASETS - found_datasets
    if missing:
        log.warning(f"  [WARN] Expected datasets not found: {missing}")

    # 3. Exclude italian (consistent with all other pipeline scripts)
    italian_rows = (df["dataset"] == "italian").sum()
    if italian_rows > 0:
        log.warning(f"  Excluding {italian_rows} Italian rows (cross-lingual scope: pc_gita + voice_dataset)")
        df = df[df["dataset"] != "italian"]

    # 4. Per-dataset summary
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        log.info(f"  {ds:<15}: {len(sub):>5} rows  "
                 f"PD={(sub['label_binary']==1).sum():>4}  "
                 f"HC={(sub['label_binary']==0).sum():>4}")

    log.info(f"  TOTAL: {len(df)} rows  "
             f"PD={(df['label_binary']==1).sum()}  "
             f"HC={(df['label_binary']==0).sum()}")

    # 5. Feature count
    meta_cols = {"dataset", "subject_id", "language", "speech_type",
                 "disease_label", "label_binary", "file"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    log.info(f"  Feature columns: {len(feat_cols)}")

    # 6. Save to opensmile/ folder
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"  Saved  : {OUTPUT_CSV}")
    log.info(f"  Size   : {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB")

    log.info("=" * 70)
    log.info("  DONE. Next steps:")
    log.info("  1. Run dataset_wise_analysis_v2.py --data_file features/opensmile/training_compare_8k_full.csv")
    log.info("  2. Run with --kbest 50 or --kbest 100 for K-Best variant")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
