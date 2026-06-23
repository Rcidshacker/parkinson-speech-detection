"""
Prepare Training Datasets from Consensus Lists
================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Generated for User

PURPOSE:
    Reads the dynamic N-way consensus feature lists and filters the raw
    extraction CSVs to create lightweight, ML-ready training datasets.

    For each feature set (Custom & OpenSMILE), it generates two training files:
    1. Union Set : All features that made it into the Top-N consensus.
    2. Stable Set: Only features that proved stable across >= 2 datasets.

INPUTS:
    - features/features_sustained_a.csv
    - features/features_opensmile_egemaps.csv
    - results/final/ranking_*/merged_consensus_list.csv

OUTPUTS:
    - features/training_sustained_a_UNION.csv
    - features/training_sustained_a_STABLE.csv
    - features/training_opensmile_egemaps_UNION.csv
    - features/training_opensmile_egemaps_STABLE.csv
"""

import os
import sys
import glob
import logging
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
FEATURES_DIR = os.path.join(BASE, "features")
MODELING_DIR = os.path.join(BASE, "features", "modeling")
RESULTS_DIR = os.path.join(BASE, "results", "final")
LOG_DIR = os.path.join(BASE, "logs")

# Subfolder lookup: maps the feature_set name (parsed from consensus folder)
# to the actual subfolder where the raw CSV lives.
FEATURE_SET_DIRS = {
    "sustained_a":     os.path.join(FEATURES_DIR, "handcrafted"),
    "egemaps_8k":      os.path.join(FEATURES_DIR, "opensmile"),
    "compare_8k":      os.path.join(FEATURES_DIR, "opensmile"),
}

# Metadata columns that MUST be preserved for the ML script
META_COLS = [
    "dataset", "subject_id", "language", "speech_type",
    "disease_label", "label_binary", "file"
]

# ============================================================================
# LOGGING SETUP
# ============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"prepare_training_datasets_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("prepare_training")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_latest_consensus_files():
    """Finds the most recent merged_consensus_list.csv for each feature set."""
    search_pattern = os.path.join(RESULTS_DIR, "ranking_*", "merged_consensus_list.csv")
    all_lists = glob.glob(search_pattern)
    
    if not all_lists:
        logger.error(f"No consensus lists found in {RESULTS_DIR}")
        sys.exit(1)
        
    # Group by feature set (e.g., 'sustained_a', 'opensmile_egemaps')
    latest_files = {}
    for filepath in all_lists:
        # Extract folder name, e.g., 'ranking_sustained_a_20260326_001139'
        folder_name = os.path.basename(os.path.dirname(filepath))
        
        # Parse the feature set name from the folder
        parts = folder_name.split("_")
        feature_set = "_".join(parts[1:-2]) # extracts 'sustained_a' or 'opensmile_egemaps'
        
        # If we haven't seen this set, or if this file is newer (based on string sorting of timestamp)
        if feature_set not in latest_files or filepath > latest_files[feature_set]:
            latest_files[feature_set] = filepath
            
    return latest_files

def generate_training_csv(raw_df, features_to_keep, output_path, description):
    """Filters the raw dataframe and saves the training CSV."""
    # Ensure we only try to keep features that actually exist in the raw dataframe
    available_features = [f for f in features_to_keep if f in raw_df.columns]
    missing_features = [f for f in features_to_keep if f not in raw_df.columns]
    
    if missing_features:
        logger.warning(f"  Missing {len(missing_features)} expected features in raw data. (e.g., {missing_features[:3]})")
        
    keep_cols = [c for c in META_COLS if c in raw_df.columns] + available_features
    
    out_df = raw_df[keep_cols].copy()
    out_df.to_csv(output_path, index=False)
    
    logger.info(f"  [{description}] Saved to: {os.path.basename(output_path)}")
    logger.info(f"    -> Rows: {len(out_df)} | Features: {len(available_features)}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 80)
    logger.info("  PREPARE FINAL TRAINING DATASETS")
    logger.info("=" * 80)
    
    # 1. Find the latest consensus files
    logger.info("\n[STEP 1] Locating latest consensus feature lists...")
    consensus_files = get_latest_consensus_files()
    
    for fset, path in consensus_files.items():
        logger.info(f"  Found '{fset}' -> {os.path.basename(os.path.dirname(path))}")

    # 2. Process each feature set
    logger.info("\n[STEP 2] Generating ML-ready datasets...")
    
    for feature_set, consensus_path in consensus_files.items():
        logger.info("-" * 60)
        logger.info(f"  Processing Feature Set: {feature_set.upper()}")
        
        # Locate corresponding raw CSV — check the mapped subfolder first, then root as fallback
        subfolder = FEATURE_SET_DIRS.get(feature_set, FEATURES_DIR)
        raw_csv_path = os.path.join(subfolder, f"features_{feature_set}.csv")
        if not os.path.exists(raw_csv_path):
            raw_csv_path = os.path.join(FEATURES_DIR, f"features_{feature_set}.csv")
        if not os.path.exists(raw_csv_path):
            logger.error(f"  Raw CSV not found! Expected: {raw_csv_path}")
            continue
            
        # Load data
        raw_df = pd.read_csv(raw_csv_path, low_memory=False)
        consensus_df = pd.read_csv(consensus_path)
        
        logger.info(f"  Loaded Raw Data  : {len(raw_df)} rows")
        logger.info(f"  Loaded Consensus : {len(consensus_df)} features ranked")
        
        # Scenario A: The "Union" Set (All features in the consensus list)
        union_features = consensus_df["feature"].tolist()
        os.makedirs(MODELING_DIR, exist_ok=True)
        union_out_path = os.path.join(MODELING_DIR, f"training_{feature_set}_UNION.csv")
        generate_training_csv(raw_df, union_features, union_out_path, "UNION SET")
        
        # Scenario B: The "Stable" Set (Features present in >= 2 datasets)
        stable_features = consensus_df[consensus_df["dataset_count"] >= 2]["feature"].tolist()
        if stable_features:
            stable_out_path = os.path.join(MODELING_DIR, f"training_{feature_set}_STABLE.csv")
            generate_training_csv(raw_df, stable_features, stable_out_path, "STABLE SET (>=2 datasets)")
        else:
            logger.warning("  [STABLE SET] Skipped: No features met the >= 2 dataset threshold.")

    logger.info("\n" + "=" * 80)
    logger.info("  DATASET PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info("  Next Step: Feed these new CSVs directly into dataset_wise_analysis_v2.py")
    logger.info("  Example: python scripts\\dataset_wise_analysis_v2.py --data_file ../features/training_sustained_a_STABLE.csv")

if __name__ == "__main__":
    main()