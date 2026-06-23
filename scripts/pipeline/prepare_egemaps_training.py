"""
Prepare eGeMAPS Training Datasets
===================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE:
    Merge eGeMAPS-extracted CSVs into training-ready files for
    dataset_wise_analysis_v2.py — apples-to-apples comparison
    with our handcrafted 112-feature pipeline.

INPUTS  (features/opensmile/):
    features_egemaps_pcgita.csv         267 rows, 88 features
    features_egemaps_voiced.csv         296 rows, 88 features

OUTPUTS (features/opensmile/):
    training_egemaps_full88f.csv        563 rows, 88 features  — all eGeMAPS
    training_egemaps_biomarker_eq6f.csv 563 rows,  6 features  — jitter+shimmer+HNR only
    training_egemaps_no_mfcc_72f.csv    563 rows, 72 features  — eGeMAPS minus MFCCs

HOW TO RUN:
    python scripts\\prepare_egemaps_training.py

THEN run analysis:
    python scripts\\dataset_wise_analysis_v2.py --data_file ../features/opensmile/training_egemaps_full88f.csv
    python scripts\\dataset_wise_analysis_v2.py --data_file ../features/opensmile/training_egemaps_biomarker_eq6f.csv
    python scripts\\dataset_wise_analysis_v2.py --data_file ../features/opensmile/training_egemaps_no_mfcc_72f.csv
"""

import os
import sys
import pandas as pd

# ============================================================================
# PATHS
# ============================================================================

BASE       = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
FEATURES_DIR = os.path.join(BASE, "features")
OUTPUT_DIR   = os.path.join(BASE, "features", "opensmile")

# DATA AVAILABILITY NOTE:
#   extract_features_sustained_a_opensmile.py outputs ONE combined CSV per
#   sample rate (pc_gita + italian + voice_dataset all merged). It does NOT
#   produce per-dataset split files. Per-dataset split is done here via the
#   'dataset' column filter. This is the canonical source for eGeMAPS features.
#
#   Dataset column values confirmed from the extractor output:
#     'pc_gita'      → Spanish, ~267 rows at 8kHz
#     'voice_dataset'→ English (Kaggle), ~296 rows at 8kHz
#     'italian'      → Italian, excluded (same as eval engine's load_full_data)
COMBINED_EGEMAPS_CSV = os.path.join(FEATURES_DIR, "opensmile", "features_egemaps_8k.csv")

# Label names as they appear in the 'dataset' column of the combined CSV
PCGITA_LABEL = "pc_gita"
VOICED_LABEL = "voice_dataset"

# ============================================================================
# eGeMAPS FEATURE CATEGORY MAPPING
# 88 features total — manually verified against eGeMAPSv02 spec
# ============================================================================

META_COLS = ["dataset", "subject_id", "language", "speech_type",
             "disease_label", "label_binary", "file"]

# eGeMAPS biomarker equivalents (closest to our handcrafted biomarkers)
# jitterLocal → jitter_local
# shimmerLocaldB → shimmer_local
# HNRdBACF → HNR
BIOMARKER_EQ_PATTERNS = ["jitter", "shimmer", "HNR"]

# eGeMAPS MFCC features (MFCC1–4 only — eGeMAPS is minimalistic)
MFCC_PATTERNS = ["mfcc"]


def categorise_features(feature_cols):
    biomarker_eq = [f for f in feature_cols
                    if any(p.lower() in f.lower() for p in BIOMARKER_EQ_PATTERNS)]
    mfcc_feats   = [f for f in feature_cols
                    if any(p.lower() in f.lower() for p in MFCC_PATTERNS)]
    non_mfcc     = [f for f in feature_cols if f not in mfcc_feats]
    return biomarker_eq, mfcc_feats, non_mfcc


# ============================================================================
# MAIN
# ============================================================================

def main():
    # This function reads the combined eGeMAPS CSV (all datasets in one file,
    # separated by the 'dataset' column) and:
    #   1. Filters to PC-GITA and VOICED subsets only (drops Italian, matching
    #      the eval engine's load_full_data() behaviour)
    #   2. Verifies column alignment between the two subsets
    #   3. Merges and shuffles them to produce training_egemaps_full88f.csv
    #
    # HISTORY: The .bak version assumed per-dataset files existed
    # (features_egemaps_pcgita.csv, features_egemaps_voiced.csv). Those files
    # were never produced by extract_features_sustained_a_opensmile.py, which
    # always outputs combined CSVs. The correct approach is to filter by the
    # 'dataset' column that the extractor already populates.
    print("=" * 65)
    print("  Prepare eGeMAPS Training Datasets")
    print("=" * 65)

    # 1. Load the combined source
    if not os.path.exists(COMBINED_EGEMAPS_CSV):
        print(f"[ERROR] Not found: {COMBINED_EGEMAPS_CSV}")
        print("  → Run extract_features_sustained_a_opensmile.py first.")
        print("  → Expected output: features/features_egemaps_8k.csv")
        sys.exit(1)

    df_all = pd.read_csv(COMBINED_EGEMAPS_CSV)
    print(f"\nLoaded combined eGeMAPS CSV: {len(df_all)} rows")
    print(f"  Datasets found: {sorted(df_all['dataset'].unique().tolist())}")

    # 2. Split by dataset label
    pcgita = df_all[df_all['dataset'] == PCGITA_LABEL].copy()
    voiced = df_all[df_all['dataset'] == VOICED_LABEL].copy()

    if len(pcgita) == 0:
        print(f"[ERROR] No rows with dataset='{PCGITA_LABEL}' found.")
        print(f"  → Available values: {df_all['dataset'].unique().tolist()}")
        sys.exit(1)
    if len(voiced) == 0:
        print(f"[ERROR] No rows with dataset='{VOICED_LABEL}' found.")
        print(f"  → Available values: {df_all['dataset'].unique().tolist()}")
        sys.exit(1)

    n_italian = (df_all['dataset'] == 'italian').sum()
    if n_italian > 0:
        print(f"  [INFO] Dropped {n_italian} Italian rows (cross-lingual scope: PC-GITA + VOICED only)")

    print(f"\nSubset sizes:")
    print(f"  PC-GITA ({PCGITA_LABEL}) : {len(pcgita)} rows | PD={(pcgita['label_binary']==1).sum()} HC={(pcgita['label_binary']==0).sum()}")
    print(f"  VOICED  ({VOICED_LABEL}) : {len(voiced)} rows | PD={(voiced['label_binary']==1).sum()} HC={(voiced['label_binary']==0).sum()}")

    # 3. Verify column alignment before merge
    assert list(pcgita.columns) == list(voiced.columns), \
        f"Column mismatch between PC-GITA and VOICED subsets:\n" \
        f"  Symmetric diff: {set(pcgita.columns) ^ set(voiced.columns)}"

    feature_cols = [c for c in pcgita.columns if c not in META_COLS]
    if len(feature_cols) != 88:
        print(f"[WARNING] Expected 88 eGeMAPS features, got {len(feature_cols)} — proceeding")

    biomarker_eq, mfcc_feats, non_mfcc = categorise_features(feature_cols)

    print(f"\neGeMAPS feature breakdown:")
    print(f"  All features    : {len(feature_cols)}")
    print(f"  Biomarker equiv : {len(biomarker_eq)}  → {biomarker_eq}")
    print(f"  MFCC features   : {len(mfcc_feats)}")
    print(f"  Non-MFCC        : {len(non_mfcc)}")

    # 4. Merge and shuffle (reproducible)
    combined = pd.concat([pcgita, voiced], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nCombined: {len(combined)} rows | PD={(combined['label_binary']==1).sum()} HC={(combined['label_binary']==0).sum()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Save full 88f — this is now a GENUINE two-source merge, not a renamed copy
    out_full = os.path.join(OUTPUT_DIR, "training_egemaps_full88f.csv")
    combined[META_COLS + feature_cols].to_csv(out_full, index=False)
    print(f"\n[1] Saved: training_egemaps_full88f.csv  ({len(feature_cols)} features, {len(combined)} rows)")

    # 6. Save biomarker equivalent (6f)
    out_bio = os.path.join(OUTPUT_DIR, "training_egemaps_biomarker_eq6f.csv")
    combined[META_COLS + biomarker_eq].to_csv(out_bio, index=False)
    print(f"[2] Saved: training_egemaps_biomarker_eq6f.csv  ({len(biomarker_eq)} features)")

    # 7. Save no-MFCC (72f)
    out_nomfcc = os.path.join(OUTPUT_DIR, "training_egemaps_no_mfcc_72f.csv")
    combined[META_COLS + non_mfcc].to_csv(out_nomfcc, index=False)
    print(f"[3] Saved: training_egemaps_no_mfcc_72f.csv  ({len(non_mfcc)} features)")

    print("\n" + "=" * 65)
    print("  DONE. Now run analysis:")
    print("=" * 65)
    print("  python scripts\\dataset_wise_analysis_v2.py --data_file features/opensmile/training_egemaps_full88f.csv")
    print("  python scripts\\dataset_wise_analysis_v2.py --data_file features/opensmile/training_egemaps_biomarker_eq6f.csv")
    print("  python scripts\\dataset_wise_analysis_v2.py --data_file features/opensmile/training_egemaps_no_mfcc_72f.csv")


if __name__ == "__main__":
    main()
