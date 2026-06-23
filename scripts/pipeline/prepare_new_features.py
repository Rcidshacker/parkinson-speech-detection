"""
Prepare Features for Modeling Pipeline
========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

INPUT  : features/features_sustained_a.csv
         (handcrafted 112-feature extraction output)

OUTPUT : features/features_sv_modeling.csv
         Clean, un-normalized, model-ready CSV (columns that don't exist
         in the input are silently skipped during the drop step).

WHAT THIS DOES:
    1. Loads features_sustained_a.csv
    2. Drops columns confirmed useless from PDF analysis:
         pYIN F0        → 42% NaN + octave errors
         rpde/dfa/ppe   → 100% NaN (not computable from raw audio)
         delta MFCCs    → AUC ~0.50-0.55 (near-random for sustained vowel)
         delta-delta    → same
         jitter_ddp     → = jitter_rap * 3 (r=1.000, mathematically identical)
         shimmer_dda    → = shimmer_apq3 * 3 (r=1.000, mathematically identical)
         praat_f0_mean  → r=0.980 with praat_f0_median; median is more robust
         rms tracking   → preprocessing artifacts, not speech features
         duration_s     → extraction metadata, not a speech feature
         vowel_letter   → NaN for VOICED anyway
         attempt_num    → NaN for VOICED anyway
    3. Creates clean speaker_id column for GroupKFold (no speaker leakage)
    4. Validates label column, dataset counts, feature NaN state
    5. Outputs raw (un-normalized) features — normalization happens
       INSIDE each experiment pipeline per supervisor instruction:
       "Always normalize the training set and apply to the testing set"

NORMALIZATION NOTE:
    Normalization is NOT applied here by design.
    Per supervisor: fit scaler on training set → apply to test set.
    This means the scaler is fitted inside each cross-validation fold
    and each cross-dataset experiment independently.
    All downstream scripts use Pipeline([scaler, model]) to enforce this.

HOW TO RUN:
    cd "C:\\Users\\Lenovo\\Desktop\\Code\\2026\\BE mini project"
    venv\\Scripts\\activate
    python scripts\\pipeline\\prepare_new_features.py
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE       = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
INPUT_CSV  = os.path.join(BASE, "features", "handcrafted", "features_sustained_a.csv")   # FIX: was features_extracted_sv.csv
OUTPUT_CSV = os.path.join(BASE, "features", "modeling", "features_sv_modeling.csv")
LOG_DIR    = os.path.join(BASE, "logs")

# Meta columns — identity, not features
META_COLS = [
    "dataset", "subject_id", "language", "gender", "speech_type",
    "disease_label", "label_binary", "multiclass_label",
    "updrs_total", "moca_score", "meds_status", "file",
]

# Columns to drop — reasons documented above
# NOTE: Any column in this list that is absent from the input CSV is silently skipped.
DROP_PYIN     = [f"pyin_f0_{s}" for s in ["mean","std","min","max","range","median"]]
DROP_NAN100   = ["rpde", "dfa", "ppe"]
DROP_DELTA    = ([f"dmfcc_{i:02d}_mean"  for i in range(1, 14)] +
                 [f"d2mfcc_{i:02d}_mean" for i in range(1, 14)])
DROP_REDUNDANT = ["jitter_ddp",    # = jitter_rap * 3  (r=1.000)
                  "shimmer_dda",   # = shimmer_apq3 * 3 (r=1.000)
                  "praat_f0_mean"] # r=0.980 with median; median kept
DROP_TRACKING = ["rms_original", "rms_after", "rms_target", "rms_clipped",
                 "duration_s", "vowel_letter", "attempt_num"]

ALL_DROP = (DROP_PYIN + DROP_NAN100 + DROP_DELTA +
            DROP_REDUNDANT + DROP_TRACKING)


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"prepare_features_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("prepare")

def section(title):
    log.info("")
    log.info("=" * 65)
    log.info(f"  {title}")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    start = datetime.now()
    section("Prepare Features for Modeling Pipeline")
    log.info(f"  Input  : {INPUT_CSV}")
    log.info(f"  Output : {OUTPUT_CSV}")
    log.info(f"  Start  : {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. LOAD ──────────────────────────────────────────────
    section("1. Load")

    if not os.path.exists(INPUT_CSV):
        log.error(f"  [ERROR] File not found: {INPUT_CSV}")
        log.error("  Run extract_features_sustained_a.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info(f"  Loaded : {len(df)} rows x {len(df.columns)} columns")
    log.info(f"  Datasets: {list(df['dataset'].unique())}")

    # ── 2. LABEL ─────────────────────────────────────────────
    section("2. Validate Labels")

    df["label"] = pd.to_numeric(df["label_binary"], errors="coerce")
    null_labels = df["label"].isna().sum()
    if null_labels > 0:
        log.warning(f"  [WARN] {null_labels} null labels — dropping those rows")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    log.info(f"  PD (1)  : {(df['label']==1).sum()}")
    log.info(f"  HC (0)  : {(df['label']==0).sum()}")
    log.info(f"  Balance : {(df['label']==0).sum()/(df['label']==1).sum():.3f} HC/PD ratio")

    # ── 3. SPEAKER ID ────────────────────────────────────────
    section("3. Speaker ID for GroupKFold")

    df["speaker_id"] = df["subject_id"].astype(str)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        spk = sub["speaker_id"].nunique()
        files_per_spk = len(sub) / max(spk, 1)
        log.info(f"  {ds:<15}: {spk:>4} speakers  "
                 f"{len(sub):>5} files  "
                 f"{files_per_spk:.1f} files/speaker")

    # ── 4. DROP COLUMNS ──────────────────────────────────────
    section("4. Drop Columns")

    drop_actual  = [c for c in ALL_DROP if c in df.columns]
    drop_missing = [c for c in ALL_DROP if c not in df.columns]

    log.info(f"  Planned to drop : {len(ALL_DROP)} columns")
    log.info(f"  Actually present: {len(drop_actual)} columns")
    if drop_missing:
        log.info(f"  Not found (ok)  : {len(drop_missing)} columns silently skipped")

    df = df.drop(columns=drop_actual)
    log.info(f"  Columns after drop: {len(df.columns)}")

    # ── 5. FEATURE COLUMNS ───────────────────────────────────
    section("5. Feature Column Audit")

    meta_present = [c for c in META_COLS + ["label", "speaker_id"]
                    if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_present
                 and pd.api.types.is_numeric_dtype(df[c])]

    log.info(f"  Meta columns    : {len(meta_present)}")
    log.info(f"  Feature columns : {len(feat_cols)}")

    nan_counts = df[feat_cols].isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        log.info("  NaN in features : 0  OK")
    else:
        log.warning(f"  NaN in {len(nan_cols)} feature columns:")
        for col, cnt in nan_cols.items():
            log.warning(f"    {col:<35}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")

    # ── 6. STATISTICAL SANITY CHECK ──────────────────────────
    section("6. Biomarker Sanity Check")

    log.info(f"  {'Feature':<22}  {'PD mean':>10}  {'HC mean':>10}  "
             f"{'Direction':>12}  {'p-value':>10}  {'Cohen d':>8}")
    log.info("  " + "-" * 82)

    check_feats = [
        ("jitter_local",  "PD>HC"),
        ("jitter_ppq5",   "PD>HC"),
        ("shimmer_local", "PD>HC"),
        ("shimmer_apq11", "PD>HC"),
        ("hnr",           "PD<HC"),
        ("nhr",           "PD>HC"),
    ]

    for feat, expected in check_feats:
        if feat not in df.columns:
            continue
        pd_vals = df[df["label"]==1][feat].dropna()
        hc_vals = df[df["label"]==0][feat].dropna()
        pd_m = pd_vals.mean()
        hc_m = hc_vals.mean()
        _, p = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")
        pooled_std = np.sqrt(((len(pd_vals)-1)*pd_vals.std()**2 +
                              (len(hc_vals)-1)*hc_vals.std()**2) /
                             (len(pd_vals)+len(hc_vals)-2))
        d = abs(pd_m - hc_m) / (pooled_std + 1e-10)
        ok  = "OK" if ((expected == "PD>HC" and pd_m > hc_m) or
                       (expected == "PD<HC" and pd_m < hc_m)) else "INVERTED"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log.info(f"  {feat:<22}  {pd_m:>10.4f}  {hc_m:>10.4f}  "
                 f"{ok:>12}  {p:>8.2e}{sig}  {d:>7.3f}")

    # ── 7. DATASET BREAKDOWN ─────────────────────────────────
    section("7. Dataset Breakdown")

    for ds in df["dataset"].unique():
        sub  = df[df["dataset"] == ds]
        spk  = sub["speaker_id"].nunique()
        lang = sub["language"].iloc[0] if "language" in sub.columns else "?"
        log.info(f"  {ds:<15}: {len(sub):>5} rows  "
                 f"PD={(sub['label']==1).sum():>4}  HC={(sub['label']==0).sum():>4}  "
                 f"Speakers={spk:>4}  Lang={lang}")

    # ── 8. SAVE ───────────────────────────────────────────────
    section("8. Save")

    final_cols = meta_present + feat_cols
    df_out = df[final_cols]
    df_out.to_csv(OUTPUT_CSV, index=False)

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"  Saved  : {OUTPUT_CSV}")
    log.info(f"  Rows   : {len(df_out)}")
    log.info(f"  Columns: {len(df_out.columns)}  ({len(meta_present)} meta + {len(feat_cols)} features)")
    log.info(f"  Time   : {elapsed:.1f}s")

    section("Done")
    log.info("  Next: python scripts\\pipeline\\dataset_matrix.py")
    log.info("        -> 3x3 cross-dataset generalization matrix")


if __name__ == "__main__":
    main()
