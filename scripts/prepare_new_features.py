"""
Prepare Features for Modeling Pipeline
========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

INPUT  : features/features_extracted_sv.csv
         (1,614 rows, 134 columns — v4 extraction output)

OUTPUT : features/features_sv_modeling.csv
         Clean, un-normalized, model-ready CSV with 77 features.

WHAT THIS DOES:
    1. Loads features_extracted_sv.csv
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

FEATURE GROUPS IN OUTPUT (77 features):
    Praat F0    : std, min, max, range, median            (5)
    Jitter      : local, rap, ppq5                        (3)
    Shimmer     : local, apq3, apq5, apq11                (4)
    HNR/NHR     : hnr, nhr                                (2)
    MFCC mean   : 1–13                                    (13)
    MFCC std    : 1–13                                    (13)
    Log Energy  : mean, std                               (2)
    Spectral    : centroid×2, bandwidth×2, rolloff,       (7)
                  flux×2
    ZCR         : mean, std                               (2)
    Mel         : mean, std                               (2)
    Chroma      : 12 × (mean + std)                       (24)
    TOTAL                                                 (77)

HOW TO RUN:
    cd "C:\\Users\\Lenovo\\Desktop\\Code\\2026\\BE mini project"
    venv\\Scripts\\activate
    python scripts\\prepare_new_features.py
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
INPUT_CSV  = os.path.join(BASE, "features", "features_extracted_sv.csv")
OUTPUT_CSV = os.path.join(BASE, "features", "features_sv_modeling.csv")
LOG_DIR    = os.path.join(BASE, "logs")

# Meta columns — identity, not features
META_COLS = [
    "dataset", "subject_id", "language", "gender", "speech_type",
    "disease_label", "label_binary", "multiclass_label",
    "updrs_total", "moca_score", "meds_status", "file",
]

# Columns to drop — reasons documented above
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
        log.error("  Run extract_features_all.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    log.info(f"  Loaded : {len(df)} rows × {len(df.columns)} columns")
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

    # PC-GITA: 100 named speakers (AVPEPUDEAC0001 etc.)
    # VOICED:  296 files, each file = one speaker (no ID overlap)
    df["speaker_id"] = df["subject_id"].astype(str)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        spk = sub["speaker_id"].nunique()
        files_per_spk = len(sub) / spk
        log.info(f"  {ds:<15}: {spk:>4} speakers  "
                 f"{len(sub):>5} files  "
                 f"{files_per_spk:.1f} files/speaker")

    # ── 4. DROP COLUMNS ──────────────────────────────────────
    section("4. Drop Columns")

    drop_actual = [c for c in ALL_DROP if c in df.columns]
    drop_missing = [c for c in ALL_DROP if c not in df.columns]

    log.info(f"  Planned to drop : {len(ALL_DROP)} columns")
    log.info(f"  Actually present: {len(drop_actual)} columns")
    if drop_missing:
        log.info(f"  Not found (ok)  : {drop_missing}")

    log.info("")
    log.info(f"  pYIN F0 (42% NaN + octave errors)  : {len([c for c in DROP_PYIN if c in df.columns])}")
    log.info(f"  rpde/dfa/ppe (100% NaN)            : {len([c for c in DROP_NAN100 if c in df.columns])}")
    log.info(f"  Delta + Delta-Delta MFCC (AUC~0.50): {len([c for c in DROP_DELTA if c in df.columns])}")
    log.info(f"  Redundant (r>0.97 with another col): {len([c for c in DROP_REDUNDANT if c in df.columns])}")
    log.info(f"  Tracking/meta (not speech features): {len([c for c in DROP_TRACKING if c in df.columns])}")

    df = df.drop(columns=drop_actual)
    log.info(f"\n  Columns after drop: {len(df.columns)}")

    # ── 5. FEATURE COLUMNS ───────────────────────────────────
    section("5. Feature Column Audit")

    # Build list of kept meta cols
    meta_present = [c for c in META_COLS + ["label", "speaker_id"]
                    if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_present
                 and pd.api.types.is_numeric_dtype(df[c])]

    log.info(f"  Meta columns    : {len(meta_present)}")
    log.info(f"  Feature columns : {len(feat_cols)}")
    log.info("")

    # NaN check
    nan_counts = df[feat_cols].isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        log.info("  NaN in features : 0  ✓")
    else:
        log.warning(f"  NaN in {len(nan_cols)} feature columns:")
        for col, cnt in nan_cols.items():
            log.warning(f"    {col:<35}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")

    # Feature group breakdown
    groups = {
        "Praat F0"  : [c for c in feat_cols if "praat_f0" in c],
        "Jitter"    : [c for c in feat_cols if "jitter" in c],
        "Shimmer"   : [c for c in feat_cols if "shimmer" in c],
        "HNR/NHR"   : [c for c in feat_cols if c in ["hnr","nhr"]],
        "MFCC mean" : [c for c in feat_cols if "mfcc" in c and "_mean" in c and "dmfcc" not in c],
        "MFCC std"  : [c for c in feat_cols if "mfcc" in c and "_std"  in c],
        "Spectral"  : [c for c in feat_cols if "spectral" in c or "log_energy" in c
                       or "zcr" in c or "mel" in c],
        "Chroma"    : [c for c in feat_cols if "chroma" in c],
    }
    log.info("  Feature groups:")
    for grp, cols in groups.items():
        log.info(f"    {grp:<12}: {len(cols):>3} features")
    log.info(f"    {'TOTAL':<12}: {len(feat_cols):>3} features")

    # ── 6. STATISTICAL SUMMARY ───────────────────────────────
    section("6. Biomarker Sanity Check")

    log.info(f"  {'Feature':<22}  {'PD mean':>10}  {'HC mean':>10}  "
             f"{'Direction':>12}  {'p-value':>10}  {'Cohen d':>8}")
    log.info("  " + "─" * 82)

    check_feats = [
        ("jitter_local",  "PD>HC"),
        ("jitter_ppq5",   "PD>HC"),
        ("shimmer_local", "PD>HC"),
        ("shimmer_apq11", "PD>HC"),
        ("hnr",           "PD<HC"),
        ("nhr",           "PD>HC"),
        ("spectral_flux_mean", "PD>HC"),
    ]

    for feat, expected in check_feats:
        if feat not in df.columns:
            continue
        pd_vals = df[df["label"]==1][feat].dropna()
        hc_vals = df[df["label"]==0][feat].dropna()
        pd_m    = pd_vals.mean()
        hc_m    = hc_vals.mean()

        # Mann-Whitney U
        _, p = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")

        # Cohen's d
        pooled_std = np.sqrt(((len(pd_vals)-1)*pd_vals.std()**2 +
                              (len(hc_vals)-1)*hc_vals.std()**2) /
                             (len(pd_vals)+len(hc_vals)-2))
        d = abs(pd_m - hc_m) / (pooled_std + 1e-10)

        if expected == "PD>HC":
            ok = "✓" if pd_m > hc_m else "✗ INVERTED"
        else:
            ok = "✓" if pd_m < hc_m else "✗ INVERTED"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        log.info(f"  {feat:<22}  {pd_m:>10.4f}  {hc_m:>10.4f}  "
                 f"{ok:>12}  {p:>8.2e}{sig}  {d:>7.3f}")

    # ── 7. DATASET BREAKDOWN ─────────────────────────────────
    section("7. Dataset Breakdown")

    log.info(f"  {'Dataset':<15}  {'Rows':>6}  {'PD':>5}  {'HC':>5}  "
             f"{'Speakers':>9}  {'Lang':>5}")
    log.info("  " + "─" * 55)
    for ds in df["dataset"].unique():
        sub  = df[df["dataset"] == ds]
        spk  = sub["speaker_id"].nunique()
        lang = sub["language"].iloc[0]
        log.info(f"  {ds:<15}  {len(sub):>6}  "
                 f"{(sub['label']==1).sum():>5}  "
                 f"{(sub['label']==0).sum():>5}  "
                 f"{spk:>9}  {lang:>5}")
    log.info("  " + "─" * 55)
    log.info(f"  {'TOTAL':<15}  {len(df):>6}  "
             f"{(df['label']==1).sum():>5}  "
             f"{(df['label']==0).sum():>5}")

    # ── 8. SAVE ───────────────────────────────────────────────
    section("8. Save")

    # Final column order: meta first, features after
    final_cols = meta_present + feat_cols
    df_out = df[final_cols]

    df_out.to_csv(OUTPUT_CSV, index=False)

    elapsed = (datetime.now() - start).total_seconds()
    log.info(f"  Saved  : {OUTPUT_CSV}")
    log.info(f"  Rows   : {len(df_out)}")
    log.info(f"  Columns: {len(df_out.columns)}  "
             f"({len(meta_present)} meta + {len(feat_cols)} features)")
    log.info(f"  Time   : {elapsed:.1f}s")
    log.info(f"  Log    : {LOG_FILE}")

    section("Done — Next Steps")
    log.info("  1. python scripts\\baseline_classifier.py")
    log.info("     → 8 models, 5-fold GroupKFold, bootstrap CIs")
    log.info("  2. python scripts\\dataset_matrix.py")
    log.info("     → 3×3 cross-dataset matrix, cascade pipeline")
    log.info("")
    log.info("  NORMALIZATION REMINDER:")
    log.info("  Features are raw (un-normalized) by design.")
    log.info("  Per supervisor: fit scaler on training set → apply to test set.")
    log.info("  Both downstream scripts use Pipeline([StandardScaler, model]).")


if __name__ == "__main__":
    main()
