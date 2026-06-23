"""
Sustained Vowel "a" Feature Extraction Pipeline — OpenSMILE (ComParE_2016 only)
================================================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das

DESCRIPTION:
    Extracts ComParE_2016 (6,373 functionals) features using OpenSMILE across
    three sample rates: 8,000 Hz | 10,000 Hz | 16,000 Hz.
    Total Output = 3 Feature CSVs + 3 Deduplication Reports.

DEDUPLICATION (file-level MD5 hashing — replaces old feature-hash L1):
    Before preprocessing, every collected WAV path is hashed with MD5 on its
    raw bytes. Duplicate files (same bytes, different filenames) are dropped
    immediately — before any audio loading or OpenSMILE extraction runs.
    This is faster, deterministic, and independent of sample rate or features.

PREPROCESSING PIPELINE:
    1. Collect WAV paths from all datasets; compute MD5 per file.
    2. Drop duplicate MD5s globally (log counts per dataset); save dedup report.
    3. Loop over target sample rates [8000, 10000, 16000].
    4. Parallel load, convert to mono, resample, RMS-normalize to 0.04.
    5. Save temp WAVs to disk; extract ComParE via smile.process_files().
    6. Apply L2 (neutralise voice_dataset subject IDs) + VarianceThreshold.
    7. Save CSV and clean up temp files.
"""

import os
import re
import sys
import hashlib
import tempfile
import shutil
import warnings
import logging
import traceback
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import opensmile
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import scipy.stats

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

DATASET_PATHS = {
    "pc_gita": {
        "root": os.path.join(BASE, "Dataset", "PC-GITA"),
    },
    "italian": {
        "root": os.path.join(BASE, "Dataset", "Italian Parkinson's Voice and speech"),
    },
    "voice_dataset": {
        "root": os.path.join(BASE, "Dataset", "Voice_Dataset"),
    },
}

FEATURES_DIR = os.path.join(BASE, "features")
LOG_DIR      = os.path.join(BASE, "logs")

TARGET_RATES = [8000, 10000, 16000]
TARGET_RMS   = 0.04
N_JOBS       = -1

VARIANCE_THRESHOLD = 0.001

META_COLS = {
    "dataset", "subject_id", "language", "speech_type",
    "disease_label", "label_binary", "file", "temp_path",
    "duration_s", "rms_original", "rms_after", "rms_clipped"
}

# =============================================================================
# LOGGING SETUP
# =============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"extraction_opensmile_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("opensmile_extractor")


# =============================================================================
# MD5 HASHING
# =============================================================================
def compute_md5(filepath):
    """Return the MD5 hex digest of a file's raw bytes. Pure stdlib — no deps."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# FILE COLLECTION
# =============================================================================
def collect_pcgita(root):
    records = []
    speaker_re = re.compile(r"(AVPEPUDEA[C]?\d{4})", re.IGNORECASE)
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"): continue
            path_parts = os.path.join(dirpath, fname).replace("\\", "/").lower().split("/")
            if "vowels" not in path_parts or "a" not in path_parts: continue

            label, dlabel = None, None
            for part in path_parts:
                if part in ("pd", "patologica") or part.startswith("pd_"): label, dlabel = 1, "PD"; break
                if part in ("hc", "control") or part.startswith("hc_"): label, dlabel = 0, "HC"; break
            if label is None: continue

            fpath = os.path.join(dirpath, fname)
            m = speaker_re.search(fname)
            records.append({
                "path": fpath, "dataset": "pc_gita",
                "subject_id": m.group(1).upper() if m else "UNKNOWN",
                "language": "es", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
                "file_md5": compute_md5(fpath),
            })
    return records

def collect_italian(root):
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav") or not fname.upper().startswith("VA"): continue
            wav_path = os.path.join(dirpath, fname)
            path_parts = wav_path.replace("\\", "/").lower().split("/")

            label, dlabel = None, None
            if "28 people with parkinson's disease" in wav_path.lower(): label, dlabel = 1, "PD"
            elif "22 elderly healthy control" in wav_path.lower(): label, dlabel = 0, "HC"
            else:
                for part in path_parts:
                    if "parkinson" in part or "pd" in part: label, dlabel = 1, "PD"; break
                    if "healthy" in part or "control" in part or "hc" in part: label, dlabel = 0, "HC"; break
            if label is None: continue

            records.append({
                "path": wav_path, "dataset": "italian", "subject_id": os.path.splitext(fname)[0],
                "language": "it", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
                "file_md5": compute_md5(wav_path),
            })
    return records

def collect_voice_dataset(root):
    records = []
    if not os.path.exists(root): return records
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"): continue
            wav_path = os.path.join(dirpath, fname)
            path_lower = wav_path.lower()

            if "healthy" in path_lower: label, dlabel = 0, "HC"
            elif "parkinson" in path_lower: label, dlabel = 1, "PD"
            else: continue

            records.append({
                "path": wav_path, "dataset": "voice_dataset", "subject_id": os.path.splitext(fname)[0],
                "language": "unknown", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
                "file_md5": compute_md5(wav_path),
            })
    return records


# =============================================================================
# PREPROCESSING
# =============================================================================
def rms_normalize(y, target_rms=TARGET_RMS):
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms < 1e-9: return y, 0.0, 0
    scale = target_rms / current_rms
    peak = np.max(np.abs(y))
    if peak * scale > 1.0: scale = 0.99 / peak
    y_norm = y * scale
    clipped = int(np.sum(np.abs(y_norm) > 1.0))
    return y_norm, float(current_rms), clipped

def preprocess_audio_file(rec, target_sr, temp_dir):
    """Loads, resamples, normalizes, and saves to a temp folder for batch extraction."""
    try:
        y, sr_orig = librosa.load(rec["path"], sr=None, mono=True)
        if sr_orig != target_sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
        duration_s = len(y) / target_sr
        y, original_rms, clipped = rms_normalize(y)
        
        # Create a guaranteed unique filename for the temp file
        safe_name = f"{rec['dataset']}_{rec['subject_id']}_{rec['file']}"
        temp_path = os.path.join(temp_dir, safe_name)
        sf.write(temp_path, y, target_sr, subtype="PCM_16")
        
        rec_out = {k: v for k, v in rec.items() if k != "path"}
        rec_out.update({
            "temp_path": temp_path,
            "duration_s": round(duration_s, 4),
            "rms_original": round(original_rms, 6),
            "rms_after": round(float(np.sqrt(np.mean(y ** 2))), 6),
            "rms_clipped": clipped
        })
        return rec_out
    except Exception as e:
        logger.error(f"Failed to preprocess {rec['file']}: {str(e)}")
        return None


# =============================================================================
# ANTI-LEAKAGE PIPELINE
# =============================================================================
def deduplicate_by_md5(records):
    """
    File-level deduplication using pre-computed MD5 hashes.
    Drops every record whose file_md5 has already been seen, keeping the first
    occurrence. Operates globally across ALL datasets so cross-dataset
    duplicates (e.g. voice_dataset file that also appears in pc_gita) are also
    caught. Returns (deduped_records, report_df).
    """
    seen_md5 = {}       # md5 -> kept record
    kept, report_rows = [], []

    for rec in records:
        md5 = rec["file_md5"]
        if md5 not in seen_md5:
            seen_md5[md5] = rec
            kept.append(rec)
        else:
            k = seen_md5[md5]
            report_rows.append({
                "file_md5":      md5,
                "kept_file":     k["file"],
                "kept_dataset":  k["dataset"],
                "removed_file":  rec["file"],
                "removed_dataset": rec["dataset"],
                "disease_label": rec["disease_label"],
            })

    report_df = pd.DataFrame(report_rows)
    return kept, report_df

def neutralise_voice_dataset_ids(df, logger):
    logger.info("  [ANTI-LEAKAGE L2] Neutralising voice_dataset subject IDs")
    mask_vd = df["dataset"] == "voice_dataset"
    sorted_ids = sorted(df.loc[mask_vd, "subject_id"].unique())
    id_map = {old: f"vd_{i+1:04d}" for i, old in enumerate(sorted_ids)}
    df.loc[mask_vd, "subject_id"] = df.loc[mask_vd, "subject_id"].map(id_map)
    return df


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
def main():
    logger.info("=" * 80)
    logger.info("  OPENSMILE FEATURE EXTRACTION — ComParE_2016 only")
    logger.info("=" * 80)

    # 1. Collect all files (MD5 computed inside each collector)
    logger.info("Collecting WAV files and computing MD5 hashes...")
    all_records = []
    all_records.extend(collect_pcgita(DATASET_PATHS["pc_gita"]["root"]))
    all_records.extend(collect_italian(DATASET_PATHS["italian"]["root"]))
    all_records.extend(collect_voice_dataset(DATASET_PATHS["voice_dataset"]["root"]))

    logger.info(f"Total raw files found: {len(all_records)}")
    if not all_records:
        logger.error("No files found. Exiting.")
        sys.exit(1)

    # 2. Global file-level MD5 deduplication (runs ONCE, before any audio loading)
    logger.info("\n[DEDUP] File-level MD5 deduplication...")
    all_records, dedup_report_df = deduplicate_by_md5(all_records)

    removed_total = len(dedup_report_df)
    if removed_total > 0:
        per_ds = dedup_report_df.groupby("removed_dataset").size()
        for ds, n in per_ds.items():
            logger.info(f"  Removed {n} duplicate(s) from [{ds}]")
    logger.info(f"  Total duplicates removed : {removed_total}")
    logger.info(f"  Unique files proceeding  : {len(all_records)}")

    # Save the single global MD5 dedup report once
    opensmile_dir = os.path.join(FEATURES_DIR, "opensmile")
    os.makedirs(opensmile_dir, exist_ok=True)
    global_dedup_csv = os.path.join(opensmile_dir, f"dedup_report_md5_{TIMESTAMP}.csv")
    if not dedup_report_df.empty:
        dedup_report_df.to_csv(global_dedup_csv, index=False)
        logger.info(f"  Dedup report saved -> {global_dedup_csv}")

    # 3. Initialize LLD-level ComParE extractor for CMVN pipeline
    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    # Domain-invariant targets: MFCC[1-12] + ZCR (structural, not room-acoustic)
    TARGET_LLDS = [f"mfcc_sma[{i}]" for i in range(1, 13)] + ["pcm_zcr_sma"]

    def extract_cmvn_functionals(temp_path, smile_extractor):
        """Extracts LLDs, applies utterance-level CMN (mean-only), computes 4 robust functionals.

        CMN (with_std=False): subtracts per-utterance mean to strip static mic EQ/room tone.
        Variance is deliberately NOT normalized — the frame-level std IS the tremor signal.
        amean is excluded because CMN forces it to exactly 0.0 for every file.
        """
        try:
            lld_df = smile_extractor.process_file(temp_path)
            cols_to_keep = [c for c in lld_df.columns if c in TARGET_LLDS]
            if not cols_to_keep:
                return None
            lld_df = lld_df[cols_to_keep]
            # CMN only — subtract mean, preserve variance (vocal tremor lives here)
            scaler = StandardScaler(with_std=False)
            lld_norm = scaler.fit_transform(lld_df.values)
            features = {"temp_path": temp_path}
            for i, col in enumerate(cols_to_keep):
                data = lld_norm[:, i]
                # amean omitted — always 0.0 after CMN, carries no information
                features[f"{col}_stddev"]   = float(np.std(data))
                features[f"{col}_skewness"] = float(scipy.stats.skew(data, nan_policy='omit'))
                features[f"{col}_kurtosis"] = float(scipy.stats.kurtosis(data, nan_policy='omit'))
                q75, q25 = np.percentile(data, [75, 25])
                features[f"{col}_iqr"]      = float(q75 - q25)
            return features
        except Exception as e:
            logger.error(f"Failed LLD extraction for {temp_path}: {e}")
            return None

    # 4. Process for each Sample Rate
    for sr in TARGET_RATES:
        logger.info("\n" + "=" * 60)
        logger.info(f"  PROCESSING SAMPLE RATE : {sr} Hz")
        logger.info("=" * 60)

        temp_dir = os.path.join(BASE, f"temp_wavs_{sr}")
        os.makedirs(temp_dir, exist_ok=True)

        # Parallel Preprocessing
        logger.info(f"[1/3] Preprocessing audio to {sr} Hz...")
        prepped_records = Parallel(n_jobs=N_JOBS)(
            delayed(preprocess_audio_file)(rec, sr, temp_dir)
            for rec in tqdm(all_records, desc=f"Prep {sr}Hz")
        )
        prepped_records = [r for r in prepped_records if r is not None]
        df_meta = pd.DataFrame(prepped_records)
        temp_paths_list = df_meta['temp_path'].tolist()

        # LLD extraction + utterance-level CMN + 4 robust functionals per LLD
        logger.info(f"\n[2/3] Extracting LLDs, applying CMN (mean-only), aggregating 4 functionals...")
        func_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(extract_cmvn_functionals)(tp, smile_lld)
            for tp in tqdm(temp_paths_list, desc="LLD+CMVN")
        )
        valid_results = [r for r in func_results if r is not None]
        df_features = pd.DataFrame(valid_results)

        # Merge with metadata on temp_path
        df_combined = pd.merge(df_meta, df_features, on='temp_path')

        # Feature columns (exclude all meta)
        feat_cols = [c for c in df_combined.columns
                     if c not in META_COLS and c != "file_md5"
                     and pd.api.types.is_numeric_dtype(df_combined[c])]

        # L2: neutralise voice_dataset subject IDs
        logger.info(f"[3/3] Applying L2 (VarianceThreshold skipped — CMVN pruned to 65 features)...")
        df_combined = neutralise_voice_dataset_ids(df_combined, logger)

        # Skip VarianceThreshold: CMN targets 13 LLDs × 4 functionals = 52 features
        final_feat_cols = feat_cols
        logger.info(f"      CMN features: {len(final_feat_cols)}")

        # Ordered meta columns for output (drop temp_path and file_md5)
        ordered_meta = [c for c in ["dataset", "subject_id", "language", "speech_type",
                                     "disease_label", "label_binary", "file"]
                        if c in df_combined.columns]
        df_out = df_combined[ordered_meta + final_feat_cols]

        # Save
        k_label = f"{int(sr / 1000)}k"
        out_csv = os.path.join(opensmile_dir, f"features_compare_{k_label}.csv")
        df_out.to_csv(out_csv, index=False)
        logger.info(f"      -> {out_csv}")

        # Cleanup temp WAVs
        logger.info(f"Cleaning up temporary {sr} Hz audio files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("\n" + "=" * 80)
    logger.info("  ALL EXTRACTIONS COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()