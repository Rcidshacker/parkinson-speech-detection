"""
Sustained Vowel "a" Feature Extraction Pipeline — OpenSMILE (ComParE)
================================================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das

DESCRIPTION:
    This script replaces the Praat/librosa pipeline with the industry-standard
    OpenSMILE toolkit. It extracts massive feature sets:
      1. ComParE_2016 (6,373 functionals)
    
    It runs the entire pipeline automatically across three sample rates:
      - 8,000 Hz
      - 10,000 Hz
      - 16,000 Hz
    
    Total Output = 6 Feature CSVs + 6 Deduplication Reports.

PREPROCESSING PIPELINE:
    1. Loop over target sample rates [8000, 10000, 16000]
    2. Parallel load, convert to mono, resample, and RMS normalize to 0.04
    3. Save temporarily to disk (OpenSMILE's C++ backend is fastest via file I/O)
    4. Extract ComParE via smile.process_files()
    6. Apply Anti-Leakage (L1, L2) and VarianceThreshold (FQ3)
    7. Save CSVs and clean up temp files.
"""

import os
import glob
import re
import sys
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
    "neurovoz": {
        "root": os.path.join(BASE, "Dataset", "neurovoz", "zenodo_upload", "audios"),
    },
}

FEATURES_DIR = os.path.join(BASE, "features")
LOG_DIR      = os.path.join(BASE, "logs")

TARGET_RATES = [8000]
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
            
            m = speaker_re.search(fname)
            records.append({
                "path": os.path.join(dirpath, fname), "dataset": "pc_gita", 
                "subject_id": m.group(1).upper() if m else "UNKNOWN",
                "language": "es", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
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
                    if "healthy" in part or "control" or "hc" in part: label, dlabel = 0, "HC"; break
            if label is None: continue
            
            records.append({
                "path": wav_path, "dataset": "italian", "subject_id": os.path.splitext(fname)[0],
                "language": "it", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
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
            })
    return records

def collect_neurovoz_files(root):
    """
    Scans Neurovoz audios and maps metadata from filenames.
    Pattern: [LABEL]_A[TASK]_[ID].wav
    """
    records = []
    if not os.path.exists(root): return records
    
    # We only want the sustained vowel 'a' (Tasks A1, A2, A3)
    for wav_path in glob.glob(os.path.join(root, "*.wav")):
        filename = os.path.basename(wav_path)
        if "_A" in filename:  # Filters for sustained vowel tasks
            parts = filename.split('_')
            
            # Map Label: PD -> 1, HC -> 0
            label = 1 if parts[0] == 'PD' else 0
            # Extract Subject ID (the last part before .wav)
            subject_id = parts[-1].replace('.wav', '')
            
            records.append({
                "path": wav_path,
                "dataset": "neurovoz",
                "subject_id": f"neuro_{subject_id}", # Prefix prevents ID collision with other datasets
                "language": "es",
                "speech_type": "vowel_a",
                "disease_label": "PD" if label == 1 else "HC",
                "label_binary": label,
                "file": filename
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
def deduplicate_voice_dataset(df, feat_cols, logger):
    logger.info("  [ANTI-LEAKAGE L1] voice_dataset deduplication")
    mask_vd = df["dataset"] == "voice_dataset"
    df_vd = df[mask_vd].copy().reset_index(drop=True)
    df_other = df[~mask_vd].copy()

    df_vd["_fhash"] = df_vd[feat_cols].round(4).apply(lambda row: hash(tuple(row)), axis=1)
    dup_mask = df_vd.duplicated(subset=["_fhash"], keep=False)
    dup_groups = df_vd[dup_mask].groupby("_fhash")

    report_rows = []
    for h, grp in dup_groups:
        ids, files = grp["subject_id"].tolist(), grp["file"].tolist()
        label = grp["disease_label"].iloc[0]
        for i in range(1, len(ids)):
            report_rows.append({
                "kept_subject_id": ids[0], "kept_file": files[0],
                "removed_subject_id": ids[i], "removed_file": files[i],
                "disease_label": label, "feature_hash": h,
            })

    report_df = pd.DataFrame(report_rows)
    df_vd_clean = df_vd.drop_duplicates(subset=["_fhash"], keep="first").drop(columns=["_fhash"])
    
    logger.info(f"  Rows removed : {len(df_vd) - len(df_vd_clean)}")
    return pd.concat([df_other, df_vd_clean], ignore_index=True), report_df

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
    logger.info("  OPENSMILE FEATURE EXTRACTION (ComParE)")
    logger.info("=" * 80)

    # 1. Collect all files once
    all_records = []
    all_records.extend(collect_pcgita(DATASET_PATHS["pc_gita"]["root"]))
    all_records.extend(collect_italian(DATASET_PATHS["italian"]["root"]))
    all_records.extend(collect_voice_dataset(DATASET_PATHS["voice_dataset"]["root"]))
    all_records.extend(collect_neurovoz_files(DATASET_PATHS["neurovoz"]["root"]))
    
    logger.info(f"Total raw files found: {len(all_records)}")
    if not all_records:
        logger.error("No files found. Exiting.")
        sys.exit(1)

    # 2. Initialize OpenSMILE extractors
    smile_compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # 3. Process for each Sample Rate
    for sr in TARGET_RATES:
        logger.info("\n" + "=" * 60)
        logger.info(f"  PROCESSING SAMPLE RATE : {sr} Hz")
        logger.info("=" * 60)
        
        temp_dir = os.path.join(BASE, f"temp_wavs_{sr}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Parallel Preprocessing
        logger.info(f"[1/4] Preprocessing audio to {sr} Hz...")
        prepped_records = Parallel(n_jobs=N_JOBS)(
            delayed(preprocess_audio_file)(rec, sr, temp_dir) 
            for rec in tqdm(all_records, desc=f"Prep {sr}Hz")
        )
        prepped_records = [r for r in prepped_records if r is not None]
        df_meta = pd.DataFrame(prepped_records)
        temp_paths_list = df_meta['temp_path'].tolist()

        for config_name, extractor in [("compare", smile_compare)]:
            logger.info(f"\n[2/4] Extracting {config_name.upper()} features via OpenSMILE...")
            
            # OpenSMILE native batch processing
            df_features = extractor.process_files(temp_paths_list)
            
            # Clean up the MultiIndex returned by OpenSMILE
            df_features = df_features.reset_index()
            df_features = df_features.rename(columns={'file': 'temp_path'})
            if 'start' in df_features.columns: df_features = df_features.drop(columns=['start', 'end', 'timedelta'], errors='ignore')
            
            # Merge with metadata
            df_combined = pd.merge(df_meta, df_features, on='temp_path')
            
            # Get feature columns specifically
            feat_cols = [c for c in df_combined.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df_combined[c])]
            
            # Anti-Leakage
            logger.info(f"[3/4] Running Anti-Leakage constraints for {config_name}...")
            df_combined, report_df = deduplicate_voice_dataset(df_combined, feat_cols, logger)
            df_combined = neutralise_voice_dataset_ids(df_combined, logger)
            
            # Variance Threshold Cleanup
            vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
            vt.fit(df_combined[feat_cols].fillna(0))
            kept_mask = vt.get_support()
            final_feat_cols = [feat_cols[i] for i, keep in enumerate(kept_mask) if keep]
            df_combined = df_combined[list(META_COLS.intersection(df_combined.columns)) + final_feat_cols]
            
            logger.info(f"      Features before cleanup: {len(feat_cols)} | After: {len(final_feat_cols)}")

            # Save Outputs
            logger.info(f"[4/4] Saving output files...")
            k_label = f"{int(sr/1000)}k"
            out_csv = os.path.join(FEATURES_DIR, f"features_{config_name}_{k_label}.csv")
            dedup_csv = os.path.join(FEATURES_DIR, f"dedup_report_{config_name}_{k_label}.csv")
            
            # Drop the temp_path column before saving
            if 'temp_path' in df_combined.columns: df_combined = df_combined.drop(columns=['temp_path'])
            df_combined.to_csv(out_csv, index=False)
            if not report_df.empty: report_df.to_csv(dedup_csv, index=False)
            logger.info(f"      -> {out_csv}")

        # Cleanup Temp Directory
        logger.info(f"Cleaning up temporary {sr} Hz audio files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("\n" + "=" * 80)
    logger.info("  ALL EXTRACTIONS COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()