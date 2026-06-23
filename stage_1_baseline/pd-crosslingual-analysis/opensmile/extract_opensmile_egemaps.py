"""
OpenSMILE eGeMAPS Feature Extraction
=====================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Generated for User

PURPOSE:
    Extract eGeMAPS (88 features) using OpenSMILE from raw audio files.
    This is a validated clinical feature set.

DATASETS:
    PC-GITA (Spanish)     : Vowels/hc/A/*.wav, Vowels/pd/A/*.wav
    Italian Parkinson's   : VA*.wav files (recursive)
    Voice_Dataset         : Healthy/*.wav, Parkinsons/*.wav

HOW TO RUN:
    1. pip install opensmile
    2. venv\Scripts\activate
    3. python scripts\extract_opensmile_egemaps.py

OUTPUT:
    features/features_opensmile_egemaps.csv
"""

import os
import re
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

DATASET_PATHS = {
    "pc_gita": os.path.join(BASE, "Dataset", "PC-GITA"),
    "italian": os.path.join(BASE, "Dataset", "Italian Parkinson's Voice and speech"),
    "voice_dataset": os.path.join(BASE, "Dataset", "Voice_Dataset"),
}

OUTPUT_CSV = os.path.join(BASE, "features", "features_opensmile_egemaps.csv")
LOG_DIR = os.path.join(BASE, "logs")


# ============================================================================
# LOGGING SETUP
# ============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"extraction_opensmile_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("opensmile_extractor")


# ============================================================================
# OPENSMILE SETUP
# ============================================================================
def setup_opensmile():
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        logger.info(f"OpenSMILE version: {opensmile.__version__}")
        logger.info("Feature set: eGeMAPSv02 (88 features)")
        return smile
    except ImportError:
        logger.error("OpenSMILE not installed. Please run: pip install opensmile")
        sys.exit(1)


# ============================================================================
# DATASET COLLECTION FUNCTIONS (Matched to previous pipeline)
# ============================================================================
def collect_pcgita(root):
    records = []
    speaker_re = re.compile(r"(AVPEPUDEA[C]?\d{4})", re.IGNORECASE)
    
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            
            wav_path = os.path.join(dirpath, fname)
            path_parts = wav_path.replace("\\", "/").lower().split("/")
            
            if "vowels" not in path_parts or "a" not in path_parts:
                continue
            
            label, dlabel = None, None
            for part in path_parts:
                if part in ("pd", "patologica") or part.startswith("pd_"):
                    label, dlabel = 1, "PD"
                    break
                if part in ("hc", "control") or part.startswith("hc_"):
                    label, dlabel = 0, "HC"
                    break
            
            if label is None:
                continue
            
            m = speaker_re.search(fname)
            records.append({
                "path": wav_path,
                "dataset": "pc_gita",
                "subject_id": m.group(1).upper() if m else "UNKNOWN",
                "language": "es",
                "speech_type": "sustained_vowel_a",
                "disease_label": dlabel,
                "label_binary": label,
                "file": fname,
            })
    return records


def collect_italian(root):
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav") or not fname.upper().startswith("VA"):
                continue
            
            wav_path = os.path.join(dirpath, fname)
            path_parts = wav_path.replace("\\", "/").lower().split("/")
            
            label, dlabel = None, None
            if "28 people with parkinson's disease" in wav_path.lower():
                label, dlabel = 1, "PD"
            elif "22 elderly healthy control" in wav_path.lower():
                label, dlabel = 0, "HC"
            else:
                for part in path_parts:
                    if "parkinson" in part or "pd" in part:
                        label, dlabel = 1, "PD"
                        break
                    if "healthy" in part or "control" in part or "hc" in part:
                        label, dlabel = 0, "HC"
                        break
            
            if label is None:
                continue
            
            records.append({
                "path": wav_path,
                "dataset": "italian",
                "subject_id": os.path.splitext(fname)[0],
                "language": "it",
                "speech_type": "sustained_vowel_a",
                "disease_label": dlabel,
                "label_binary": label,
                "file": fname,
            })
    return records


def collect_voice_dataset(root):
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            
            wav_path = os.path.join(dirpath, fname)
            path_lower = wav_path.lower()
            
            if "healthy" in path_lower:
                label, dlabel = 0, "HC"
            elif "parkinson" in path_lower:
                label, dlabel = 1, "PD"
            else:
                continue
            
            records.append({
                "path": wav_path,
                "dataset": "voice_dataset",
                "subject_id": os.path.splitext(fname)[0],
                "language": "unknown",
                "speech_type": "sustained_vowel_a",
                "disease_label": dlabel,
                "label_binary": label,
                "file": fname,
            })
    return records


# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("  OpenSMILE eGeMAPS FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"  Start Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Output CSV : {OUTPUT_CSV}")
    
    # Setup OpenSMILE
    logger.info("\n[STEP 1] Initializing OpenSMILE...")
    smile = setup_opensmile()
    
    # Collect Files
    logger.info("\n[STEP 2] Collecting audio files...")
    all_records = []
    
    if os.path.exists(DATASET_PATHS["pc_gita"]):
        recs = collect_pcgita(DATASET_PATHS["pc_gita"])
        all_records.extend(recs)
        logger.info(f"  PC-GITA        : {len(recs)} files")
        
    if os.path.exists(DATASET_PATHS["italian"]):
        recs = collect_italian(DATASET_PATHS["italian"])
        all_records.extend(recs)
        logger.info(f"  Italian        : {len(recs)} files")
        
    if os.path.exists(DATASET_PATHS["voice_dataset"]):
        recs = collect_voice_dataset(DATASET_PATHS["voice_dataset"])
        all_records.extend(recs)
        logger.info(f"  Voice_Dataset  : {len(recs)} files")
        
    if not all_records:
        logger.error("No files found! Check dataset paths.")
        sys.exit(1)
        
    logger.info("-" * 40)
    logger.info(f"  TOTAL FILES    : {len(all_records)}")
    logger.info(f"  PD             : {sum(1 for r in all_records if r['label_binary'] == 1)}")
    logger.info(f"  HC             : {sum(1 for r in all_records if r['label_binary'] == 0)}")
    logger.info("-" * 40)

    # Extract Features
    logger.info("\n[STEP 3] Extracting OpenSMILE features...")
    results = []
    errors = []
    
    for rec in tqdm(all_records, desc="Extracting eGeMAPS"):
        try:
            # Process with OpenSMILE
            features_df = smile.process_file(rec["path"])
            features = features_df.iloc[0].to_dict()
            
            # Combine metadata and features
            row = {k: v for k, v in rec.items() if k != "path"}
            row.update(features)
            results.append(row)
        except Exception as e:
            errors.append({"file": rec["file"], "error": str(e)})

    if errors:
        logger.warning(f"\n  {len(errors)} files failed to process:")
        for e in errors[:5]:
            logger.warning(f"    - {e['file']}: {e['error']}")

    # Save Data
    logger.info("\n[STEP 4] Saving DataFrame...")
    df = pd.DataFrame(results)
    
    # Enforce column order (Metadata first, then features)
    meta_cols = ["dataset", "subject_id", "language", "speech_type", 
                 "disease_label", "label_binary", "file"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total Rows       : {len(df)}")
    logger.info(f"  Total Columns    : {len(df.columns)} (7 Meta + {len(feat_cols)} Features)")
    logger.info(f"  Processing Time  : {elapsed:.1f} seconds")
    logger.info(f"  Saved to         : {OUTPUT_CSV}")
    logger.info(f"  Log file         : {LOG_FILE}")


if __name__ == "__main__":
    main()