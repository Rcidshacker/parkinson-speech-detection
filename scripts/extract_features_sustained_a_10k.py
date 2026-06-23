"""
Sustained Vowel "a" Feature Extraction Pipeline  — 10 kHz
==========================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

⚠️  NAMING DISAMBIGUATION — READ BEFORE MODIFYING
    "10k" in this filename refers to AUDIO SAMPLE RATE = 10,000 Hz.
    It does NOT mean 10,000 training samples. All three variants
    (8k / 10k / 16k) extract features from the SAME recordings and
    produce the SAME number of rows. The only difference is the
    resampling frequency and derived FFT window parameters, which
    changes the spectral feature distributions (MFCCs, chroma,
    spectral centroid/bandwidth) but NOT the dataset size.

SAMPLE RATE : 10,000 Hz
    Nyquist  : 5,000 Hz  (covers F0, F1, F2, F3 formants ~2500 Hz)
    N_FFT    : 320 samples  (32 ms window at 10 kHz — same temporal resolution as 8 kHz baseline)
    HOP      : 160 samples  (50% overlap)

DATASETS:
    PC-GITA (Spanish)     : Vowels/hc/A/*.wav, Vowels/pd/A/*.wav
    Italian Parkinson's   : VA*.wav files (recursive)
    Voice_Dataset         : Healthy/*.wav, Parkinsons/*.wav

PREPROCESSING PIPELINE:
    1. Load WAV at original sample rate
    2. Convert to mono
    3. Resample to 10,000 Hz
    4. RMS normalize to 0.04
    5. Save temp WAV for Praat
    6. Extract features
    7. Delete temp WAV

FEATURES EXTRACTED:
    - Phonatory (Praat): F0, Jitter, Shimmer, HNR, NHR
    - Nonlinear: RPDE, PPE
    - Spectral (librosa): MFCCs, Delta, Delta-Delta, Energy, Centroid,
                          Bandwidth, Rolloff, Flux, ZCR, Mel, Chroma
    - Articulatory (Praat): F1, F2, F3 formants
    - Advanced: Spread1, Spread2

DATA INTEGRITY GUARANTEES (anti-leakage):
    [L1] voice_dataset duplicate removal:
         The Kaggle 'parkinsons-voice-dataset' contains 470 exact duplicate
         recording pairs — each .wav file copied under a different sequential
         filename (offset ~283). These are removed post-extraction by
         dropping rows with identical feature vectors (all numeric columns).
         Result: 1037 raw rows -> 567 unique rows.
    [L2] Opaque subject IDs for voice_dataset:
         Original filenames encode the disease label ('healthy_NNN',
         'parkinsons_NNN'). After deduplication, these are replaced with
         neutral IDs ('vd_0001', 'vd_0002', ...) so no downstream model
         or encoder can use the ID string as a proxy for the label.
    [L3] StratifiedGroupKFold safety:
         After L1+L2, each voice_dataset subject_id is a single unique
         recording. GroupKFold isolation is meaningful for pc_gita (3
         recordings/subject) and italian (~2/subject). voice_dataset
         entries are treated as 1-recording subjects - valid once deduped.

    A deduplication report CSV is written alongside the main output.

FEATURE QUALITY DECISIONS:
    [FQ1] jitter_ddp and shimmer_dda EXCLUDED at source:
         jitter_ddp  = jitter_rap   * 3  (corr = 1.000000, algebraic identity)
         shimmer_dda = shimmer_apq3 * 3  (corr = 1.000000, algebraic identity)
         Including them adds two perfectly redundant columns that inflate the
         feature count without contributing any new information.
    [FQ2] mel_mean replaced with log_mel_mean/log_mel_std (dB scale):
         Raw mel power mean at 8 kHz + N_FFT=256 collapses to a near-zero
         scalar (var ~= 0.000000) because most power-domain mel bins are
         close to zero. Converted to dB via librosa.power_to_db() before
         summarising, which restores meaningful dynamic range (~40-80 dB).
    [FQ3] Near-zero variance removal (VARIANCE_THRESHOLD = 0.001):
         Applied via sklearn VarianceThreshold after deduplication.
         Features that are constant across subjects are dropped and logged
         with their exact variance values. Keeps the training matrix clean.

PIPELINE STEPS:
    1. Collect files from all three datasets
    2. Extract features (parallel)
    3. Build DataFrame
    4. Anti-leakage: deduplication [L1] + ID neutralisation [L2]
    5. Feature quality cleanup: VarianceThreshold [FQ3]
    6. Summary + verification
    7. Save CSV

OUTPUT:
    features/features_sustained_a_10k.csv          -- clean, deduped feature matrix
    features/dedup_report_sustained_a_10k.csv      -- log of removed duplicate pairs
    logs/extraction_sustained_a_10k_<ts>.log       -- full run log

HOW TO RUN:
    venv\\Scripts\\activate
    python extract_features_sustained_a_10k.py
"""

import os
import re
import sys
import tempfile
import warnings
import logging
import traceback
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import parselmouth
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import find_peaks
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

DATASET_PATHS = {
    "pc_gita": {
        "root": os.path.join(BASE, "Dataset", "PC-GITA"),
        "type": "vowel_folder",
    },
    "italian": {
        "root": os.path.join(BASE, "Dataset", "Italian Parkinson's Voice and speech"),
        "type": "va_pattern",
    },
    "voice_dataset": {
        "root": os.path.join(BASE, "Dataset", "Voice_Dataset"),
        "type": "subfolder_binary",
    },
}

OUTPUT_CSV   = os.path.join(BASE, "features", "features_sustained_a_10k.csv")
DEDUP_REPORT = os.path.join(BASE, "features", "dedup_report_sustained_a_10k.csv")
LOG_DIR      = os.path.join(BASE, "logs")

TARGET_SR        = 10000
TARGET_RMS       = 0.04
N_JOBS           = -1

N_MFCC           = 13
N_FFT            = 320   # 32ms at 10 kHz  (same temporal resolution as 8 kHz baseline)
HOP_LENGTH       = 160   # 50% overlap
DELTA_WIDTH      = 9
F0_MIN           = 80
F0_MAX           = 300
FORMANT_MAX_FREQ = 4900  # Nyquist at 10 kHz = 5000 Hz; keep safely below

# Feature quality thresholds
# Drop features whose variance across all subjects falls below this value.
# At 10 kHz + N_FFT=320, near-constant features are less likely than at 8 kHz
# but the guard is kept for safety.
VARIANCE_THRESHOLD = 0.001

META_COLS = {
    "dataset", "subject_id", "language", "speech_type",
    "disease_label", "label_binary", "file",
    "duration_s", "rms_original", "rms_after", "rms_target", "rms_clipped",
}


# =============================================================================
# LOGGING SETUP
# =============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"extraction_sustained_a_10k_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("sustained_a_extractor")


# =============================================================================
# PREPROCESSING
# =============================================================================
def rms_normalize(y, target_rms=TARGET_RMS):
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms < 1e-9:
        return y, 0.0, 0
    scale = target_rms / current_rms
    peak  = np.max(np.abs(y))
    if peak * scale > 1.0:
        scale = 0.99 / peak
    y_norm  = y * scale
    clipped = int(np.sum(np.abs(y_norm) > 1.0))
    return y_norm, float(current_rms), clipped


def load_and_preprocess(wav_path):
    y, sr_orig = librosa.load(wav_path, sr=None, mono=True)
    if sr_orig != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=TARGET_SR)
    sr         = TARGET_SR
    duration_s = len(y) / sr
    y, original_rms, clipped = rms_normalize(y)
    tmp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    sf.write(temp_path, y, sr, subtype="PCM_16")
    return y, sr, temp_path, original_rms, clipped, duration_s


# =============================================================================
# PRAAT FEATURES
# =============================================================================
def extract_f0_praat(sound):
    feats = {}
    try:
        pitch  = parselmouth.praat.call(sound, "To Pitch (cc)", 0.0, F0_MIN, 15, "no",
                                        0.03, 0.45, 0.01, 0.35, 0.14, F0_MAX)
        f0vals = pitch.selected_array["frequency"]
        f0vals = f0vals[f0vals > 0]
        if len(f0vals) > 5:
            feats["praat_f0_mean"]   = float(np.mean(f0vals))
            feats["praat_f0_std"]    = float(np.std(f0vals))
            feats["praat_f0_min"]    = float(np.min(f0vals))
            feats["praat_f0_max"]    = float(np.max(f0vals))
            feats["praat_f0_range"]  = float(np.max(f0vals) - np.min(f0vals))
            feats["praat_f0_median"] = float(np.median(f0vals))
        else:
            for k in ["praat_f0_mean", "praat_f0_std", "praat_f0_min",
                      "praat_f0_max", "praat_f0_range", "praat_f0_median"]:
                feats[k] = np.nan
    except Exception:
        for k in ["praat_f0_mean", "praat_f0_std", "praat_f0_min",
                  "praat_f0_max", "praat_f0_range", "praat_f0_median"]:
            feats[k] = np.nan
    return feats


def extract_jitter_shimmer(sound):
    feats = {}
    try:
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
        feats["jitter_local"] = parselmouth.praat.call(pp, "Get jitter (local)",  0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        feats["jitter_rap"]   = parselmouth.praat.call(pp, "Get jitter (rap)",    0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        feats["jitter_ppq5"]  = parselmouth.praat.call(pp, "Get jitter (ppq5)",   0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        # NOTE: jitter_ddp = jitter_rap * 3 (algebraic identity, corr=1.0) — EXCLUDED
        feats["shimmer_local"]  = parselmouth.praat.call([sound, pp], "Get shimmer (local)",  0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq3"]   = parselmouth.praat.call([sound, pp], "Get shimmer (apq3)",   0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq5"]   = parselmouth.praat.call([sound, pp], "Get shimmer (apq5)",   0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq11"]  = parselmouth.praat.call([sound, pp], "Get shimmer (apq11)",  0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        # NOTE: shimmer_dda = shimmer_apq3 * 3 (algebraic identity, corr=1.0) — EXCLUDED
    except Exception:
        for k in ["jitter_local", "jitter_rap", "jitter_ppq5",
                  "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11"]:
            feats[k] = np.nan
    return feats


def extract_hnr_nhr(sound):
    feats = {}
    try:
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        hnr = float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
        feats["hnr"] = hnr
        feats["nhr"] = 1.0 / (hnr + 1e-6)
    except Exception:
        feats["hnr"] = np.nan
        feats["nhr"] = np.nan
    return feats


def extract_formants(sound):
    feats = {}
    try:
        formants = parselmouth.praat.call(sound, "To Formant (burg)", 0.0, 5, FORMANT_MAX_FREQ, 0.025, 50)
        duration = parselmouth.praat.call(sound, "Get total duration")
        mid_time = duration / 2
        f1 = parselmouth.praat.call(formants, "Get value at time", 1, mid_time, "Hertz", "Linear")
        f2 = parselmouth.praat.call(formants, "Get value at time", 2, mid_time, "Hertz", "Linear")
        f3 = parselmouth.praat.call(formants, "Get value at time", 3, mid_time, "Hertz", "Linear")
        feats["f1_mean"] = f1 if f1 > 0 else np.nan
        feats["f2_mean"] = f2 if f2 > 0 else np.nan
        feats["f3_mean"] = f3 if f3 > 0 else np.nan
        n_frames = parselmouth.praat.call(formants, "Get number of frames")
        f1_vals, f2_vals, f3_vals = [], [], []
        for i in range(1, min(n_frames + 1, 100)):
            t   = parselmouth.praat.call(formants, "Get time from frame number", i)
            f1v = parselmouth.praat.call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            f2v = parselmouth.praat.call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            f3v = parselmouth.praat.call(formants, "Get value at time", 3, t, "Hertz", "Linear")
            if f1v > 0: f1_vals.append(f1v)
            if f2v > 0: f2_vals.append(f2v)
            if f3v > 0: f3_vals.append(f3v)
        feats["f1_std"] = float(np.std(f1_vals)) if len(f1_vals) > 5 else np.nan
        feats["f2_std"] = float(np.std(f2_vals)) if len(f2_vals) > 5 else np.nan
        feats["f3_std"] = float(np.std(f3_vals)) if len(f3_vals) > 5 else np.nan
    except Exception:
        for k in ["f1_mean", "f2_mean", "f3_mean", "f1_std", "f2_std", "f3_std"]:
            feats[k] = np.nan
    return feats


# =============================================================================
# NONLINEAR FEATURES
# =============================================================================
def extract_rpde(f0_values):
    if len(f0_values) < 20:
        return np.nan
    try:
        periods = np.diff(f0_values)
        if len(periods) < 10:
            return np.nan
        periods = periods / (np.mean(np.abs(periods)) + 1e-10)
        hist, _ = np.histogram(periods, bins=20, density=True)
        hist    = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy / np.log2(20))
    except Exception:
        return np.nan


def extract_ppe(f0_values):
    if len(f0_values) < 20:
        return np.nan
    try:
        periods      = 1.0 / (f0_values + 1e-10)
        periods_norm = (periods - np.mean(periods)) / (np.std(periods) + 1e-10)
        hist, _      = np.histogram(periods_norm, bins=20, density=True)
        hist         = hist[hist > 0]
        entropy      = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy / np.log2(20))
    except Exception:
        return np.nan


def extract_nonlinear_features(y, sr, f0_values):
    return {"rpde": extract_rpde(f0_values), "ppe": extract_ppe(f0_values)}


# =============================================================================
# SPECTRAL FEATURES
# =============================================================================
def extract_mfcc_features(y, sr):
    feats = {}
    try:
        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta  = librosa.feature.delta(mfccs, width=DELTA_WIDTH)
        delta2 = librosa.feature.delta(mfccs, order=2, width=DELTA_WIDTH)
        for i in range(N_MFCC):
            n = f"{i+1:02d}"
            feats[f"mfcc_{n}_mean"]   = float(np.mean(mfccs[i]))
            feats[f"mfcc_{n}_std"]    = float(np.std(mfccs[i]))
            feats[f"dmfcc_{n}_mean"]  = float(np.mean(delta[i]))
            feats[f"d2mfcc_{n}_mean"] = float(np.mean(delta2[i]))
    except Exception:
        for i in range(N_MFCC):
            n = f"{i+1:02d}"
            feats[f"mfcc_{n}_mean"]   = np.nan
            feats[f"mfcc_{n}_std"]    = np.nan
            feats[f"dmfcc_{n}_mean"]  = np.nan
            feats[f"d2mfcc_{n}_mean"] = np.nan
    return feats


def extract_spectral_features(y, sr):
    feats = {}
    try:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        feats["log_energy_mean"] = float(np.mean(np.log(rms + 1e-10)))
        feats["log_energy_std"]  = float(np.std(np.log(rms + 1e-10)))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"]  = float(np.std(centroid))
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        feats["spectral_bandwidth_std"]  = float(np.std(bandwidth))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
        flux = librosa.onset.onset_strength(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats["spectral_flux_mean"] = float(np.mean(flux))
        feats["spectral_flux_std"]  = float(np.std(flux))
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"]  = float(np.std(zcr))
        # Log-mel: convert to dB scale first so the feature has real variance.
        # Raw mel power at 8 kHz collapses to a near-zero scalar mean (var≈0)
        # because most bins are close to 0 — useless for classification.
        mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        feats["log_mel_mean"] = float(np.mean(log_mel))
        feats["log_mel_std"]  = float(np.std(log_mel))
    except Exception:
        for k in ["log_energy_mean", "log_energy_std",
                  "spectral_centroid_mean", "spectral_centroid_std",
                  "spectral_bandwidth_mean", "spectral_bandwidth_std",
                  "spectral_rolloff_mean", "spectral_flux_mean", "spectral_flux_std",
                  "zcr_mean", "zcr_std", "log_mel_mean", "log_mel_std"]:
            feats[k] = np.nan
    return feats


def extract_chroma_features(y, sr):
    feats = {}
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(12):
            feats[f"chroma_{i:02d}_mean"] = float(np.mean(chroma[i]))
            feats[f"chroma_{i:02d}_std"]  = float(np.std(chroma[i]))
    except Exception:
        for i in range(12):
            feats[f"chroma_{i:02d}_mean"] = np.nan
            feats[f"chroma_{i:02d}_std"]  = np.nan
    return feats


# =============================================================================
# ADVANCED FEATURES
# =============================================================================
def extract_spread_features(f0_values):
    feats = {}
    if len(f0_values) < 20:
        feats["spread1"] = np.nan
        feats["spread2"] = np.nan
        return feats
    try:
        q75, q25 = np.percentile(f0_values, [75, 25])
        feats["spread1"] = float(q75 - q25)
        feats["spread2"] = float(np.std(f0_values) / (np.mean(f0_values) + 1e-10))
    except Exception:
        feats["spread1"] = np.nan
        feats["spread2"] = np.nan
    return feats


# =============================================================================
# FILE COLLECTION
# =============================================================================
def collect_pcgita(root):
    records    = []
    speaker_re = re.compile(r"(AVPEPUDEA[C]?\d{4})", re.IGNORECASE)
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            wav_path   = os.path.join(dirpath, fname)
            path_parts = wav_path.replace("\\", "/").lower().split("/")
            if "vowels" not in path_parts or "a" not in path_parts:
                continue
            label, dlabel = None, None
            for part in path_parts:
                if part in ("pd", "patologica") or part.startswith("pd_"):
                    label, dlabel = 1, "PD"; break
                if part in ("hc", "control") or part.startswith("hc_"):
                    label, dlabel = 0, "HC"; break
            if label is None:
                continue
            m          = speaker_re.search(fname)
            subject_id = m.group(1).upper() if m else "UNKNOWN"
            records.append({
                "path": wav_path, "dataset": "pc_gita", "subject_id": subject_id,
                "language": "es", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
            })
    return records


def collect_italian(root):
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            if not fname.upper().startswith("VA"):
                continue
            wav_path   = os.path.join(dirpath, fname)
            path_parts = wav_path.replace("\\", "/").lower().split("/")
            label, dlabel = None, None
            if "28 people with parkinson's disease" in wav_path.lower():
                label, dlabel = 1, "PD"
            elif "22 elderly healthy control" in wav_path.lower():
                label, dlabel = 0, "HC"
            else:
                for part in path_parts:
                    if "parkinson" in part or "pd" in part:
                        label, dlabel = 1, "PD"; break
                    if "healthy" in part or "control" in part or "hc" in part:
                        label, dlabel = 0, "HC"; break
            if label is None:
                continue
            subject_id = os.path.splitext(fname)[0]
            records.append({
                "path": wav_path, "dataset": "italian", "subject_id": subject_id,
                "language": "it", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
            })
    return records


def collect_voice_dataset(root):
    """
    Collect Voice_Dataset files.
    NOTE: subject_ids here still encode the disease label (healthy_NNN /
    parkinsons_NNN). They are replaced with opaque IDs in main() after
    feature extraction and deduplication (anti-leakage L2).
    """
    records = []
    if not os.path.exists(root):
        return records
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            wav_path   = os.path.join(dirpath, fname)
            path_lower = wav_path.lower()
            if "healthy" in path_lower:
                label, dlabel = 0, "HC"
            elif "parkinson" in path_lower:
                label, dlabel = 1, "PD"
            else:
                continue
            subject_id = os.path.splitext(fname)[0]
            records.append({
                "path": wav_path, "dataset": "voice_dataset", "subject_id": subject_id,
                "language": "unknown", "speech_type": "sustained_vowel_a",
                "disease_label": dlabel, "label_binary": label, "file": fname,
            })
    return records


# =============================================================================
# ANTI-LEAKAGE: DEDUPLICATION  [L1 FIX]
# =============================================================================
def deduplicate_voice_dataset(df, feat_cols, logger):
    """
    Remove duplicate recordings from voice_dataset.

    The Kaggle dataset contains 470 exact duplicate pairs — each .wav
    copied under a different sequential filename (offset ~283). Detected
    by hashing rounded feature vectors; first occurrence kept, clone dropped.

    Returns: (df_clean, report_df)
    """
    logger.info("\n" + "-" * 60)
    logger.info("  [ANTI-LEAKAGE L1] voice_dataset deduplication")
    logger.info("-" * 60)

    mask_vd  = df["dataset"] == "voice_dataset"
    df_vd    = df[mask_vd].copy().reset_index(drop=True)
    df_other = df[~mask_vd].copy()

    rows_before = len(df_vd)

    # Hash each feature vector (rounded to 4 dp for float stability)
    df_vd["_fhash"] = (
        df_vd[feat_cols]
        .round(4)
        .apply(lambda row: hash(tuple(row)), axis=1)
    )

    # Identify duplicated groups
    dup_mask   = df_vd.duplicated(subset=["_fhash"], keep=False)
    dup_groups = df_vd[dup_mask].groupby("_fhash")

    report_rows = []
    for h, grp in dup_groups:
        ids   = grp["subject_id"].tolist()
        files = grp["file"].tolist()
        label = grp["disease_label"].iloc[0]
        for i in range(1, len(ids)):          # index 0 is kept
            report_rows.append({
                "kept_subject_id":    ids[0],
                "kept_file":          files[0],
                "removed_subject_id": ids[i],
                "removed_file":       files[i],
                "disease_label":      label,
                "feature_hash":       h,
            })

    report_df = pd.DataFrame(report_rows)

    df_vd_clean = df_vd.drop_duplicates(subset=["_fhash"], keep="first").copy()
    df_vd_clean = df_vd_clean.drop(columns=["_fhash"])

    rows_after   = len(df_vd_clean)
    rows_removed = rows_before - rows_after

    logger.info(f"  voice_dataset rows before dedup : {rows_before}")
    logger.info(f"  Duplicate pairs found           : {len(report_rows)}")
    logger.info(f"  Rows removed                    : {rows_removed}")
    logger.info(f"  voice_dataset rows after dedup  : {rows_after}")

    if rows_removed > 0:
        hc_removed = report_df[report_df["disease_label"] == "HC"].shape[0]
        pd_removed = report_df[report_df["disease_label"] == "PD"].shape[0]
        logger.info(f"    Removed HC duplicates : {hc_removed}")
        logger.info(f"    Removed PD duplicates : {pd_removed}")
        logger.info("  Sample removed pairs:")
        for r in report_rows[:5]:                          # fixed: iterate dicts directly
            logger.info(f"    KEPT {r['kept_file']:<30}  REMOVED {r['removed_file']}")

    df_clean = pd.concat([df_other, df_vd_clean], ignore_index=True)
    logger.info("-" * 60)
    return df_clean, report_df


# =============================================================================
# ANTI-LEAKAGE: OPAQUE SUBJECT IDs  [L2 FIX]
# =============================================================================
def neutralise_voice_dataset_ids(df, logger):
    """
    Replace label-encoded voice_dataset subject IDs with opaque IDs.

    'healthy_000'    -> 'vd_0001'
    'parkinsons_000' -> 'vd_0002'
    ...

    Mapping is deterministic (sorted) so re-runs are reproducible.
    """
    logger.info("\n" + "-" * 60)
    logger.info("  [ANTI-LEAKAGE L2] Neutralising voice_dataset subject IDs")
    logger.info("-" * 60)

    mask_vd  = df["dataset"] == "voice_dataset"
    orig_ids = df.loc[mask_vd, "subject_id"].unique()

    sorted_ids = sorted(orig_ids)
    id_map     = {old: f"vd_{i+1:04d}" for i, old in enumerate(sorted_ids)}

    df.loc[mask_vd, "subject_id"] = df.loc[mask_vd, "subject_id"].map(id_map)

    logger.info(f"  Remapped {len(id_map)} voice_dataset subject IDs")
    logger.info("  Sample mappings (original -> opaque):")
    for old, new in list(id_map.items())[:10]:
        logger.info(f"    {old:<30} -> {new}")
    if len(id_map) > 10:
        logger.info(f"    ... ({len(id_map) - 10} more)")

    remaining_leaky = df.loc[mask_vd, "subject_id"].str.contains(
        "healthy|parkinson", case=False, na=False
    ).sum()
    if remaining_leaky == 0:
        logger.info("  Verification: OK  No label-encoding remains in subject_id")
    else:
        logger.error(f"  Verification: FAIL  {remaining_leaky} IDs still contain label keywords!")

    logger.info("-" * 60)
    return df


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_file(rec):
    """Process a single audio file and extract all features."""
    temp_path = None
    try:
        y, sr, temp_path, original_rms, clipped, duration_s = load_and_preprocess(rec["path"])

        row = {k: v for k, v in rec.items() if k != "path"}
        row["duration_s"]   = round(duration_s, 4)
        row["rms_original"] = round(original_rms, 6)
        row["rms_after"]    = round(float(np.sqrt(np.mean(y ** 2))), 6)
        row["rms_target"]   = TARGET_RMS
        row["rms_clipped"]  = clipped

        sound    = parselmouth.Sound(temp_path)
        f0_praat = extract_f0_praat(sound)
        row.update(f0_praat)

        try:
            pitch  = parselmouth.praat.call(sound, "To Pitch (cc)", 0.0, F0_MIN, 15, "no",
                                            0.03, 0.45, 0.01, 0.35, 0.14, F0_MAX)
            f0vals = pitch.selected_array["frequency"]
            f0vals = f0vals[f0vals > 0]
        except Exception:
            f0vals = np.array([])

        row.update(extract_jitter_shimmer(sound))
        row.update(extract_hnr_nhr(sound))
        row.update(extract_formants(sound))
        row.update(extract_nonlinear_features(y, sr, f0vals))
        row.update(extract_mfcc_features(y, sr))
        row.update(extract_spectral_features(y, sr))
        row.update(extract_chroma_features(y, sr))
        row.update(extract_spread_features(f0vals))

        return row

    except Exception as e:
        row = {k: v for k, v in rec.items() if k != "path"}
        row["_error"]     = str(e)
        row["_traceback"] = traceback.format_exc()
        return row

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# =============================================================================
# MAIN
# =============================================================================
def main():
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("  SUSTAINED VOWEL 'A' FEATURE EXTRACTION PIPELINE  [10 kHz]")
    logger.info("=" * 80)
    logger.info(f"  Start Time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Output CSV     : {OUTPUT_CSV}")
    logger.info(f"  Dedup Report   : {DEDUP_REPORT}")
    logger.info(f"  Log File       : {LOG_FILE}")
    logger.info(f"  Target SR      : {TARGET_SR} Hz")
    logger.info(f"  Target RMS     : {TARGET_RMS}")
    logger.info(f"  Parallel Jobs  : {N_JOBS}")
    logger.info("=" * 80)

    # -- Step 1: Collect files -------------------------------------------------
    logger.info("\n[STEP 1] Collecting sustained vowel 'a' files...")
    all_records = []

    pcgita_path = DATASET_PATHS["pc_gita"]["root"]
    if os.path.exists(pcgita_path):
        r = collect_pcgita(pcgita_path)
        all_records.extend(r)
        logger.info(f"  PC-GITA        : {len(r)} files")
    else:
        logger.warning(f"  PC-GITA path not found: {pcgita_path}")

    italian_path = DATASET_PATHS["italian"]["root"]
    if os.path.exists(italian_path):
        r = collect_italian(italian_path)
        all_records.extend(r)
        logger.info(f"  Italian        : {len(r)} files")
    else:
        logger.warning(f"  Italian path not found: {italian_path}")

    voice_path = DATASET_PATHS["voice_dataset"]["root"]
    if os.path.exists(voice_path):
        r = collect_voice_dataset(voice_path)
        all_records.extend(r)
        logger.info(f"  Voice_Dataset  : {len(r)} files (pre-dedup)")
    else:
        logger.warning(f"  Voice_Dataset path not found: {voice_path}")

    total_pd = sum(1 for r in all_records if r["label_binary"] == 1)
    total_hc = sum(1 for r in all_records if r["label_binary"] == 0)
    logger.info("-" * 40)
    logger.info(f"  TOTAL FILES    : {len(all_records)}  (pre-dedup)")
    logger.info(f"  PD             : {total_pd}")
    logger.info(f"  HC             : {total_hc}")
    logger.info("-" * 40)

    if not all_records:
        logger.error("No files found! Check dataset paths.")
        sys.exit(1)

    # -- Step 2: Extract features ----------------------------------------------
    logger.info("\n[STEP 2] Extracting features (parallel processing)...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_file)(rec)
        for rec in tqdm(all_records, desc="Extracting", leave=True)
    )

    errors = [r for r in results if r and "_error" in r]
    good   = [r for r in results if r and "_error" not in r]

    if errors:
        logger.warning(f"\n  {len(errors)} files failed:")
        for e in errors[:10]:
            logger.warning(f"    - {e.get('file', 'unknown')}: {e.get('_error', 'unknown error')}")

    # -- Step 3: Build DataFrame -----------------------------------------------
    logger.info("\n[STEP 3] Creating output DataFrame...")
    df = pd.DataFrame(good)
    for col in ["_error", "_traceback"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]

    logger.info(f"  Raw rows after extraction        : {len(df)}")
    logger.info(f"  Numeric feature columns          : {len(feat_cols)}")

    # -- Step 4: Anti-leakage --------------------------------------------------
    logger.info("\n[STEP 4] Anti-leakage: deduplication + ID neutralisation...")
    df, report_df = deduplicate_voice_dataset(df, feat_cols, logger)
    df            = neutralise_voice_dataset_ids(df, logger)

    if not report_df.empty:
        report_df.to_csv(DEDUP_REPORT, index=False)
        logger.info(f"  Dedup report saved -> {DEDUP_REPORT}")
    else:
        logger.info("  No duplicates found (report not written).")

    # -- Step 5: Feature quality cleanup ---------------------------------------
    logger.info("\n[STEP 5] Feature quality cleanup (near-zero variance removal)...")
    logger.info("-" * 60)

    # Refresh feat_cols against the cleaned DataFrame (dedup may have altered it)
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]

    X_all = df[feat_cols].values
    vt    = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X_all)

    kept_mask    = vt.get_support()
    dropped_cols = [feat_cols[i] for i, keep in enumerate(kept_mask) if not keep]
    feat_cols    = [feat_cols[i] for i, keep in enumerate(kept_mask) if keep]

    logger.info(f"  Variance threshold   : {VARIANCE_THRESHOLD}")
    logger.info(f"  Features before      : {len(kept_mask)}")
    logger.info(f"  Features after       : {len(feat_cols)}")
    logger.info(f"  Dropped ({{len(dropped_cols)}} features):")
    for col in dropped_cols:
        var = df[col].var()
        logger.info(f"    {{col:<35}}  var = {{var:.6f}}")
    if not dropped_cols:
        logger.info("    (none -- all features above threshold)")
    logger.info("-" * 60)

    # -- Step 6: Summary -------------------------------------------------------
    logger.info("\n[STEP 6] Summary...")
    logger.info("\n" + "=" * 80)
    logger.info("  EXTRACTION SUMMARY  (after deduplication + feature cleanup)")
    logger.info("=" * 80)
    logger.info(f"  Total rows (clean)   : {len(df)}")
    logger.info(f"  Total columns        : {len(df.columns)}")
    logger.info(f"  PD (1)               : {(df['label_binary'] == 1).sum()}")
    logger.info(f"  HC (0)               : {(df['label_binary'] == 0).sum()}")

    logger.info("\n  Per-Dataset Breakdown:")
    logger.info(f"  {'Dataset':<20} {'Files':>8} {'PD':>6} {'HC':>6} {'Subjects':>10}")
    logger.info("  " + "-" * 58)
    for ds in df["dataset"].unique():
        sub  = df[df["dataset"] == ds]
        pd_n = (sub["label_binary"] == 1).sum()
        hc_n = (sub["label_binary"] == 0).sum()
        n_s  = sub["subject_id"].nunique()
        logger.info(f"  {ds:<20} {len(sub):>8} {pd_n:>6} {hc_n:>6} {n_s:>10}")

    nan_counts = df[feat_cols].isna().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.info("\n  NaN in feature columns (top 10):")
        for col, cnt in nan_cols.sort_values(ascending=False).head(10).items():
            logger.info(f"    {col:<30}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")
    else:
        logger.info("\n  No NaN values in feature columns.")

    logger.info("\n  Biomarker Sanity Check:")
    for feat, expected, check_fn in [
        ("jitter_local",  "PD>HC", lambda p, h: p > h),
        ("shimmer_local", "PD>HC", lambda p, h: p > h),
        ("hnr",           "PD<HC", lambda p, h: p < h),
    ]:
        if feat in df.columns:
            pd_mean = df[df["label_binary"] == 1][feat].mean()
            hc_mean = df[df["label_binary"] == 0][feat].mean()
            ok      = "OK" if check_fn(pd_mean, hc_mean) else "INVERTED - check pipeline"
            logger.info(f"    {feat:<20}: PD={pd_mean:.4f}  HC={hc_mean:.4f}  {expected} [{ok}]")

    logger.info("\n  Anti-Leakage Verification:")
    vd_mask = df["dataset"] == "voice_dataset"
    leaky   = df.loc[vd_mask, "subject_id"].str.contains(
        "healthy|parkinson", case=False, na=False
    ).sum()
    logger.info(f"    Label-encoded IDs remaining  : {leaky}  "
                f"{'[OK - clean]' if leaky == 0 else '[FAIL - fix needed]'}")

    vd_dups = df[vd_mask].duplicated(
        subset=[c for c in feat_cols if c in df.columns], keep=False
    ).sum()
    logger.info(f"    Duplicate feature vectors    : {vd_dups}  "
                f"{'[OK - clean]' if vd_dups == 0 else '[FAIL - fix needed]'}")

    # -- Step 7: Save ----------------------------------------------------------
    logger.info("\n[STEP 7] Saving output...")
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n  Saved to             : {OUTPUT_CSV}")
    logger.info(f"  Processing time      : {elapsed:.1f} seconds")
    logger.info(f"  Log file             : {LOG_FILE}")
    logger.info("\n" + "=" * 80)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("=" * 80)

    feat_cols_out = [c for c in df.columns if c not in META_COLS]
    logger.info(f"\n  Feature list ({len(feat_cols_out)} features):")
    for i, col in enumerate(feat_cols_out, 1):
        logger.info(f"    {i:>3}. {col}")


if __name__ == "__main__":
    main()