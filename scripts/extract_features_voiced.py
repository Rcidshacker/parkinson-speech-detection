"""
VOICED Standalone Feature Extraction
======================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

PURPOSE:
    Standalone extraction for VOICED (Italian) dataset only.
    Produces features/features_voiced_standalone.csv

    Use this when:
    - You want to re-extract VOICED without touching PC-GITA
    - Adding or verifying VOICED data independently
    - Feeding into per_dataset_ranking.py for individual feature ranking

DATASET:
    VOICED (Italian) : ~296 sustained vowel files (flat folder)
    Source           : data/active/voiced/Audio Files
    Labels           : PT prefix = PD, all others = HC

PREPROCESSING (supervisor confirmed, unchanged):
    1. Load WAV at original sample rate
    2. Convert to mono
    3. Resample to 8,000 Hz
    4. Trim leading/trailing silence (top_db=25)
    5. Reject if duration < 0.5s
    6. RMS normalize to 0.04
    7. Save temp WAV for Praat
    8. Extract features
    9. Delete temp WAV

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\extract_features_voiced.py

OUTPUT:
    features/features_voiced_standalone.csv
    logs/extraction_voiced_<timestamp>.log
"""

import os
import sys
import tempfile
import warnings
import logging
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm
from datetime import datetime
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE       = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
DATA_PATH  = os.path.join(BASE, "data", "active", "voiced", "Audio Files")
OUTPUT_CSV = os.path.join(BASE, "features", "features_voiced_standalone.csv")
LOG_DIR    = os.path.join(BASE, "logs")

TARGET_SR    = 8000
TARGET_RMS   = 0.04
SILENCE_DB   = 25
N_JOBS       = -1
N_MFCC       = 13
N_FFT        = 256
HOP_LENGTH   = 128
DELTA_WIDTH  = 9
F0_MIN       = 80
F0_MAX       = 300
MIN_DURATION = 0.5


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"extraction_voiced_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("voiced_extractor")


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
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
    y, sr     = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=SILENCE_DB)
    if len(y_trim) >= TARGET_SR * 0.5:
        y = y_trim

    duration_s = len(y) / TARGET_SR
    if duration_s < MIN_DURATION:
        raise ValueError(
            f"Too short after trim: {duration_s:.3f}s < {MIN_DURATION}s — skipped"
        )

    y, original_rms, clipped = rms_normalize(y)

    tmp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    sf.write(temp_path, y, TARGET_SR, subtype="PCM_16")

    return y, sr, temp_path, original_rms, clipped, duration_s


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_f0_praat(sound):
    feats = {}
    try:
        pitch  = call(sound, "To Pitch (cc)", 0.0, F0_MIN, 15,
                      "no", 0.03, 0.45, 0.01, 0.35, 0.14, F0_MAX)
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
            for k in ["praat_f0_mean","praat_f0_std","praat_f0_min",
                      "praat_f0_max","praat_f0_range","praat_f0_median"]:
                feats[k] = float("nan")
    except Exception:
        for k in ["praat_f0_mean","praat_f0_std","praat_f0_min",
                  "praat_f0_max","praat_f0_range","praat_f0_median"]:
            feats[k] = float("nan")
    return feats


def extract_f0_pyin(y, sr):
    feats = {}
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=float(F0_MIN), fmax=float(F0_MAX),
            sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH,
        )
        f0_voiced = f0[voiced_flag & (voiced_prob > 0.5)]
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        if len(f0_voiced) > 5:
            feats["pyin_f0_mean"]   = float(np.mean(f0_voiced))
            feats["pyin_f0_std"]    = float(np.std(f0_voiced))
            feats["pyin_f0_min"]    = float(np.min(f0_voiced))
            feats["pyin_f0_max"]    = float(np.max(f0_voiced))
            feats["pyin_f0_range"]  = float(np.max(f0_voiced) - np.min(f0_voiced))
            feats["pyin_f0_median"] = float(np.median(f0_voiced))
        else:
            for k in ["pyin_f0_mean","pyin_f0_std","pyin_f0_min",
                      "pyin_f0_max","pyin_f0_range","pyin_f0_median"]:
                feats[k] = float("nan")
    except Exception:
        for k in ["pyin_f0_mean","pyin_f0_std","pyin_f0_min",
                  "pyin_f0_max","pyin_f0_range","pyin_f0_median"]:
            feats[k] = float("nan")
    return feats


def extract_praat_voice_quality(temp_wav_path):
    feats = {}
    try:
        sound = parselmouth.Sound(temp_wav_path)
        feats.update(extract_f0_praat(sound))

        pp = call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)

        feats["jitter_local"] = call(pp, "Get jitter (local)",
                                     0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        feats["jitter_rap"]   = call(pp, "Get jitter (rap)",
                                     0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        feats["jitter_ppq5"]  = call(pp, "Get jitter (ppq5)",
                                     0, 0, 1/F0_MAX, 1/F0_MIN, 1.3)
        feats["jitter_ddp"]   = feats["jitter_rap"] * 3

        feats["shimmer_local"] = call([sound, pp], "Get shimmer (local)",
                                      0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq3"]  = call([sound, pp], "Get shimmer (apq3)",
                                      0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq5"]  = call([sound, pp], "Get shimmer (apq5)",
                                      0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_apq11"] = call([sound, pp], "Get shimmer (apq11)",
                                      0, 0, 1/F0_MAX, 1/F0_MIN, 1.3, 1.6)
        feats["shimmer_dda"]   = feats["shimmer_apq3"] * 3

        harmonicity  = call(sound, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
        feats["hnr"] = float(call(harmonicity, "Get mean", 0, 0))
        feats["nhr"] = 1.0 / (feats["hnr"] + 1e-6)

        feats["rpde"] = float("nan")
        feats["dfa"]  = float("nan")
        feats["ppe"]  = float("nan")

    except Exception:
        for col in ["praat_f0_mean","praat_f0_std","praat_f0_min","praat_f0_max",
                    "praat_f0_range","praat_f0_median",
                    "jitter_local","jitter_rap","jitter_ppq5","jitter_ddp",
                    "shimmer_local","shimmer_apq3","shimmer_apq5","shimmer_apq11",
                    "shimmer_dda","hnr","nhr","rpde","dfa","ppe"]:
            feats[col] = float("nan")
    return feats


def extract_librosa_features(y, sr):
    feats = {}
    try:
        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta  = librosa.feature.delta(mfccs, width=DELTA_WIDTH)
        delta2 = librosa.feature.delta(mfccs, order=2, width=DELTA_WIDTH)

        for i in range(N_MFCC):
            n = f"{i+1:02d}"
            feats[f"mfcc_{n}_mean"]   = float(np.mean(mfccs[i]))
            feats[f"mfcc_{n}_std"]    = float(np.std(mfccs[i]))
            feats[f"dmfcc_{n}_mean"]  = float(np.mean(delta[i]))
            feats[f"d2mfcc_{n}_mean"] = float(np.mean(delta2[i]))

        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        feats["log_energy_mean"] = float(np.mean(np.log(rms + 1e-10)))
        feats["log_energy_std"]  = float(np.std(np.log(rms + 1e-10)))

        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr,
                        n_fft=N_FFT, hop_length=HOP_LENGTH)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                        n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr,
                        n_fft=N_FFT, hop_length=HOP_LENGTH)
        flux      = librosa.onset.onset_strength(y=y, sr=sr,
                        n_fft=N_FFT, hop_length=HOP_LENGTH)

        feats["spectral_centroid_mean"]  = float(np.mean(centroid))
        feats["spectral_centroid_std"]   = float(np.std(centroid))
        feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        feats["spectral_bandwidth_std"]  = float(np.std(bandwidth))
        feats["spectral_rolloff_mean"]   = float(np.mean(rolloff))
        feats["spectral_flux_mean"]      = float(np.mean(flux))
        feats["spectral_flux_std"]       = float(np.std(flux))

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT,
                                                  hop_length=HOP_LENGTH)
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"]  = float(np.std(zcr))

        mel = librosa.feature.melspectrogram(y=y, sr=sr,
                  n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats["mel_mean"] = float(np.mean(mel))
        feats["mel_std"]  = float(np.std(mel))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(12):
            feats[f"chroma_{i:02d}_mean"] = float(np.mean(chroma[i]))
            feats[f"chroma_{i:02d}_std"]  = float(np.std(chroma[i]))

    except Exception:
        for i in range(N_MFCC):
            n = f"{i+1:02d}"
            for s in ["mfcc","dmfcc","d2mfcc"]:
                feats[f"{s}_{n}_mean"] = float("nan")
            feats[f"mfcc_{n}_std"] = float("nan")
        for col in ["log_energy_mean","log_energy_std",
                    "spectral_centroid_mean","spectral_centroid_std",
                    "spectral_bandwidth_mean","spectral_bandwidth_std",
                    "spectral_rolloff_mean","spectral_flux_mean","spectral_flux_std",
                    "zcr_mean","zcr_std","mel_mean","mel_std"]:
            feats[col] = float("nan")
        for i in range(12):
            feats[f"chroma_{i:02d}_mean"] = float("nan")
            feats[f"chroma_{i:02d}_std"]  = float("nan")
    return feats


# ══════════════════════════════════════════════════════════════
# VOICED COLLECTOR
# ══════════════════════════════════════════════════════════════
def collect_voiced(root):
    """
    VOICED — flat folder, all files are sustained vowels.
    PT prefix = PD patient, all others = HC.
    """
    records = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(".wav"):
            continue
        label = 1 if fname.upper().startswith("PT") else 0
        records.append({
            "path":          os.path.join(root, fname),
            "dataset":       "voiced",
            "subject_id":    os.path.splitext(fname)[0],
            "language":      "it",
            "gender":        float("nan"),
            "speech_type":   "sustained_vowel",
            "disease_label": "PD" if label == 1 else "HC",
            "label_binary":  label,
            "label":         label,
            "file":          fname,
        })
    return records


# ══════════════════════════════════════════════════════════════
# PROCESS SINGLE FILE
# ══════════════════════════════════════════════════════════════
def process_file(rec):
    temp_path = None
    try:
        y, sr, temp_path, original_rms, clipped, duration_s = \
            load_and_preprocess(rec["path"])

        row = {k: v for k, v in rec.items() if k != "path"}
        row["rms_original"] = round(original_rms, 6)
        row["rms_after"]    = round(float(np.sqrt(np.mean(y**2))), 6)
        row["rms_target"]   = TARGET_RMS
        row["rms_clipped"]  = clipped
        row["duration_s"]   = round(duration_s, 4)

        row.update(extract_f0_pyin(y, sr))
        row.update(extract_praat_voice_quality(temp_path))
        row.update(extract_librosa_features(y, sr))

        return row

    except Exception as e:
        row = {k: v for k, v in rec.items() if k != "path"}
        row["_error"] = str(e)
        return row
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 65)
    logger.info("  VOICED Standalone Feature Extraction")
    logger.info(f"  Source     : {DATA_PATH}")
    logger.info(f"  Output     : {OUTPUT_CSV}")
    logger.info(f"  Target SR  : {TARGET_SR} Hz")
    logger.info(f"  Target RMS : {TARGET_RMS}")
    logger.info(f"  Min dur    : {MIN_DURATION}s")
    logger.info(f"  Start      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    if not os.path.exists(DATA_PATH):
        logger.error(f"Data path not found: {DATA_PATH}")
        sys.exit(1)

    # Collect
    logger.info("\n[Step 1] Collecting VOICED records...")
    records  = collect_voiced(DATA_PATH)
    pd_count = sum(1 for r in records if r["label_binary"] == 1)
    hc_count = sum(1 for r in records if r["label_binary"] == 0)
    logger.info(f"  Files: {len(records)}  PD={pd_count}  HC={hc_count}")
    logger.info(f"  (expect ~148 PD, ~148 HC)")

    if not records:
        logger.error("No records found. Check DATA_PATH.")
        sys.exit(1)

    # Extract
    logger.info("\n[Step 2] Extracting features (parallel)...")
    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(process_file)(rec)
        for rec in tqdm(records, desc="VOICED", leave=True)
    )

    errors = [r for r in results if r and "_error" in r]
    good   = [r for r in results if r and "_error" not in r]

    if errors:
        logger.warning(f"  {len(errors)} files failed:")
        for e in errors[:5]:
            logger.warning(f"    {e.get('file','?')}: {e.get('_error','?')}")

    df = pd.DataFrame(good)
    if "_error" in df.columns:
        df = df.drop(columns=["_error"])

    # Summary
    logger.info("\n" + "=" * 65)
    logger.info("  SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Rows extracted : {len(df)}")
    logger.info(f"  Columns        : {len(df.columns)}")
    logger.info(f"  PD             : {(df['label_binary']==1).sum()}")
    logger.info(f"  HC             : {(df['label_binary']==0).sum()}")

    # Duration sanity
    d = df["duration_s"]
    logger.info(f"\n  Duration: mean={d.mean():.2f}s  min={d.min():.2f}s  "
                f"max={d.max():.2f}s  <1s={(d<1.0).sum()}")

    # Biomarker sanity
    logger.info("\n  ── Biomarker Sanity ──")
    for feat, exp, check_fn in [
        ("jitter_local",  "PD>HC", lambda p,h: p>h),
        ("shimmer_local", "PD>HC", lambda p,h: p>h),
        ("hnr",           "PD<HC", lambda p,h: p<h),
    ]:
        pd_m = df[df["label_binary"]==1][feat].mean()
        hc_m = df[df["label_binary"]==0][feat].mean()
        ok   = "✓" if check_fn(pd_m, hc_m) else "✗ INVERTED"
        logger.info(f"  {feat:<22} PD={pd_m:.4f}  HC={hc_m:.4f}  {exp} {ok}")

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n  Saved  → {OUTPUT_CSV}")
    logger.info(f"  Log    → {LOG_FILE}")
    logger.info(f"  End    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n  Next: run per_dataset_ranking.py")


if __name__ == "__main__":
    main()
