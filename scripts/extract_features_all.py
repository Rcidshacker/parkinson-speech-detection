"""
Master Feature Extraction Pipeline — v4
========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

CHANGES FROM v3 → v4 (March 19, 2026):
────────────────────────────────────────
  PC-GITA source    : REVERTED to data/active/pc_gita (original).
                      Why : v3 switched to data-20260315.../processed/PC-GITA/Vowels,
                            which turned out to be a quality-filtered subset with only
                            563 files and a severe class imbalance (391 PD vs 172 HC,
                            70/30). hc/U folder didn't even exist. HNR gap collapsed
                            to 0.08 dB. Original folder has 1,322 files, balanced
                            646 PD / 676 HC, all biomarkers correct.

  Min-duration guard: Added MIN_DURATION = 0.5s check in load_and_preprocess().
                      Any file shorter than 0.5s after silence trim raises ValueError
                      → logged as error, skipped cleanly.
                      Why : Original PC-GITA had 4 files under 0.5s and 129 under 1s.
                            The sub-0.5s files cause Praat PointProcess to fail or
                            return garbage jitter/shimmer values. Files 0.5–1.0s are
                            borderline but Praat can still extract meaningful values
                            at 150Hz (75+ pitch cycles). Keeping them with a warning.

  collect_pcgita()  : Reverted to v2 PCGITA_TASK_MAP + _pcgita_speech_type() logic.
                      vowel_letter and attempt_num columns kept as NaN for PC-GITA
                      since original folder doesn't have per-vowel subfolders named.
                      (If NeuroVoz is added later, vowel_letter can be populated then.)

  UNINA Italy       : Still removed (irrecoverable, documented).
  MDVR-KCL          : Still removed (no sustained vowel task).

DATASETS (v4):
    PC-GITA (Spanish) : ~1,318 sustained vowel files (original, after <0.5s drop)
    VOICED  (Italian) :   ~296 sustained vowel files
    TOTAL             : ~1,614 files  |  646 PD + 676 HC (balanced)

FEATURE GROUPS (unchanged from v2/v3):
    Praat CC F0     : mean, std, min, max, range, median     (6)
    pYIN F0         : mean, std, min, max, range, median     (6)
    Jitter          : local, rap, ppq5, ddp                  (4)
    Shimmer         : local, apq3, apq5, apq11, dda          (5)
    HNR / NHR       : hnr, nhr                              (2)
    RPDE/DFA/PPE    : NaN (not computable from audio)        (3)
    MFCCs (13)      : mean + std per coefficient            (26)
    Delta MFCCs     : mean per coefficient (width=9)        (13)
    Delta-Delta     : mean per coefficient (width=9)        (13)
    Log Energy      : mean, std                              (2)
    Spectral        : centroid, bandwidth, rolloff, flux    (7)
    ZCR             : mean, std                              (2)
    Mel             : mean, std                              (2)
    Chroma          : mean + std per pitch class            (24)
    Duration        : duration_s only                        (1)
    RMS tracking    : original, after, target, clipped       (4)
    TOTAL           : ~120 features

PREPROCESSING ORDER (supervisor confirmed March 18, unchanged):
    1. Load WAV at original sample rate
    2. Convert to mono
    3. Resample to 8,000 Hz  FIRST
    4. Trim leading/trailing silence (top_db=25)
    5. Reject if duration < 0.5s  ← NEW
    6. RMS normalize to 0.04  SECOND
    7. Save temp WAV for Praat
    8. Extract features
    9. Delete temp WAV

HOW TO RUN (from project root):
    venv\\Scripts\\activate
    python scripts\\extract_features_all.py

ESTIMATED TIME (~20 minutes total):
    VOICED  :  ~2 minutes
    PC-GITA :  ~18 minutes  (1,318 files, 44.1kHz → 8kHz resampling)
"""

import os
import re
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
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

PATHS = {
    # Original PC-GITA — 44.1kHz, 1,322 sustained vowel files, balanced 646PD/676HC
    "pc_gita": os.path.join(BASE, "data", "active", "pc_gita"),
    # VOICED — flat folder, PT prefix = PD, all others = HC
    "voiced":  os.path.join(BASE, "data", "active", "voiced", "Audio Files"),
}

OUTPUT_CSV = os.path.join(BASE, "features", "features_extracted_sv.csv")
LOG_DIR    = os.path.join(BASE, "logs")

TARGET_SR   = 8000
TARGET_RMS  = 0.04
SILENCE_DB  = 25
N_JOBS      = -1

N_MFCC      = 13
N_FFT       = 256     # 32ms at 8kHz
HOP_LENGTH  = 128     # 50% overlap
DELTA_WIDTH = 9

F0_MIN      = 80
F0_MAX      = 300
MIN_DURATION = 0.5    # seconds — files shorter than this after trim are skipped


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"extraction_sv_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("sv_extractor_v4")


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
def rms_normalize(y, target_rms=TARGET_RMS):
    """
    Normalize to target RMS. Automatically prevents clipping.
    Returns (y_normalized, original_rms, clipped_sample_count).
    """
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
    """
    Preprocessing pipeline (supervisor order, unchanged from v2):
        1. Load + mono + resample to 8kHz
        2. Trim silence
        3. RMS normalize to 0.04
        4. Save temp WAV for Praat
    Returns (y, sr, temp_path, original_rms, clipped, duration_s)
    """
    y, sr     = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=SILENCE_DB)
    if len(y_trim) >= TARGET_SR * 0.5:
        y = y_trim

    duration_s = len(y) / TARGET_SR

    # Guard: reject files too short for reliable Praat extraction
    if duration_s < MIN_DURATION:
        raise ValueError(
            f"Too short after silence trim: {duration_s:.3f}s < {MIN_DURATION}s — skipped"
        )

    y, original_rms, clipped = rms_normalize(y)

    tmp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    sf.write(temp_path, y, TARGET_SR, subtype="PCM_16")

    return y, sr, temp_path, original_rms, clipped, duration_s


# ══════════════════════════════════════════════════════════════
# F0 EXTRACTION — DUAL: Praat CC + pYIN (unchanged from v2)
# ══════════════════════════════════════════════════════════════
def extract_f0_praat(sound):
    """Praat Cross-Correlation F0 — recommended for pathological voices."""
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
    """pYIN (probabilistic YIN) — kept for comparison, expected high NaN rate."""
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


# ══════════════════════════════════════════════════════════════
# PRAAT VOICE QUALITY FEATURES (unchanged from v2)
# ══════════════════════════════════════════════════════════════
def extract_praat_voice_quality(temp_wav_path):
    """Jitter, Shimmer, HNR via Praat PointProcess."""
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

        # Nonlinear dynamics — not computable from raw audio, kept as NaN
        # to maintain column consistency with UCI/Sakar feature-only datasets
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


# ══════════════════════════════════════════════════════════════
# LIBROSA FEATURES (unchanged from v2)
# n_fft=256 (32ms at 8kHz), n_mfcc=13, hop_length=128
# ══════════════════════════════════════════════════════════════
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
# PROCESS SINGLE FILE (unchanged from v2)
# ══════════════════════════════════════════════════════════════
def process_file(rec):
    """Extract all features for one file. Returns flat dict row."""
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
# DATASET COLLECTORS
# ══════════════════════════════════════════════════════════════

# ── PC-GITA (v4 — reverted to original, with PCGITA_TASK_MAP) ────
PCGITA_TASK_MAP = {
    "ddk analysis":    "ddk",
    "modulated vowels":"modulated_vowel",
    "monologue":       "monologue",
    "read text":       "read_text",
    "sentences":       "sentences",
    "sentences2":      "sentences",
    "vowels":          "sustained_vowel",
    "words":           "words",
}
PCGITA_SPEAKER_RE = re.compile(r"(AVPEPUDEA[C]?\d{4})", re.IGNORECASE)


def _pcgita_label(wav_path):
    for part in wav_path.replace("\\", "/").split("/"):
        p = part.lower()
        if p in ("pd", "patologica") or p.startswith("pd_"): return 1, "PD"
        if p in ("hc", "control")   or p.startswith("hc_"): return 0, "HC"
    return None, None


def _pcgita_speech_type(wav_path):
    """Identify task from the subfolder immediately under audio_pd_gita_co."""
    parts = wav_path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.lower() == "audio_pd_gita_co" and i + 1 < len(parts):
            return PCGITA_TASK_MAP.get(parts[i + 1].lower(), "unknown")
    return "unknown"


def collect_pcgita(root):
    """
    Collect PC-GITA sustained vowel records from original folder.
    Structure: audio_pd_gita_co/vowels/{pd,hc}/AVPEPUDEA..._.wav
    Only files in the vowels/ task subfolder are collected.
    """
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(dirpath, fname)
            if _pcgita_speech_type(wav_path) != "sustained_vowel":
                continue
            label, dlabel = _pcgita_label(wav_path)
            if label is None:
                continue
            m = PCGITA_SPEAKER_RE.search(fname)
            records.append({
                "path":             wav_path,
                "dataset":          "pc_gita",
                "subject_id":       m.group(1).upper() if m else "UNKNOWN",
                "language":         "es",
                "gender":           float("nan"),
                "speech_type":      "sustained_vowel",
                "vowel_letter":     float("nan"),
                "attempt_num":      float("nan"),
                "disease_label":    dlabel,
                "label_binary":     label,
                "multiclass_label": float("nan"),
                "updrs_total":      float("nan"),
                "moca_score":       float("nan"),
                "meds_status":      float("nan"),
                "file":             fname,
            })
    return records


# ── VOICED (unchanged from v2) ────────────────────────────────
def collect_voiced(root):
    """
    VOICED dataset — flat folder, all files are sustained vowels.
    PT prefix = PD patient, others = HC.
    """
    records = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(".wav"):
            continue
        label = 1 if fname.upper().startswith("PT") else 0
        records.append({
            "path":             os.path.join(root, fname),
            "dataset":          "voiced",
            "subject_id":       os.path.splitext(fname)[0],
            "language":         "it",
            "gender":           float("nan"),
            "speech_type":      "sustained_vowel",
            "vowel_letter":     float("nan"),   # VOICED has no vowel breakdown
            "attempt_num":      float("nan"),
            "disease_label":    "PD" if label == 1 else "HC",
            "label_binary":     label,
            "multiclass_label": float("nan"),
            "updrs_total":      float("nan"),
            "moca_score":       float("nan"),
            "meds_status":      float("nan"),
            "file":             fname,
        })
    return records


# ══════════════════════════════════════════════════════════════
# EXTRACT ONE DATASET
# ══════════════════════════════════════════════════════════════
def extract_dataset(name, records):
    if not records:
        logger.info(f"  [{name}] No records — skipped")
        return pd.DataFrame()

    pd_count = sum(1 for r in records if r["label_binary"] == 1)
    hc_count = sum(1 for r in records if r["label_binary"] == 0)
    logger.info(f"  [{name}] {len(records)} files  PD={pd_count}  HC={hc_count}")

    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(process_file)(rec)
        for rec in tqdm(records, desc=f"{name}", leave=True)
    )

    errors = [r for r in results if r and "_error" in r]
    good   = [r for r in results if r and "_error" not in r]

    if errors:
        logger.warning(f"  [{name}] {len(errors)} files failed:")
        for e in errors[:5]:
            logger.warning(f"    {e.get('file','?')}: {e.get('_error','?')}")

    df = pd.DataFrame(good)
    if "_error" in df.columns:
        df = df.drop(columns=["_error"])

    logger.info(f"  [{name}] Done — {len(df)} rows extracted")
    return df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 65)
    logger.info("  Feature Extraction — Sustained Vowel Only (v4)")
    logger.info(f"  PC-GITA source : original data/active/pc_gita")
    logger.info(f"  UNINA          : REMOVED (irrecoverable biomarker inversion)")
    logger.info(f"  Preprocessing  : 8kHz → silence trim → RMS={TARGET_RMS}")
    logger.info(f"  F0             : Praat CC + pYIN")
    logger.info(f"  F0 range       : {F0_MIN}–{F0_MAX} Hz")
    logger.info(f"  n_mfcc         : {N_MFCC}")
    logger.info(f"  n_fft          : {N_FFT} ({N_FFT/TARGET_SR*1000:.0f}ms at {TARGET_SR}Hz)")
    logger.info(f"  hop_length     : {HOP_LENGTH}")
    logger.info(f"  Output         : {OUTPUT_CSV}")
    logger.info(f"  Start          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # Verify paths exist before starting
    missing = [f"{name}: {path}" for name, path in PATHS.items()
               if not os.path.exists(path)]
    if missing:
        for m in missing:
            logger.error(f"  Path not found: {m}")
        sys.exit(1)

    # ── Step 1: Collect ──────────────────────────────────────
    logger.info("\n[Step 1] Collecting sustained vowel records...")

    collectors = {
        "voiced":  (collect_voiced,  PATHS["voiced"]),
        "pc_gita": (collect_pcgita,  PATHS["pc_gita"]),
    }

    all_records = {}
    total = 0
    for name, (fn, path) in collectors.items():
        recs = fn(path)
        all_records[name] = recs
        total += len(recs)
        logger.info(f"  {name:<15} {len(recs):>5} files")
    logger.info(f"  {'TOTAL':<15} {total:>5} files")

    # PC-GITA subject count sanity
    if all_records.get("pc_gita"):
        pc_recs = all_records["pc_gita"]
        pd_subs = len(set(r["subject_id"] for r in pc_recs if r["label_binary"]==1))
        hc_subs = len(set(r["subject_id"] for r in pc_recs if r["label_binary"]==0))
        logger.info(f"  PC-GITA subjects: PD={pd_subs}  HC={hc_subs}  (expect 50+50)")

    # ── Step 2: Extract ──────────────────────────────────────
    logger.info("\n[Step 2] Extracting features...")
    all_dfs = []

    for name, (fn, path) in collectors.items():
        logger.info(f"\n── {name} ──")
        df = extract_dataset(name, all_records[name])
        if len(df) > 0:
            all_dfs.append(df)
            interim = OUTPUT_CSV.replace(".csv", f"_{name}.csv")
            df.to_csv(interim, index=False)
            logger.info(f"  Interim saved → {interim}")

    # ── Step 3: Combine ──────────────────────────────────────
    logger.info("\n[Step 3] Combining datasets...")
    if not all_dfs:
        logger.error("No data extracted. Check paths and folder structure.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Step 4: Final Summary ────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  FINAL SUMMARY — v3")
    logger.info("=" * 65)
    logger.info(f"  Total rows : {len(combined)}")
    logger.info(f"  Columns    : {len(combined.columns)}")
    logger.info(f"  PD (1)     : {(combined['label_binary']==1).sum()}")
    logger.info(f"  HC (0)     : {(combined['label_binary']==0).sum()}")

    logger.info(f"\n  {'Dataset':<15}  {'Rows':>5}  {'PD':>5}  {'HC':>5}  Lang")
    logger.info("  " + "─" * 42)
    for ds in combined["dataset"].unique():
        sub = combined[combined["dataset"] == ds]
        logger.info(f"  {ds:<15}  {len(sub):>5}  "
                    f"{(sub['label_binary']==1).sum():>5}  "
                    f"{(sub['label_binary']==0).sum():>5}  "
                    f"{sub['language'].iloc[0]}")

    # Duration sanity
    logger.info("\n  ── Duration (seconds) ──")
    for ds in combined["dataset"].unique():
        sub = combined[combined["dataset"] == ds]
        d   = sub["duration_s"]
        short = (d < 1.0).sum()
        logger.info(f"  {ds:<15}  mean={d.mean():.2f}s  "
                    f"min={d.min():.2f}s  max={d.max():.2f}s  "
                    f"<1s={short}")

    # RMS sanity
    logger.info("\n  ── RMS ──")
    for ds in combined["dataset"].unique():
        sub = combined[combined["dataset"] == ds]
        logger.info(f"  {ds:<15}  "
                    f"before={sub['rms_original'].mean():.4f}  "
                    f"after={sub['rms_after'].mean():.4f}  "
                    f"clipped={int(sub['rms_clipped'].sum())}")

    # Biomarker sanity
    logger.info("\n  ── Biomarker Sanity (Praat CC) ──")
    biomarkers = [
        ("jitter_local",  "PD>HC", lambda p, h: p > h),
        ("shimmer_local", "PD>HC", lambda p, h: p > h),
        ("hnr",           "PD<HC", lambda p, h: p < h),
        ("praat_f0_mean", "info",  None),
        ("praat_f0_std",  "info",  None),
    ]
    for feat, exp, check_fn in biomarkers:
        if feat not in combined.columns:
            continue
        pd_m = combined[combined["label_binary"]==1][feat].mean()
        hc_m = combined[combined["label_binary"]==0][feat].mean()
        if check_fn is not None:
            ok = "✓" if check_fn(pd_m, hc_m) else "✗ INVERTED"
        else:
            ok = ""
        logger.info(f"  {feat:<22}  PD={pd_m:.4f}  HC={hc_m:.4f}  {exp} {ok}")

    # Per-dataset biomarker check
    logger.info("\n  ── Per-Dataset Biomarker ──")
    for ds in combined["dataset"].unique():
        sub  = combined[combined["dataset"] == ds]
        pd_j = sub[sub["label_binary"]==1]["jitter_local"].mean()
        hc_j = sub[sub["label_binary"]==0]["jitter_local"].mean()
        pd_h = sub[sub["label_binary"]==1]["hnr"].mean()
        hc_h = sub[sub["label_binary"]==0]["hnr"].mean()
        j_ok = "✓" if pd_j > hc_j else "✗"
        h_ok = "✓" if pd_h < hc_h else "✗"
        logger.info(f"  {ds:<15}  jitter {j_ok} (PD={pd_j:.4f} HC={hc_j:.4f})  "
                    f"hnr {h_ok} (PD={pd_h:.2f} HC={hc_h:.2f})")

    # PC-GITA per-vowel biomarker breakdown
    if "pc_gita" in combined["dataset"].values and "vowel_letter" in combined.columns:
        logger.info("\n  ── PC-GITA per-vowel jitter sanity ──")
        pc = combined[combined["dataset"] == "pc_gita"]
        for vl in sorted(pc["vowel_letter"].dropna().unique()):
            sub  = pc[pc["vowel_letter"] == vl]
            pd_j = sub[sub["label_binary"]==1]["jitter_local"].mean()
            hc_j = sub[sub["label_binary"]==0]["jitter_local"].mean()
            ok   = "✓" if pd_j > hc_j else "✗"
            logger.info(f"    /{vl.lower()}/  PD={pd_j:.4f}  HC={hc_j:.4f}  {ok}  "
                        f"(n={len(sub)})")

    # pYIN reliability
    logger.info("\n  ── F0 Reliability (NaN count) ──")
    for algo in ["praat", "pyin"]:
        col = f"{algo}_f0_mean"
        if col in combined.columns:
            nan_n = combined[col].isna().sum()
            logger.info(f"  {algo:<8}  NaN={nan_n}/{len(combined)} "
                        f"({nan_n/len(combined)*100:.1f}%)")

    # NaN summary
    meta_cols = {"dataset","subject_id","language","gender","speech_type",
                 "vowel_letter","attempt_num","disease_label","label_binary",
                 "multiclass_label","updrs_total","moca_score","meds_status",
                 "file","rms_original","rms_after","rms_target","rms_clipped",
                 "duration_s"}
    feat_cols = [c for c in combined.columns if c not in meta_cols]
    nan_feats = {c: int(combined[c].isna().sum())
                 for c in feat_cols if combined[c].isna().sum() > 0}
    if nan_feats:
        logger.info("\n  ── NaN in feature columns ──")
        for col, cnt in sorted(nan_feats.items(), key=lambda x: -x[1])[:15]:
            logger.info(f"  {col:<30}: {cnt}/{len(combined)} "
                        f"({cnt/len(combined)*100:.1f}%)")

    combined.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\n  Saved → {OUTPUT_CSV}")
    logger.info(f"  End   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)
    logger.info("")
    logger.info("  Next steps:")
    logger.info("  1. Verify biomarkers all ✓ per dataset")
    logger.info("  2. Check duration_s — no files < 1s expected in PC-GITA v3")
    logger.info("  3. Check pYIN NaN rate — should improve vs v2 (cleaner files)")
    logger.info("  4. Paste log summary to Claude for sign-off")
    logger.info("  5. Then run downstream: prepare_features.py → train_models.py")


if __name__ == "__main__":
    main()
