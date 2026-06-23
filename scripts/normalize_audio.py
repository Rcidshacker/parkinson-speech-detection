"""
Loudness Normalization — PC-GITA Vowels
========================================
Normalizes all PC-GITA vowel WAV files to -23 LUFS (EBU R128 standard)
INPUT  : data-20260315T041334Z-1-002/data/raw/PC-GITA/Vowels/
OUTPUT : data/processed/PC-GITA-normalized/  (same folder structure)

WHY: PC-GITA recordings vary in amplitude across sessions.
     Shimmer, energy, and spectral features are all affected by loudness.
     EBU R128 normalization ensures consistent perceived loudness.
"""

import os
import subprocess
import glob
from pathlib import Path
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
INPUT_ROOT  = os.path.join(BASE, "data", "active", "pc_gita", "Vowels")
OUTPUT_ROOT = os.path.join(BASE, "data", "processed", "pc_gita_normalized", "Vowels")
TARGET_LUFS = -23
LOG_FILE    = os.path.join(BASE, "logs", f"normalize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def normalize_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg-normalize", input_path,
        "-o", output_path,
        "-nt", "ebu",
        "-t", str(TARGET_LUFS),
        "-ar", "16000",   # resample to 16kHz (standard for speech)
        "-c:a", "pcm_s16le",  # 16-bit PCM WAV
        "-f",  # force overwrite
        "--quiet"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr

def main():
    # Find all WAV files
    wav_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.wav"), recursive=True)
    total = len(wav_files)

    with open(LOG_FILE, "w") as f:
        log(f"Loudness Normalization — PC-GITA Vowels", f)
        log(f"Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"Input  : {INPUT_ROOT}", f)
        log(f"Output : {OUTPUT_ROOT}", f)
        log(f"Target : {TARGET_LUFS} LUFS (EBU R128)", f)
        log(f"Resample: 16000 Hz", f)
        log(f"Total files: {total}", f)
        log("=" * 60, f)

        ok_count   = 0
        fail_count = 0
        failed     = []

        for i, input_path in enumerate(sorted(wav_files), 1):
            # Reconstruct output path preserving subfolder structure
            rel_path    = os.path.relpath(input_path, INPUT_ROOT)
            output_path = os.path.join(OUTPUT_ROOT, rel_path)

            success, err = normalize_file(input_path, output_path)

            status = "OK" if success else "FAIL"
            msg    = f"[{i:4d}/{total}] {status}  {rel_path}"
            log(msg, f)

            if success:
                ok_count += 1
            else:
                fail_count += 1
                failed.append((rel_path, err))

        log("=" * 60, f)
        log(f"Done — {ok_count} OK  |  {fail_count} FAILED", f)
        log(f"Output folder: {OUTPUT_ROOT}", f)
        log(f"Log: {LOG_FILE}", f)

        if failed:
            log("\nFailed files:", f)
            for path, err in failed:
                log(f"  {path}: {err[:100]}", f)

        log(f"\nNext step:", f)
        log(f"  Update INPUT path in extract_features_all.py to:", f)
        log(f"  {OUTPUT_ROOT}", f)

if __name__ == "__main__":
    main()
