r"""
inspect_compare_prefixes.py
============================
Project : Speech-Based Parkinson's Disease Detection
Stage   : 2 — Feature Grouping
Purpose : Extract and group unique LLD prefixes from ComParE16 CSV
          to inform Stage 2 feature grouping strategy.

Usage (PowerShell):
    $PY  = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\venv\Scripts\python.exe"
    $S2  = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\stage_2_feature_grouping"
    & $PY "$S2\scripts\inspect_compare_prefixes.py"

OUTPUT:
    - Console : grouped prefix table with feature counts
    - File    : stage_2_feature_grouping/results/compare_prefix_report.txt
    - File    : stage_2_feature_grouping/results/compare_group_columns.txt

AUDIT FIXES APPLIED:
    [FIX-1] RESULTS_DIR now routes to stage_2_feature_grouping/results/
            (was incorrectly pointing to main project results/).
    [FIX-2] Removed `lsp` from Spectral regex — LSP (Line Spectral Pairs) is
            LPC-derived; having it in Spectral created a dead-code path in
            LSF/LPC rule since first-match-wins and Spectral comes first.
    [FIX-3] Added `linregerrA` to suffix_pattern — ComParE defines both
            linregerrA and linregerrQ functionals; missing A left dirty prefixes.
    [FIX-4] Removed unused `datetime` import.
    [FIX-5] Removed phantom LLD names (`msp`, `osil`) from Spectral regex —
            not standard openSMILE ComParE LLDs; kept regex semantically clean.
    [FIX-6] Updated docstring usage path to stage_2_feature_grouping/scripts/.
"""

import os
import sys
import re
import logging
from collections import defaultdict

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
STAGE2_BASE  = os.path.join(PROJECT_BASE, "stage_2_feature_grouping")

# Input: raw feature CSV lives in main project (shared source of truth)
CSV_PATH    = os.path.join(PROJECT_BASE, "features", "opensmile", "training_compare_8k_full.csv")

# Output: stage 2 owns its results  [FIX-1]
RESULTS_DIR = os.path.join(STAGE2_BASE, "results")
REPORT_FILE = os.path.join(RESULTS_DIR, "compare_prefix_report.txt")
EXPORT_FILE = os.path.join(RESULTS_DIR, "compare_group_columns.txt")

# ── Non-feature columns to skip ───────────────────────────────────────────────
META_COLS = {"subject_id", "label", "dataset", "file", "filename", "split"}

# ── Semantic group rules: (display_name, regex_pattern) ──────────────────────
# Order matters — first match wins.
# [FIX-2] `lsp` removed from Spectral; belongs in LSF/LPC (LPC-derived).
# [FIX-5] `msp`, `osil` removed from Spectral (non-standard openSMILE names).
SEMANTIC_RULES = [
    ("Voice Quality",   r"^(jitter|shimmer|HNR|logHNR|voicingProb|F0final|F0env)"),
    ("MFCC",            r"^mfcc"),
    ("Prosodic/Energy", r"^(pcm_loudness|pcm_RMSenergy|energy|loudness)"),
    ("Spectral",        r"^(pcm_fftMag|audspec|spectral|chroma)"),
    ("Temporal/ZCR",    r"^(pcm_zcr|voicedSegments|unvoicedSegments|duration|tempo)"),
    ("LSF/LPC",         r"^(lsf|lpc|lsp)"),
    ("Other",           r".*"),   # catch-all — must remain last
]

# ── Logging setup (makedirs first so FileHandler path exists) ─────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(REPORT_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("inspect_compare_prefixes")


def classify_feature(col_name: str) -> str:
    """Map a ComParE column name to a semantic group via first-match rule."""
    for group_name, pattern in SEMANTIC_RULES:
        if re.match(pattern, col_name, re.IGNORECASE):
            return group_name
    return "Other"


def extract_lld_prefix(col_name: str) -> str:
    """
    Strip the statistical functional suffix from an openSMILE ComParE column.

    ComParE naming pattern:  <LLD_name>_<functional>
        mfcc_sma[1]_amean       → mfcc_sma[1]
        F0final_sma_stddev      → F0final_sma
        shimmerLocal_sma_amean  → shimmerLocal_sma

    [FIX-3] Added linregerrA alongside linregerrQ (both are valid functionals).
    """
    suffix_pattern = (
        r"_("
        r"amean|stddev|skewness|kurtosis"
        r"|percentile\d+(\.\d+)?"
        r"|min|max|range"
        r"|upleveltime\d+"
        r"|rlowleveltime\d+"
        r"|quartile\d+"
        r"|iqr\d+(-\d+)?"
        r"|linregc\d+"
        r"|linregerrA|linregerrQ"
        r"|stddevFallingSlope|stddevRisingSlope"
        r"|meanFallingSlope|meanRisingSlope"
        r"|numPeaks|numSegments"
        r")$"
    )
    return re.sub(suffix_pattern, "", col_name, flags=re.IGNORECASE)


def main():
    logger.info(f"CSV input  : {CSV_PATH}")
    logger.info(f"Stage 2    : {STAGE2_BASE}")

    if not os.path.exists(CSV_PATH):
        logger.error(f"File not found: {CSV_PATH}")
        sys.exit(1)

    # Header-only read — avoids loading full 6373 × ~867 matrix into memory
    all_cols = list(pd.read_csv(CSV_PATH, nrows=0).columns)
    logger.info(f"Total columns   : {len(all_cols)}")

    feature_cols = [c for c in all_cols if c.lower() not in META_COLS]
    logger.info(f"Feature columns : {len(feature_cols)}")

    # ── Classify and count ────────────────────────────────────────────────────
    group_cols     = defaultdict(list)
    prefix_count   = defaultdict(int)
    group_prefixes = defaultdict(set)

    for col in feature_cols:
        group  = classify_feature(col)
        prefix = extract_lld_prefix(col)
        group_cols[group].append(col)
        prefix_count[prefix] += 1
        group_prefixes[group].add(prefix)

    # ── Summary table ─────────────────────────────────────────────────────────
    SEP = "=" * 70
    logger.info(SEP)
    logger.info("  COMPARE16 FEATURE GROUP SUMMARY")
    logger.info(SEP)
    logger.info(f"  {'Group':<22} {'Features':>10}  {'Unique LLDs':>12}")
    logger.info(f"  {'-'*22} {'-'*10}  {'-'*12}")

    group_order = [name for name, _ in SEMANTIC_RULES]
    total_classified = 0
    for group in group_order:
        if group not in group_cols:
            continue
        n_feats = len(group_cols[group])
        n_llds  = len(group_prefixes[group])
        total_classified += n_feats
        logger.info(f"  {group:<22} {n_feats:>10}  {n_llds:>12}")

    logger.info(f"  {'─'*22} {'─'*10}")
    logger.info(f"  {'TOTAL':<22} {total_classified:>10}")
    logger.info(SEP)
    logger.info("")

    # ── Per-group LLD listing ─────────────────────────────────────────────────
    for group in group_order:
        if group not in group_cols:
            continue
        logger.info(f"── {group} ({len(group_cols[group])} features) ──")
        for p in sorted(group_prefixes[group]):
            logger.info(f"    {p:<52}  ×{prefix_count[p]:>4} functionals")
        logger.info("")

    # ── Export column lists for dataset_matrix.py --feature_group filtering ───
    with open(EXPORT_FILE, "w", encoding="utf-8") as f:
        for group in group_order:
            if group not in group_cols:
                continue
            f.write(f"[{group}]\n")
            for col in group_cols[group]:
                f.write(f"{col}\n")
            f.write("\n")

    logger.info(SEP)
    logger.info(f"  Report saved  →  {REPORT_FILE}")
    logger.info(f"  Column lists  →  {EXPORT_FILE}")
    logger.info(SEP)


if __name__ == "__main__":
    main()
