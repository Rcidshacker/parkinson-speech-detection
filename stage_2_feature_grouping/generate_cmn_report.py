"""
generate_cmn_report.py
======================
Builds Comprehensive_CMN_Results_Report.xlsx from three Stage-2 matrix run directories.

Sheets:
  1_Overview         — Background, problem statement, solution summary
  2_Full_Features    — matrix_full run (52 CMN features)
  3_Structural_MFCC  — matrix_structural_cmvn run (MFCC[1-12])
  4_Temporal_ZCR     — matrix_temporal_cmvn run (ZCR)
  5_Methodology      — Pipeline architecture notes

Author: Ruchit Das (22AM1084) — generated via Claude Code
"""

import os
import pandas as pd
import xlsxwriter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
RESULTS  = os.path.join(BASE, "stage_2_feature_grouping", "results")

DIRS = {
    "2_Full_Features":   os.path.join(RESULTS, "matrix_full_20260406_231750"),
    "3_Structural_MFCC": os.path.join(RESULTS, "matrix_structural_cmvn_20260406_231833"),
    "4_Temporal_ZCR":    os.path.join(RESULTS, "matrix_temporal_cmvn_20260406_231907"),
}

OUT_PATH = os.path.join(BASE, "stage_2_feature_grouping", "Comprehensive_CMN_Results_Report.xlsx")

# ── Workbook ─────────────────────────────────────────────────────────────────
wb = xlsxwriter.Workbook(OUT_PATH)

# ── Shared formats ────────────────────────────────────────────────────────────
def _base(wb, extra=None):
    d = {"font_name": "Calibri", "font_size": 10, "valign": "vcenter"}
    if extra:
        d.update(extra)
    return wb.add_format(d)

fmt_title    = _base(wb, {"bold": True, "font_size": 14, "font_color": "#1e3a5f",
                           "bottom": 2, "bottom_color": "#1e3a5f"})
fmt_h1       = _base(wb, {"bold": True, "font_size": 11, "font_color": "#1e3a5f",
                           "bg_color": "#dce6f1", "border": 1, "border_color": "#aaaaaa"})
fmt_body     = _base(wb, {"text_wrap": True, "border": 1, "border_color": "#cccccc",
                           "bg_color": "#ffffff"})
fmt_bullet   = _base(wb, {"text_wrap": True, "indent": 2, "bg_color": "#f9f9f9",
                           "border": 1, "border_color": "#cccccc"})
fmt_subbullet= _base(wb, {"text_wrap": True, "indent": 4, "bg_color": "#ffffff",
                           "border": 1, "border_color": "#dddddd"})

# Table header: dark navy, bold white
fmt_th       = _base(wb, {"bold": True, "font_color": "#ffffff", "bg_color": "#1e3a5f",
                           "border": 1, "border_color": "#0a1e3a", "align": "center"})
# Table index column (row labels)
fmt_idx      = _base(wb, {"bold": True, "bg_color": "#dce6f1",
                           "border": 1, "border_color": "#aaaaaa"})
# Normal data cells
fmt_cell     = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center"})
fmt_cell_num = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center",
                           "num_format": "0.0000"})
fmt_cell_pct = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center",
                           "num_format": "0.0"})

# Conditional-formatting colours (written as cell formats for manual application)
fmt_red    = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center",
                         "num_format": "0.0000", "bg_color": "#FFCCCC", "font_color": "#9C0006"})
fmt_yellow = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center",
                         "num_format": "0.0000", "bg_color": "#FFEB9C", "font_color": "#9C5700"})
fmt_green  = _base(wb, {"border": 1, "border_color": "#cccccc", "align": "center",
                         "num_format": "0.0000", "bg_color": "#C6EFCE", "font_color": "#276221"})

fmt_section  = _base(wb, {"bold": True, "font_size": 11, "font_color": "#ffffff",
                           "bg_color": "#2e75b6", "border": 1, "border_color": "#1e3a5f"})
fmt_note     = _base(wb, {"italic": True, "font_color": "#555555", "font_size": 9,
                           "text_wrap": True, "bg_color": "#f2f2f2"})


# ═════════════════════════════════════════════════════════════════════════════
# SHEET 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
ws1 = wb.add_worksheet("1_Overview")
ws1.set_column("A:A", 3)
ws1.set_column("B:B", 28)
ws1.set_column("C:H", 18)
ws1.set_zoom(90)

row = 0

# Title
ws1.merge_range(row, 1, row, 7,
    "CMN Pipeline — Experiment Results Overview", fmt_title)
ws1.set_row(row, 28)
row += 2

# ── Problem ──
ws1.merge_range(row, 1, row, 7, "THE PROBLEM", fmt_section)
ws1.set_row(row, 20); row += 1

ws1.merge_range(row, 1, row, 7,
    "High-capacity non-linear models (RF, XGBoost) trained on high-dimensional OpenSMILE "
    "Functionals (~6 373 features) achieved a hallucinated AUC of 1.0 within-corpus. The models "
    "were 100 % memorising static microphone frequency responses and Room Impulse Responses (RIR) "
    "rather than any physiological pathology. Cross-corpus AUC was consequently ~0.59 — barely "
    "above chance — because the acoustic fingerprint of a different microphone/room is entirely "
    "unseen during training.",
    fmt_body)
ws1.set_row(row, 72); row += 2

# ── Solution ──
ws1.merge_range(row, 1, row, 7, "THE SOLUTION", fmt_section)
ws1.set_row(row, 20); row += 1

solutions = [
    ("1.", "Transitioned from pre-aggregated ComParE Functionals → Low-Level Descriptors (LLDs).",
     None),
    ("2.", "Applied Cepstral Mean Normalization (CMN) at utterance level:",
     "StandardScaler(with_std=False) subtracts the per-utterance mean, mathematically "
     "stripping static microphone EQ and room tone from every frame sequence."),
    ("3.", "Preserved frame-to-frame variance:",
     "with_std=False is critical — the temporal variance of MFCCs across sustained-vowel frames "
     "IS the neuromotor tremor signal. CMVN (normalising variance too) would erase the pathology."),
    ("4.", "Truncated to 52 core features:",
     "MFCC[1-12] + ZCR × 4 functionals (stddev, skewness, kurtosis, IQR). "
     "'amean' excluded — CMN forces it to exactly 0.0 for every patient."),
    ("5.", "Enforced capacity constraints:",
     "L1-Logistic Regression (C=0.1, sparsity), Constrained RF (max_depth=3), "
     "Linear SVM (C=0.05). All use RobustScaler to resist clinical-severity outliers."),
]

for num, bullet, sub in solutions:
    ws1.write(row, 1, num, fmt_h1)
    ws1.merge_range(row, 2, row, 7, bullet, fmt_bullet)
    ws1.set_row(row, 30); row += 1
    if sub:
        ws1.write(row, 1, "", fmt_subbullet)
        ws1.merge_range(row, 2, row, 7, sub, fmt_subbullet)
        ws1.set_row(row, 42); row += 1

row += 1

# ── Result ──
ws1.merge_range(row, 1, row, 7, "RESULT", fmt_section)
ws1.set_row(row, 20); row += 1

ws1.merge_range(row, 1, row, 7,
    "Cross-corpus AUC improved from ~0.59 (noise-floor memorisation) to ~0.75 "
    "(ES→EN direction). This confirms successful domain decoupling: the model is now reading "
    "biological neuromotor tremor, not the microphone fingerprint.",
    fmt_body)
ws1.set_row(row, 48); row += 2

# Quick results summary table
ws1.merge_range(row, 1, row, 7, "QUICK RESULTS SUMMARY", fmt_section)
ws1.set_row(row, 20); row += 1

summary_headers = ["Run", "ES→EN AUC", "EN→ES AUC", "Primary Model", "Features"]
for c, h in enumerate(summary_headers):
    ws1.write(row, c + 1, h, fmt_th)
ws1.set_row(row, 18); row += 1

summary_rows = [
    ("Full CMN (52 feats)",   "0.7460", "0.6225", "L1 Logistic Regression", "52 (MFCC+ZCR CMN)"),
    ("Structural MFCC only",  "—",      "—",      "L1 Logistic Regression", "48 (MFCC[1-12] CMN)"),
    ("Temporal ZCR only",     "—",      "—",      "L1 Logistic Regression", "4  (ZCR CMN)"),
]
for sr in summary_rows:
    for c, v in enumerate(sr):
        ws1.write(row, c + 1, v, fmt_cell)
    ws1.set_row(row, 18); row += 1

row += 1
ws1.merge_range(row, 1, row, 7,
    "See sheets 2–4 for full AUC matrices, cascade breakdown, and sub-analysis details.",
    fmt_note)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: write a labelled section header inside a result sheet
# ═════════════════════════════════════════════════════════════════════════════
def section_header(ws, row, col, label, n_cols=8):
    ws.merge_range(row, col, row, col + n_cols - 1, label, fmt_section)
    ws.set_row(row, 20)
    return row + 1


def write_dataframe(ws, df, start_row, start_col,
                    auc_cols=None, pct_cols=None, index_label=None):
    """
    Writes df to ws starting at (start_row, start_col).
    Returns next free row.
    auc_cols: set of column names to colour-code by AUC threshold.
    pct_cols: set of column names to format as percentage.
    """
    # header row
    if index_label is not None:
        ws.write(start_row, start_col, index_label, fmt_th)
    for c, col in enumerate(df.columns):
        ws.write(start_row, start_col + (1 if index_label is not None else 0) + c,
                 col, fmt_th)
    ws.set_row(start_row, 18)
    r = start_row + 1

    for _, row_data in df.iterrows():
        col_offset = start_col
        # index cell
        if index_label is not None:
            ws.write(r, col_offset, row_data.name, fmt_idx)
            col_offset += 1
        for col_name in df.columns:
            val = row_data[col_name]
            is_auc = auc_cols and col_name in auc_cols
            is_pct = pct_cols and col_name in pct_cols
            if is_auc:
                try:
                    v = float(val)
                    if v < 0.60:
                        ws.write(r, col_offset, v, fmt_red)
                    elif v < 0.70:
                        ws.write(r, col_offset, v, fmt_yellow)
                    else:
                        ws.write(r, col_offset, v, fmt_green)
                except (ValueError, TypeError):
                    ws.write(r, col_offset, val, fmt_cell)
            elif is_pct:
                try:
                    ws.write(r, col_offset, float(val), fmt_cell_pct)
                except (ValueError, TypeError):
                    ws.write(r, col_offset, val, fmt_cell)
            elif isinstance(val, float):
                ws.write(r, col_offset, val, fmt_cell_num)
            else:
                ws.write(r, col_offset, val, fmt_cell)
            col_offset += 1
        ws.set_row(r, 16)
        r += 1
    return r


def set_col_widths(ws, df, start_col, index_label=None, extra=1.8):
    """Auto-set column widths based on max string length."""
    col_offset = start_col
    if index_label is not None:
        max_w = max(len(str(index_label)),
                    max(len(str(v)) for v in df.index))
        ws.set_column(col_offset, col_offset, max_w + extra)
        col_offset += 1
    for col in df.columns:
        max_w = max(len(str(col)),
                    max(len(str(v)) for v in df[col]))
        ws.set_column(col_offset, col_offset, min(max_w + extra, 30))
        col_offset += 1


# ═════════════════════════════════════════════════════════════════════════════
# Helper: build one result sheet
# ═════════════════════════════════════════════════════════════════════════════
def build_result_sheet(wb, sheet_name, run_dir, run_label):
    ws = wb.add_worksheet(sheet_name)
    ws.set_zoom(85)
    ws.set_column("A:A", 2)   # left margin

    # ── Read CSVs ──
    auc_df    = pd.read_csv(os.path.join(run_dir, "matrix_auc.csv"),    index_col=0)
    sub_df    = pd.read_csv(os.path.join(run_dir, "subanalysis_results.csv"))
    casc_df   = pd.read_csv(os.path.join(run_dir, "cascade_results.csv"))
    img_path  = os.path.join(run_dir, "matrix_plots.png")

    DATA_COL  = 1   # column B — data tables start here
    IMG_COL   = 10  # column K — image starts here

    row = 0

    # Sheet title
    ws.merge_range(row, DATA_COL, row, IMG_COL + 10,
                   f"{sheet_name.split('_', 1)[1]}  —  {run_label}", fmt_title)
    ws.set_row(row, 26)
    row += 2

    # ── Section A: AUC Matrix ──
    row = section_header(ws, row, DATA_COL,
                         "SECTION A — 3×3 AUC Matrix  (colour: red <0.60 | yellow 0.60-0.70 | green ≥0.70)",
                         n_cols=len(auc_df.columns) + 2)
    auc_numeric_cols = set(auc_df.columns)
    row = write_dataframe(ws, auc_df, row, DATA_COL,
                          auc_cols=auc_numeric_cols, index_label="Train \\ Test")
    set_col_widths(ws, auc_df, DATA_COL, index_label="Train \\ Test")
    row += 2

    # ── Section B: Sub-analysis ──
    row = section_header(ws, row, DATA_COL,
                         "SECTION B — Sub-Analysis Results (Random Forest per feature subset)",
                         n_cols=len(sub_df.columns) + 1)
    row = write_dataframe(ws, sub_df.reset_index(drop=True), row, DATA_COL,
                          auc_cols={"AUC", "CI_lo", "CI_hi"})
    set_col_widths(ws, sub_df, DATA_COL)
    row += 2

    # ── Section C: Cascade ──
    row = section_header(ws, row, DATA_COL,
                         "SECTION C — Cascade Pipeline Results  (L1-LR → Constrained RF → L2 SVM)",
                         n_cols=len(casc_df.columns) + 1)
    row = write_dataframe(ws, casc_df.reset_index(drop=True), row, DATA_COL,
                          auc_cols={"AUC", "CI_lo", "CI_hi"},
                          pct_cols={"pct_handled"})
    set_col_widths(ws, casc_df, DATA_COL)

    # ── Section D: Image ──
    if os.path.exists(img_path):
        ws.insert_image(2, IMG_COL, img_path, {
            "x_scale": 0.62,
            "y_scale": 0.62,
            "x_offset": 5,
            "y_offset": 5,
        })

    return ws


# ── Build sheets 2-4 ─────────────────────────────────────────────────────────
build_result_sheet(wb, "2_Full_Features",
                   DIRS["2_Full_Features"],
                   "Full CMN Feature Set (52 features: MFCC[1-12] + ZCR)")

build_result_sheet(wb, "3_Structural_MFCC",
                   DIRS["3_Structural_MFCC"],
                   "Structural Group — MFCC[1-12] (48 features)")

build_result_sheet(wb, "4_Temporal_ZCR",
                   DIRS["4_Temporal_ZCR"],
                   "Temporal Group — Zero Crossing Rate (4 features)")


# ═════════════════════════════════════════════════════════════════════════════
# SHEET 5 — Methodology
# ═════════════════════════════════════════════════════════════════════════════
ws5 = wb.add_worksheet("5_Methodology")
ws5.set_column("A:A", 3)
ws5.set_column("B:B", 30)
ws5.set_column("C:I", 20)
ws5.set_zoom(90)

row = 0
ws5.merge_range(row, 1, row, 8, "Pipeline Architecture & Methodology", fmt_title)
ws5.set_row(row, 28); row += 2

sections = [
    ("EXTRACTION",
     "OpenSMILE ComParE_2016  →  FeatureLevel.LowLevelDescriptors",
     ["Processes raw .wav files frame-by-frame to produce time-series LLD matrices.",
      "Target LLDs: MFCC[1] through MFCC[12] and pcm_zcr_sma (Zero Crossing Rate).",
      "Extraction runs per-utterance inside a joblib thread-pool (prefer='threads') "
      "to avoid pickling the OpenSMILE object across processes."]),

    ("NORMALISATION — CMN (not CMVN)",
     "sklearn.preprocessing.StandardScaler(with_std=False) — per utterance, across time axis",
     ["Subtracts the per-utterance mean from every frame: removes static microphone EQ "
      "and room-level DC offset.",
      "CRITICAL: with_std=False — variance is NOT normalised. The frame-to-frame standard "
      "deviation of MFCC trajectories during a sustained vowel is the mathematical "
      "representation of neuromotor tremor and vocal instability — the primary PD biomarker.",
      "Applying full CMVN (with_std=True) forces stddev = 1.0 for every speaker, "
      "mathematically erasing the tremor and making healthy and PD patients indistinguishable."]),

    ("AGGREGATION — 4 Robust Functionals",
     "numpy.std, scipy.stats.skew, scipy.stats.kurtosis, IQR (Q75-Q25)",
     ["amean EXCLUDED — CMN forces it to exactly 0.0 for every utterance; "
      "including it adds 13 zero-columns carrying no information.",
      "stddev  — tremor magnitude (temporal instability of MFCC trajectory).",
      "skewness — asymmetry of frame distribution; sensitive to aperiodic voice breaks.",
      "kurtosis — tail heaviness; captures extreme tremor events / sudden pitch breaks.",
      "IQR      — robust spread measure; resistant to outlier frames (breath artefacts).",
      "Total features: 13 LLDs × 4 functionals = 52."]),

    ("CROSS-DOMAIN EVALUATION — Leave-One-Corpus-Out (LOCO)",
     "Train on full Spanish PC-GITA → evaluate blindly on English VOICE_DATASET (and vice-versa)",
     ["No hyper-parameter tuning on target corpus — pure zero-shot cross-lingual transfer.",
      "Scaler (RobustScaler) fitted exclusively on training corpus; applied to test corpus.",
      "Off-diagonal AUC is the primary research metric — within-corpus CV is diagnostic only."]),

    ("WITHIN-CORPUS VALIDATION",
     "5-Fold StratifiedGroupKFold on subject_id",
     ["Groups by subject_id prevents the same speaker from appearing in both train and val folds.",
      "Ensures CV AUC reflects speaker-independent generalisation, not utterance-level memorisation.",
      "Scaler and imputer are fitted inside each fold via sklearn Pipeline — never on full data."]),

    ("CASCADE INFERENCE",
     "L1-LR (≥70% confidence) → Constrained RF (≥65%) → L2 SVM Linear (fallback)",
     ["Stage 1 — L1 Logistic Regression (C=0.1, liblinear solver): handles high-confidence "
      "cases. L1 penalty enforces feature sparsity — drops irrelevant features automatically.",
      "Stage 2 — Constrained RF (n_estimators=100, max_depth=3, min_samples_leaf=10): "
      "shallow trees capture broad physiological trends without memorising domain noise.",
      "Stage 3 — Linear SVM (C=0.05): linear boundary forced to find macroscopic "
      "physiological separability. Handles all remaining uncertain cases.",
      "All stages: RobustScaler (median/IQR normalisation) — resistant to clinical severity "
      "outliers that distort standard mean/variance scaling."]),
]

for sec_title, sec_subtitle, bullets in sections:
    ws5.merge_range(row, 1, row, 8, sec_title, fmt_section)
    ws5.set_row(row, 20); row += 1

    ws5.merge_range(row, 1, row, 8, sec_subtitle, fmt_h1)
    ws5.set_row(row, 28); row += 1

    for b in bullets:
        ws5.write(row, 1, "•", fmt_bullet)
        ws5.merge_range(row, 2, row, 8, b, fmt_bullet)
        ws5.set_row(row, 32); row += 1
    row += 1

# Footer note
ws5.merge_range(row, 1, row, 8,
    "Reference: PC-GITA (Spanish, 300 rows, 150 PD / 150 HC) | "
    "VOICE_DATASET (English/Kaggle, 567 rows, 280 PD / 287 HC) | "
    "Italian dataset excluded from cross-lingual analysis (n=99, too small).",
    fmt_note)
ws5.set_row(row, 24)


# ═════════════════════════════════════════════════════════════════════════════
# Close
# ═════════════════════════════════════════════════════════════════════════════
wb.close()
print(f"\n✓  Report written to:\n   {OUT_PATH}")
print(f"   Size: {os.path.getsize(OUT_PATH):,} bytes")
