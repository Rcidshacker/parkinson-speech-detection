# Speech-Based Parkinson's Disease Detection
### Cross-Lingual Generalization Study — BE Capstone, RAIT, DY Patil University

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![OpenSMILE](https://img.shields.io/badge/OpenSMILE-2.6.0-green)
![Status](https://img.shields.io/badge/Status-Stage%202%20In%20Progress-yellow)

> **Research question:** Can a model trained on Spanish Parkinson's speech detect PD in English speech — and vice versa?

Classical ML models trained on sustained-vowel acoustic features are evaluated in a 3×3 cross-dataset generalization matrix (Spanish PC-GITA ↔ English Voice Dataset) across three feature regimes: handcrafted phonatory biomarkers, eGeMAPS, and ComParE16 functionals.

**Author:** Ruchit Das (22AM1084) · **Supervisor:** Prof. Pramod Kachare · **Institution:** RAIT, DY Patil University, Navi Mumbai

---

## Key Results

### Stage 1 — Baseline (ComParE16, ~6373 features, 8 kHz)

| Direction | AUC | Notes |
|---|---|---|
| PC-GITA (ES) within-dataset | 0.813 | 5-fold StratifiedGroupKFold |
| Voice Dataset (EN) within-dataset | 0.986 | |
| ES → EN (cross-lingual) | 0.642 | True zero-overlap |
| EN → ES (cross-lingual) | 0.664 | True zero-overlap |
| Combined → Combined | 0.928 | |
| Best cross-lingual (Top-9 features + LR) | **0.842** | |

### Stage 1 — eGeMAPS (83 features)

| Direction | Best Model | AUC |
|---|---|---|
| Voice Dataset within | SVM / VotingEnsemble | 0.994 |
| PC-GITA within | XGBoost | 0.893 |
| ES → EN | Stacking | 0.604 |
| EN → ES | SVM | 0.679 |

### Stage 2 — CMN-normalized features (52 features, capacity-constrained models)

| Direction | AUC |
|---|---|
| Full CMN: ES → EN | **0.752** |
| Full CMN: EN → ES | **0.652** |
| Structural MFCC CMN only: ES → EN | 0.733 |
| Temporal ZCR CMN only: ES → EN | 0.622 |

**Key insight:** Per-utterance mean normalization (CMN) improved cross-lingual ES→EN AUC from 0.642 → 0.752 by stripping microphone EQ differences while deliberately preserving frame-level variance — the vocal tremor signal.

---

## Pipeline Architecture

```
Dataset/ (raw WAV)
    │
    ▼
Audio preprocessing
  resample → 8/10/16 kHz
  RMS-normalize to TARGET_RMS=0.04
    │
    ▼
Feature extraction
  librosa (handcrafted, ~112 features)
  OpenSMILE eGeMAPSv02 (88 → 83 after VarianceThreshold)
  OpenSMILE ComParE_2016 (6,373 functionals)
    │
    ▼
features/*.csv  ──────────────────────────────────┐
    │                                              │
    ▼                                              ▼
prepare_*_training.py                  Stage 2: LLD-CMN pipeline
  dedup (MD5 hash, Voice_Dataset)       per-utterance mean subtraction
  ID neutralization                     52 MFCC+ZCR functionals
  VarianceThreshold(0.001)
    │
    ▼
dataset_wise_analysis_v2.py            stage_2_feature_grouping/scripts/03_group_matrix.py
  7 models                               3 capacity-constrained models
  5-fold StratifiedGroupKFold            L1-LR(C=0.1) → constrained RF → linear SVM
  on subject_id                          cascade with confidence thresholds
    │
    ▼
results/<run>/
  ROC curves, confusion matrices, AUC heatmaps
  per-model JSON reports
  comprehensive_analysis_report.xlsx
```

---

## Datasets

| Dataset | Language | Rows | PD / HC | Source |
|---|---|---|---|---|
| **PC-GITA** | Spanish | 300 | 150 / 150 | Ibáñez et al., 2014 (contact authors) |
| **Voice Dataset** | English | 567 (after dedup) | 280 / 287 | [Kaggle UCI extended](https://www.kaggle.com/) |
| Italian Parkinson's Voice | Italian | 99 | — | Excluded — scope mismatch |

> **Dataset audio files are NOT included in this repository** (too large for git).  
> Place them under `Dataset/PC-GITA/`, `Dataset/Voice_Dataset/`, and `Dataset/Italian Parkinson's Voice and speech/` before running extraction.

**Deduplication note:** Voice Dataset had 470 exact duplicate audio pairs (detected by feature-hash rounding to 4 decimal places). After dedup: 1037 raw → 567 unique rows. All removed pairs are logged to `features/dedup_report_*.csv`.

---

## Setup

**Requirements:** Python 3.13, Windows (paths are Windows-style; adjust for Linux/Mac)

```powershell
# Clone and enter repo
git clone <repo-url>
cd "BE mini project"

# Create venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

```powershell
$PY   = ".\venv\Scripts\python.exe"
$BASE = "C:\path\to\BE mini project"   # ← adjust this

# Step 1 — Extract features (handcrafted + OpenSMILE)
& $PY "$BASE\scripts\pipeline\extract_features_sustained_a.py"
& $PY "$BASE\scripts\pipeline\extract_features_sustained_a_opensmile.py"

# Step 2 — Prepare training datasets
& $PY "$BASE\scripts\pipeline\prepare_egemaps_training.py"
& $PY "$BASE\scripts\pipeline\prepare_compare_training.py"

# Step 3a — Within-dataset evaluation (eGeMAPS)
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" `
    --data_file "$BASE\features\opensmile\training_egemaps_full88f.csv"

# Step 3b — Within-dataset evaluation (ComParE16, with K-Best selection)
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" `
    --data_file "$BASE\features\opensmile\training_compare_8k_full.csv" `
    --kbest 50

# Step 4 — 3×3 cross-dataset generalization matrix
& $PY "$BASE\scripts\pipeline\dataset_matrix.py"

# Step 5 — Stage 2: CMN feature group matrix
& $PY "$BASE\stage_2_feature_grouping\scripts\03_group_matrix.py"
# Filter to a specific feature group:
& $PY "$BASE\stage_2_feature_grouping\scripts\03_group_matrix.py" --feature_group structural_cmvn

# Step 6 — Compile all results to Excel
& $PY "$BASE\scripts\pipeline\compile_results_xlsx.py"
& $PY "$BASE\stage_2_feature_grouping\generate_cmn_report.py"

# Run everything in one batch
& $PY "$BASE\scripts\pipeline\_batch_run.py"
```

Results land in `results/evaluation_<dataset>_<timestamp>/` and `results/matrix_<timestamp>/`.

---

## Project Structure

```
BE mini project/
├── scripts/
│   ├── pipeline/                   # ← canonical pipeline scripts
│   │   ├── extract_features_sustained_a.py        # Handcrafted features (librosa)
│   │   ├── extract_features_sustained_a_opensmile.py  # eGeMAPS + ComParE16
│   │   ├── prepare_egemaps_training.py            # Filter + dedup → training CSV
│   │   ├── prepare_compare_training.py
│   │   ├── dataset_wise_analysis_v2.py            # Main eval engine (7 models)
│   │   ├── dataset_matrix.py                      # 3×3 cross-dataset matrix
│   │   ├── compile_results_xlsx.py                # Aggregate → Excel report
│   │   └── _batch_run.py                          # Run all feature sets
│   ├── utils/                      # Helper utilities
│   └── experimental/               # Deprecated / scratch scripts
│
├── stage_1_baseline/               # Stage 1 documentation & analysis
│   ├── README.md
│   ├── SINGLE_TRUTH_REPORT.md      # Forensic audit (3 critical bug fixes)
│   ├── FIXES_APPLIED_REPORT.md
│   └── notebooks/                  # Architecture diagrams
│
├── stage_2_feature_grouping/       # Stage 2: CMN feature grouping
│   ├── scripts/
│   │   ├── 03_group_matrix.py      # Parameterized group matrix
│   │   ├── 01_inspect_groups.py    # TODO stub
│   │   └── 02_prepare_group_csvs.py # TODO stub
│   ├── group_definitions/
│   │   └── groups.json             # Feature group definitions
│   ├── generate_cmn_report.py      # Build Comprehensive_CMN_Results_Report.xlsx
│   └── README.md
│
├── features/                       # Generated CSVs (gitignored)
├── results/                        # Evaluation outputs (gitignored)
├── logs/                           # Run logs (gitignored)
├── Dataset/                        # Raw audio (gitignored — download separately)
│
├── PROJECT_SUMMARY.md              # Technical summary with full results
├── STAGES.md                       # Stage roadmap
├── AGENTS.md                       # Coding conventions & checklist
├── CLAUDE.md                       # Project config for Claude Code
├── requirements.txt
└── .gitignore
```

---

## Models

### Stage 1 — Full-capacity models

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| SVM | `kernel='rbf', C=10` |
| Random Forest | `n_estimators=150, max_depth=20` |
| Decision Tree | `max_depth=15` |
| XGBoost | GPU→CPU fallback |
| VotingClassifier | Soft voting, all 5 base models |
| StackingClassifier | LR meta-learner, 5-fold CV |

All preprocessing (StandardScaler, SimpleImputer, optional SelectKBest) is fitted **inside** CV folds — never on full data. GroupKFold on `subject_id` prevents speaker leakage.

### Stage 2 — Capacity-constrained models (prevents noise-floor memorization on 52 features)

| Model | Config |
|---|---|
| L1 Logistic Regression | `penalty='l1', C=0.1, solver='liblinear'` |
| Constrained Random Forest | `max_depth=3, min_samples_leaf=10` |
| Linear SVM | `kernel='linear', C=0.05` |

**Cascade:** L1-LR (≥70% confidence) → Constrained RF (≥65%) → Linear SVM (always outputs).

---

## Why CMN (Not CMVN)

Stage 2 applies `StandardScaler(with_std=False)` **per utterance** before computing MFCC functionals — this subtracts the utterance mean (stripping static microphone EQ / channel bias) but **deliberately skips variance normalization**, because `std(MFCC frame)` is exactly where vocal tremor manifests.

The `amean` feature is excluded from all downstream models because it is identically `0.0` post-normalization by construction — including it would be dead weight.

---

## Roadmap

| Stage | Status | Description |
|---|---|---|
| 1 — Baseline | ✅ Done | Feature extraction + classical ML; best cross-lingual AUC 0.842 |
| 2 — Feature Grouping | 🔄 In Progress | LLD-CMN + capacity-constrained matrix; ES→EN AUC 0.752 |
| 3 — Foundation Model | 📋 Planned | Fine-tune wav2vec 2.0 / HuBERT; target cross-lingual AUC >0.90 |

---

## Technical Notes

- `8k/10k/16k` in filenames = **audio sample rate in Hz**, not row counts. All three variants contain the same recordings resampled at different frequencies.
- `RANDOM_STATE = 42`, `N_FOLDS = 5`, `N_BOOTSTRAP = 1000`
- Matplotlib runs in headless `Agg` mode (no display required).

---

## References

- Ibáñez, R. et al. (2014). *PC-GITA: A database to improve the study of Parkinson's disease.* — PC-GITA corpus
- Eyben, F. et al. — OpenSMILE toolkit (eGeMAPSv02, ComParE_2016)
- Schuller, B. et al. — ComParE challenge feature set

---

## Citation

If you use this code or findings, please cite:

```
Ruchit Das (2026). Speech-Based Parkinson's Disease Detection:
A Cross-Lingual Generalization Study. BE Capstone Project,
RAIT, DY Patil University, Navi Mumbai.
Supervisor: Prof. Pramod Kachare.
```
