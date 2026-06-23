# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech-Based Parkinson's Disease Detection — BE Capstone project by Ruchit Das (22AM1084), RAIT, DY Patil University. Python research codebase using audio feature extraction + ML classification across 3 datasets.

- **Venv:** `venv/Scripts/python.exe` (Windows)
- **No formal test suite** — research code; verify by checking output row counts and `results/` directory contents

## Running Scripts

All commands use the venv Python. On Windows PowerShell:

```powershell
$PY = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\venv\Scripts\python.exe"
$BASE = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

# Feature extraction
& $PY "$BASE\scripts\pipeline\extract_features_sustained_a.py"

# Prepare training datasets
& $PY "$BASE\scripts\pipeline\prepare_egemaps_training.py"
& $PY "$BASE\scripts\pipeline\prepare_compare_training.py"

# Primary evaluation engine
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_egemaps_full88f.csv"

# With K-Best feature selection (ComParE16 has ~6373 features)
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_compare_8k_full.csv" --kbest 50

# 3×3 cross-dataset generalization matrix
& $PY "$BASE\scripts\pipeline\dataset_matrix.py"

# Batch evaluation (runs all feature sets sequentially)
& $PY "$BASE\scripts\pipeline\_batch_run.py"

# Compile all results to Excel
& $PY "$BASE\scripts\pipeline\compile_results_xlsx.py"
```

### Stage 2 — CMN Feature Grouping (`stage_2_feature_grouping/scripts/`)

> **Note:** `01_inspect_groups.py` and `02_prepare_group_csvs.py` are TODO stubs — not yet implemented.

The extraction pipeline uses LLD-level CMN (mean-only normalization, not CMVN). Re-run extraction before the matrix if features are stale:

```powershell
# Step 1: re-extract with LLD-CMN pipeline (produces 52-feature CSV)
& $PY "$BASE\scripts\pipeline\extract_features_sustained_a_opensmile.py"

# Step 2: rebuild training CSV from new features
& $PY "$BASE\scripts\pipeline\prepare_compare_training.py"

# Step 3: run the 3×3 matrix with capacity-constrained models
& $PY "$BASE\stage_2_feature_grouping\scripts\03_group_matrix.py"

# Or filter to a specific CMN group:
& $PY "$BASE\stage_2_feature_grouping\scripts\03_group_matrix.py" --feature_group structural_cmvn
& $PY "$BASE\stage_2_feature_grouping\scripts\03_group_matrix.py" --feature_group temporal_cmvn

# Generate presentation-ready Excel report from completed runs:
& $PY "$BASE\stage_2_feature_grouping\generate_cmn_report.py"
```

## Architecture

### Data Flow
```
Dataset/ (raw audio) → extract_features_*.py → features/*.csv
                                                      ↓
features/opensmile/  ← prepare_*_training.py ← features/opensmile/*.csv
        ↓
dataset_wise_analysis_v2.py  →  results/evaluation_<dataset>_<timestamp>/
dataset_matrix.py            →  results/matrix_<timestamp>/
```

### Primary Scripts (`scripts/pipeline/`)
- **`dataset_wise_analysis_v2.py`** — main evaluation engine; trains 7 models with StratifiedGroupKFold CV; outputs ROC curves, confusion matrices, JSON reports
- **`dataset_matrix.py`** — 3×3 cross-dataset generalization matrix (train on one dataset, test on another)
- **`prepare_egemaps_training.py`** — filters `features_egemaps_8k.csv` by dataset column to produce `training_egemaps_full88f.csv`
- **`prepare_compare_training.py`** — same for ComParE16 features

### Feature Sets
| Name | Features | Source CSV | Notes |
|------|----------|------------|-------|
| Handcrafted | ~112 | `features/features_sustained_a.csv` | |
| eGeMAPS | 83 (after VarianceThreshold) | `features/opensmile/training_egemaps_*.csv` | |
| ComParE16 (Stage 1) | ~6373 | `features/opensmile/training_compare_8k_full.csv` | Raw Functionals |
| ComParE16-CMN (Stage 2) | 52 | `features/opensmile/training_compare_8k_full.csv` | LLD-CMN pipeline: MFCC[1-12] + ZCR × 4 functionals (std, skewness, kurtosis, IQR) |

Raw feature CSVs exist at three sample rates (`8k`, `10k`, `16k`) in `features/opensmile/` — e.g. `features_compare_8k.csv`. The `8k` variants are the canonical inputs.

### Datasets
```python
PCGITA_LABEL  = "pc_gita"       # Spanish, 300 rows (150 PD / 150 HC)
VOICED_LABEL  = "voice_dataset" # English/Kaggle, 567 rows (280 PD / 287 HC)
ITALIAN_LABEL = "italian"       # 99 rows — EXCLUDED from cross-lingual analysis
```

### Models (in evaluation engine — `dataset_wise_analysis_v2.py`)
1. LogisticRegression (`class_weight='balanced'`)
2. SVM (`kernel='rbf', C=10`)
3. RandomForest (`n_estimators=150, max_depth=20`)
4. DecisionTree (`max_depth=15`)
5. XGBoost (GPU→CPU fallback)
6. VotingClassifier (soft voting, all 5 base models)
7. StackingClassifier (LR meta-learner, 5-fold CV)

### Models (Stage 2 — `03_group_matrix.py`)
Capacity-constrained to prevent noise-floor memorization. All use `RobustScaler` instead of `StandardScaler`:
- **L1 Logistic Regression** — `penalty='l1', solver='liblinear', C=0.1` (main matrix model)
- **Constrained RF** — `n_estimators=100, max_depth=3, min_samples_leaf=10`
- **L2 SVM (Linear)** — `kernel='linear', C=0.05`

Cascade: L1-LR (≥70% confidence) → Constrained RF (≥65%) → L2 SVM (always outputs).

## Key Constants
```python
RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
TARGET_SR = 8000        # Hz — "8k" in filenames means sample rate, NOT row count
TARGET_RMS = 0.04
VARIANCE_THRESHOLD = 0.001
```

## Critical Patterns

**Anti-leakage (mandatory):**
- Use `StratifiedGroupKFold` on `subject_id` — never standard KFold
- Fit scalers/imputers/SelectKBest **inside** CV folds only, never on full data
- `--kbest` flag: SelectKBest is fitted inside each fold
- Stage 2 CMN: `StandardScaler(with_std=False)` is applied **per utterance** (inside `extract_cmvn_functionals`) — subtracts the mean to strip static mic EQ, but **never normalizes variance**, because frame-level std IS the vocal tremor signal. `amean` is excluded from output features as it is exactly `0.0` post-CMN for every file.

**Matplotlib (headless environment):**
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
```

**Logging pattern:**
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
```

**File paths:** Use `os.path.join(BASE, ...)` with `BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"`.

## Output Convention
- Evaluation results → `results/evaluation_<dataset>_<timestamp>/` with subdirs: `csv_results/`, `visualizations/`, `matrices/`, `reports/`
- Matrix results → `results/matrix_<timestamp>/`

## Project Stages

| Stage | Status | Description |
|-------|--------|-------------|
| 1 — Baseline | Done | Feature extraction + classical ML, best cross-lingual AUC ~0.842 |
| 2 — Feature Grouping | In Progress | LLD-CMN extraction → 52 features; capacity-constrained cross-dataset matrix; achieved ES→EN AUC 0.746, EN→ES AUC 0.623 |
| 3 — Foundation Model | Planned | Fine-tune wav2vec 2.0 / HuBERT, target AUC >0.90 |

See `STAGES.md` for a brief overview. Stage 2 config lives in `stage_2_feature_grouping/group_definitions/groups.json`.

### Stage 2 group definitions (`groups.json`)
```json
{
  "structural_cmvn": ["mfcc_sma[1]_", ..., "mfcc_sma[12]_"],
  "temporal_cmvn":   ["pcm_zcr_sma_"]
}
```
These match CMN feature names produced by `extract_features_sustained_a_opensmile.py`. Column names follow the pattern `mfcc_sma[N]_stddev`, `mfcc_sma[N]_skewness`, `mfcc_sma[N]_kurtosis`, `mfcc_sma[N]_iqr` (no `_amean` — always 0 after CMN).

## Reference Files
| File | Purpose |
|------|---------|
| `SESSION_MEMORY.md` | Forensic audit summary and current state |
| `SINGLE_TRUTH_REPORT.md` | Methodology and analysis findings |
| `FIXES_APPLIED_REPORT.md` | All bug fixes with root cause analysis |
| `AGENTS.md` | Detailed coding conventions and checklist |
| `STAGES.md` | High-level stage roadmap |
| `stage_2_feature_grouping/generate_cmn_report.py` | Builds `Comprehensive_CMN_Results_Report.xlsx` from Stage 2 result dirs; requires `xlsxwriter` |
| `stage_2_feature_grouping/Comprehensive_CMN_Results_Report.xlsx` | Presentation-ready report: AUC matrices, cascade results, sub-analysis, embedded plots, methodology |
