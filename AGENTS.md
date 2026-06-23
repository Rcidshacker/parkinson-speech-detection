# AGENTS.md — PD Speech Detection Project

## Project Overview
- **Type:** Python research codebase for speech-based Parkinson's Disease detection
- **Language:** Python 3.x (Windows PowerShell environment)
- **Virtual Environment:** `venv/Scripts/python.exe`
- **Root:** `C:\Users\Lenovo\Desktop\Code\2026\BE mini project`

## Directory Structure
```
scripts/
├── pipeline/                        # Primary working directory (recommended)
│   ├── extract_features_*.py        # Feature extraction from audio
│   ├── dataset_wise_analysis_v2.py  # Primary evaluation engine
│   ├── dataset_matrix.py            # 3×3 cross-dataset generalization matrix
│   ├── prepare_egemaps_training.py  # eGeMAPS dataset preparation
│   └── prepare_compare_training.py  # ComParE16 dataset preparation
├── _batch_run.py                    # Batch evaluation runner
└── *.py                            # Legacy/experimental scripts

features/                           # Feature CSVs (input/output data)
├── features_sustained_a.csv        # Handcrafted features (112 features)
├── features_egemaps_8k.csv         # eGeMAPS features (83 features after VarianceThreshold)
├── features_compare_8k.csv         # ComParE16 features (~6373 features)
└── opensmile/                      # Training subsets
    ├── training_egemaps_*.csv     # eGeMAPS training CSVs
    └── training_compare_8k_full.csv # ComParE16 training CSV

results/                            # Evaluation outputs (auto-timestamped)
Dataset/                            # Raw audio files
team_repos/                         # Teammate repositories
```

## Build/Lint/Test Commands

### Running Scripts
```powershell
$PY = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\venv\Scripts\python.exe"
$BASE = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

# Feature extraction (sustained vowel 'a')
& $PY "$BASE\scripts\pipeline\extract_features_sustained_a.py"

# eGeMAPS dataset preparation
& $PY "$BASE\scripts\pipeline\prepare_egemaps_training.py"

# ComParE16 dataset preparation (NEW)
& $PY "$BASE\scripts\pipeline\prepare_compare_training.py"

# Evaluation engine (primary analysis)
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_egemaps_full88f.csv"

# Evaluation with K-Best feature selection (NEW)
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_compare_8k_full.csv" --kbest 50
& $PY "$BASE\scripts\pipeline\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_compare_8k_full.csv" --kbest 100

# 3×3 Cross-dataset matrix
& $PY "$BASE\scripts\pipeline\dataset_matrix.py"

# Batch evaluation
& $PY "$BASE\scripts\_batch_run.py"
```

### No Formal Test Suite
This is research code without automated tests. Manual verification:
- Check output CSVs for expected row counts (867 rows for eGeMAPS full88f)
- Verify `results/evaluation_*` directories contain expected files
- Cross-reference with `SESSION_MEMORY.md` for expected outputs

## Code Style Guidelines

### Imports (ordered groups, blank lines between)
```python
# Standard library
import os
import sys
import logging
from datetime import datetime

# Third-party
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# sklearn submodules (separate group)
from sklearn.feature_selection import SelectKBest, f_classif

# Local imports (if any)
# from module import function
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Constants | `UPPERCASE_WITH_UNDERSCORES` | `RANDOM_STATE`, `N_FOLDS` |
| Functions | `snake_case` | `load_full_data`, `extract_features` |
| Classes | `PascalCase` | `VotingClassifier`, `StackingClassifier` |
| Files | `snake_case.py` | `dataset_wise_analysis_v2.py` |
| Variables | `snake_case` | `feature_cols`, `y_test` |

### Dataset Labels (exact strings in CSVs)
```python
PCGITA_LABEL   = "pc_gita"        # Spanish, 300 rows
VOICED_LABEL   = "voice_dataset"  # English-adjacent, 567 rows
ITALIAN_LABEL  = "italian"       # 99 rows - EXCLUDED from cross-lingual
```

### Logging Pattern
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("module_name")
```
- Log to both file and stdout via handlers
- Use `logger.info()`, `logger.warning()`, `logger.error()`
- Include row counts, timing, and data shapes

### Docstring Format
Module-level docstrings include:
```python
"""
Module Name
============
Project : Speech-Based Parkinson's Disease Detection
Author  : Name (ID)

CHANGES FROM PREVIOUS:
    1. Change description with rationale

KEY CONSTANTS:
    - CONSTANT_NAME: description
"""
```

### Error Handling
```python
try:
    # operation
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # fallback or propagate with sys.exit(1)
```

### Data Leakage Prevention
- Use `StratifiedGroupKFold` on `subject_id` to prevent speaker leakage
- Scale/impute within CV folds, never on full data first
- Document anti-leakage measures with `[L1]`, `[L2]` prefixes
- Document feature quality decisions with `[FQ1]`, `[FQ2]` prefixes

### Bug Fix Documentation
- Mark fixed bugs with `# FIX-X: Description` comments
- Include root cause analysis and before/after in comments

### Matplotlib Usage
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI required)
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode
```

### File Paths (Windows)
```python
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
OUTPUT_CSV = os.path.join(BASE, "features", "features_sustained_a.csv")
```

## Key Constants
```python
RANDOM_STATE = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
TARGET_SR = 8000  # Hz
TARGET_RMS = 0.04
VARIANCE_THRESHOLD = 0.001
KBEST_K = None  # Set via --kbest CLI arg; None = use all features
```

## Output Convention
Results auto-route to: `results/evaluation_<dataset>_<timestamp>/`
```
csv_results/         # analysis_summary.csv
visualizations/      # ROC curves, heatmaps
matrices/           # Confusion matrices
reports/            # detailed_results.json
```

Matrix outputs go to: `results/matrix_<timestamp>/`

## Anti-Leakage Checklist
- [ ] Subject IDs must not encode disease labels
- [ ] No duplicate feature vectors (deduplicate after extraction)
- [ ] Use GroupKFold, never standard KFold
- [ ] Scale/impute within CV folds only
- [ ] Log Italian row exclusion explicitly (not silently)
- [ ] When using `--kbest`, SelectKBest is fitted inside each CV fold

## Model Configuration
7 models are trained in evaluation engine (`dataset_wise_analysis_v2.py`):
1. LogisticRegression (class_weight='balanced')
2. SVM (kernel='rbf', C=10)
3. RandomForest (n_estimators=150, max_depth=20)
4. DecisionTree (max_depth=15)
5. XGBoost (GPU→CPU fallback)
6. VotingClassifier (soft voting, all fitted models)
7. StackingClassifier (LR meta-learner, 5-fold CV)

## Important Reference Files
| File | Purpose |
|------|---------|
| `SESSION_MEMORY.md` | Current state and key constants |
| `SINGLE_TRUTH_REPORT.md` | Forensic analysis and methodology |
| `FIXES_APPLIED_REPORT.md` | Record of all applied bug fixes |
| `files/SCRIPT_FIXES.md` | Detailed fix instructions for scripts |

## Feature Sets
| Set | Features | Location |
|-----|----------|----------|
| Handcrafted | ~112 | `features/features_sustained_a.csv` |
| eGeMAPS | 83 | `features/opensmile/training_egemaps_*.csv` |
| ComParE16 | ~6373 | `features/opensmile/training_compare_8k_full.csv` |
