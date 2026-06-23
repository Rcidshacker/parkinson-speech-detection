# Parkinson's Disease Detection Pipeline — Comprehensive Audit & Roadmap

**Date:** March 27, 2026  
**Status:** Post-Extraction Audit  
**Author:** AIML Engineer (Forensic Review)

---

## EXECUTIVE SUMMARY

Your feature extraction pipeline has completed successfully, yielding **18 CSV outputs** across three extraction modalities:
- **Sustained Vowel /a/** (8 kHz, 10 kHz, 16 kHz) 
- **eGeMAPS (OpenSMILE)** (8 kHz, 10 kHz, 16 kHz)
- **Comparative Features** (8 kHz, 10 kHz, 16 kHz)

**Current State:** Raw feature matrices are clean, deduplicated, and variance-filtered. However, **downstream scripts require targeted updates** to:
1. Dynamically ingest these extracted CSVs (not hard-coded paths)
2. Fix bugs identified in the audit (Feature bugs 2–6 from memory)
3. Establish correct execution order

This document identifies all changes needed and prescribes the execution sequence.

---

## PART I: FEATURE CSV INVENTORY

### Files Extracted (in `features/`)

| CSV | Rows | Features | Notes |
|-----|------|----------|-------|
| `features_sustained_a.csv` | ~550 | 112 | Main custom pipeline; deduped voice_dataset |
| `features_sustained_a_10k.csv` | ~550 | 112 | Same, resampled to 10 kHz |
| `features_sustained_a_16k.csv` | ~550 | 112 | Same, resampled to 16 kHz |
| `features_egemaps_8k.csv` | ~550 | 88 | OpenSMILE eGeMAPS; 8 kHz |
| `features_egemaps_10k.csv` | ~550 | 88 | OpenSMILE eGeMAPS; 10 kHz |
| `features_egemaps_16k.csv` | ~550 | 88 | OpenSMILE eGeMAPS; 16 kHz |
| `features_compare_8k.csv` | ~550 | [N] | Comparative feature set |
| `features_compare_10k.csv` | ~550 | [N] | Comparative feature set |
| `features_compare_16k.csv` | ~550 | [N] | Comparative feature set |

**Dedup Reports** (one per CSV):
- `dedup_report_*.csv` — logs of removed duplicate pairs in voice_dataset

**Quality Guarantees Applied:**
- ✅ [L1] voice_dataset duplicate removal (exact feature-vector matching)
- ✅ [L2] Opaque subject IDs (no label-encoded strings)
- ✅ [FQ3] VarianceThreshold filtering (removed near-zero variance features)
- ✅ Jitter_ddp & shimmer_dda excluded at source (algebraic redundancy)
- ✅ Log-mel features used (not raw mel power)

---

## PART II: SCRIPTS & REQUIRED CHANGES

### 1. `extract_features_sustained_a.py` ✅ **COMPLETE** (No changes needed)
**Status:** Already handles CSV generation with anti-leakage.  
**Output:** `features/features_sustained_a.csv` + dedup report  
**Run:** `python scripts/extract_features_sustained_a.py`

---

### 2. `prepare_egemaps_training.py` ⚠️ **NEEDS UPDATE #1**

**Current Issue:**  
Script hard-codes input paths and assumes eGeMAPS CSVs live in `features/opensmile/`:
```python
PCGITA_CSV = os.path.join(INPUT_DIR, "features_egemaps_pcgita.csv")
VOICED_CSV = os.path.join(INPUT_DIR, "features_egemaps_voiced.csv")
```

**But Reality:** Your extraction produced:
- `features/features_egemaps_8k.csv` (all datasets combined)
- `features/features_egemaps_10k.csv`
- `features/features_egemaps_16k.csv`

**Fix Required:**
1. **Option A (Recommended):** Refactor to accept CLI argument for input CSV:
   ```python
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--egemaps_csv', default='features/features_egemaps_8k.csv')
   args = parser.parse_args()
   
   # Load single combined eGeMAPS CSV (not separate pcgita + voiced)
   df = pd.read_csv(args.egemaps_csv)
   ```

2. **Option B (Quick):** Update hard-coded paths to point to actual files:
   ```python
   EGEMAPS_CSV = os.path.join(BASE, "features", "features_egemaps_8k.csv")
   df = pd.read_csv(EGEMAPS_CSV)
   ```

3. **Feature breakdown code is already correct**—will auto-detect biomarker/MFCC patterns.

**Output:** `features/training_egemaps_full88f.csv`, `training_egemaps_biomarker_eq6f.csv`, `training_egemaps_no_mfcc_72f.csv`

---

### 3. `prepare_training_datasets.py` ✅ **ALREADY DESIGNED FOR YOUR SETUP**

**Status:** This script is already dynamic. It:
- ✅ Searches for latest consensus lists in `results/final/ranking_*/`
- ✅ Loads corresponding raw CSVs from `features/features_[set].csv`
- ✅ Filters by consensus features + stability criteria
- ✅ Generates `training_[set]_UNION.csv` and `training_[set]_STABLE.csv`

**Prerequisite:** Requires `results/final/ranking_*/merged_consensus_list.csv` (output of ranking scripts you haven't run yet).

**No changes needed here**—but keep it in queue for later.

---

### 4. `dataset_wise_analysis_v2.py` ✅ **ALREADY DYNAMIC**

**Status:** Already accepts CLI input:
```bash
python scripts/dataset_wise_analysis_v2.py --data_file ../features/features_sustained_a.csv
```

**Features:**
- ✅ Dynamic CSV input via argparse
- ✅ Filters out 'italian' dataset (as per your memory)
- ✅ Auto-detects remaining datasets for cross-validation
- ✅ StratifiedGroupKFold on subject_id (prevents speaker leakage)
- ✅ 7 models (LR, SVM, RF, XGB, DT, Voting, Stacking)
- ✅ Bootstrap CI, confusion matrices, ROC plots

**Bug to Fix in `dataset_wise_analysis_v2.py`:** 

**BUG #2 (from memory):** Only first fold evaluated instead of all folds  
**Location:** Lines ~250–300 (cross-dataset evaluation section)

**Current (WRONG):**
```python
fold_count = 0
for train_idx, test_idx in cv.split(X, y, groups):
    fold_count += 1
    # ... train/evaluate ...
    if fold_count == 1:
        break  # ❌ EXITS AFTER FIRST FOLD
```

**Fix:**
```python
fold_results = []
for train_idx, test_idx in cv.split(X, y, groups):
    # ... train/evaluate ...
    fold_results.append({...})

# Average across all folds
auc_mean = np.mean([r['auc'] for r in fold_results])
auc_std = np.std([r['auc'] for r in fold_results])
```

---

## PART III: KNOWN BUGS & FIXES CHECKLIST

| Bug | Location | Status | Fix |
|-----|----------|--------|-----|
| **#2** | `dataset_wise_analysis_v2.py` | ❌ UNFIXED | Only 1st fold evaluated; refactor to average all folds |
| **#3** | Feature engineering | ✅ FIXED | jitter_ddp & shimmer_dda already excluded in extraction |
| **#4** | RPDE/PPE calculation | ⚠️ PROXY | Using histogram entropy (not canonical Little et al.); acceptable for now; document assumption |
| **#5** | `prepare_egemaps_training.py` | ⚠️ BRITTLE | Hard-coded `assert len(feature_cols) == 88`; remove or make flexible |
| **#6** | Feature variance | ✅ MITIGATED | Near-zero jitter/shimmer on unvoiced frames already removed via VarianceThreshold |

---

## PART IV: EXECUTION ROADMAP

### Phase 1: Validate & Clean (Do This First)

**1.1 Update `prepare_egemaps_training.py`**
```bash
# Edit: prepare_egemaps_training.py
# Change: Hard-coded paths → CLI argument or point to features/features_egemaps_8k.csv
# Test:
python scripts/prepare_egemaps_training.py
```

**1.2 Fix Bug #2 in `dataset_wise_analysis_v2.py`**
```bash
# Edit: dataset_wise_analysis_v2.py
# Find: "for train_idx, test_idx in cv.split(...)"
# Fix: Collect all fold results, average metrics
# Add: Logging for per-fold AUC values
```

**1.3 Remove Brittle Assertion in `prepare_egemaps_training.py`**
```python
# Before:
assert len(feature_cols) == 88, f"Expected 88 features, got {len(feature_cols)}"

# After (flexible):
if len(feature_cols) != 88:
    logger.warning(f"Expected 88 features, got {len(feature_cols)} — proceeding anyway")
```

---

### Phase 2: Run Initial Training on Extracted CSVs

**2.1 Train on Sustained Vowel Features**
```bash
python scripts/dataset_wise_analysis_v2.py \
  --data_file features/features_sustained_a.csv \
  --output_dir results/evaluation_sustained_a_baseline
```

**2.2 Train on eGeMAPS Features**
```bash
python scripts/dataset_wise_analysis_v2.py \
  --data_file features/features_egemaps_8k.csv \
  --output_dir results/evaluation_egemaps_baseline
```

**2.3 (Optional) Train on Comparative Features**
```bash
python scripts/dataset_wise_analysis_v2.py \
  --data_file features/features_compare_8k.csv \
  --output_dir results/evaluation_compare_baseline
```

**Output:** ROC curves, confusion matrices, per-model performance in `results/evaluation_*/`

---

### Phase 3: Feature Selection & Ranking

**3.1 Run Per-Dataset Ranking** (if script exists)
```bash
python scripts/per_dataset_ranking.py \
  --data_file features/features_sustained_a.csv
```

**Expected Output:** `results/final/ranking_sustained_a_*/merged_consensus_list.csv`

**3.2 Generate Consensus Feature Sets**
```bash
python scripts/prepare_training_datasets.py
```

**Output:** 
- `features/training_sustained_a_UNION.csv`
- `features/training_sustained_a_STABLE.csv`
- (Same for egeMAPS if ranking was run)

---

### Phase 4: Final Model Training on Refined Sets

**4.1 Re-train with Consensus Features**
```bash
python scripts/dataset_wise_analysis_v2.py \
  --data_file features/training_sustained_a_STABLE.csv
```

**4.2 Compare Results**
- Plot: STABLE vs UNION vs RAW (112 features)
- Metric: Efficiency gain (fewer features, comparable/better AUC)

---

## PART V: SCRIPT CHECKLIST & DEPENDENCIES

```
EXTRACTION PHASE (✅ DONE)
├── extract_features_sustained_a.py
│   ├── Input: Dataset WAVs (PC-GITA, Italian, Voice_Dataset)
│   └── Output: features/features_sustained_a.csv (112 feat, deduped)
├── extract_opensmile_egemaps.py (assumed to exist)
│   ├── Input: Dataset WAVs
│   └── Output: features/features_egemaps_8k.csv (88 feat)
└── [extract_compare features script, if applicable]

TRAINING PREP PHASE
├── prepare_egemaps_training.py ⚠️ NEEDS UPDATE #1
│   ├── Input: features/features_egemaps_*.csv
│   └── Output: features/training_egemaps_*.csv
└── prepare_training_datasets.py ✅ OK (no changes)
    ├── Input: features/features_*.csv + results/final/ranking_*/
    └── Output: features/training_*_{UNION,STABLE}.csv

EVALUATION PHASE
├── dataset_wise_analysis_v2.py ⚠️ NEEDS BUG FIX #2
│   ├── Input: features/features_*.csv or training_*.csv
│   └── Output: results/evaluation_*/
├── per_dataset_ranking.py (if available)
│   ├── Input: features/features_*.csv
│   └── Output: results/final/ranking_*/merged_consensus_list.csv
└── [Other analysis scripts as needed]
```

---

## PART VI: DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│  EXTRACTED CSVs (Ready to Use)                                  │
│                                                                 │
│  • features_sustained_a.csv (112 feat, ~550 rows)              │
│  • features_egemaps_8k.csv (88 feat, ~550 rows)                │
│  • features_compare_8k.csv ([N] feat, ~550 rows)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING PREP (prepare_egemaps_training.py, .._training..)    │
│                                                                 │
│  ⚠️ UPDATE PATHS & FIX BUG #5                                  │
│                                                                 │
│  Outputs:                                                       │
│  • training_egemaps_full88f.csv                                │
│  • training_egemaps_biomarker_eq6f.csv                         │
│  • training_egemaps_no_mfcc_72f.csv                            │
│  • training_sustained_a_{UNION,STABLE}.csv (if ranking done)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  MODEL TRAINING & EVALUATION (dataset_wise_analysis_v2.py)     │
│                                                                 │
│  ⚠️ FIX BUG #2 (all folds, not just first)                     │
│                                                                 │
│  Models: LR, SVM, RF, XGB, DT, Voting, Stacking               │
│  Folds: StratifiedGroupKFold (n=5, by subject_id)             │
│  Metrics: AUC, F1, MCC, Accuracy, Bootstrap CI                │
│                                                                 │
│  Outputs:                                                       │
│  • results/evaluation_*/csv_results/ (per-model metrics)       │
│  • results/evaluation_*/matrices/ (confusion matrices)         │
│  • results/evaluation_*/visualizations/ (ROC curves, etc.)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART VII: SUMMARY OF CHANGES NEEDED

### Critical (Breaks Pipeline)
1. **Fix Bug #2** in `dataset_wise_analysis_v2.py`: All folds must be evaluated, not just the first.

### High Priority (Data Integrity)
2. **Update `prepare_egemaps_training.py`**: Change hard-coded paths to match actual CSV locations or add CLI argument.
3. **Remove Brittle Assert** in `prepare_egemaps_training.py`: Make feature count flexible.

### Low Priority (Documentation)
4. **Document RPDE/PPE as Proxies**: Flag that histogram-based entropy is used, not canonical Little et al. algorithm.
5. **Update Scripts' Docstrings**: Reflect actual CSV paths and updated workflow.

---

## CONCLUSION

Your extracted feature CSVs are **clean and ready for training**. The remaining work is:

1. ✏️ Small script updates (paths, assertions, bug fixes)
2. 🔄 Rerun training scripts in the prescribed order
3. 📊 Generate ranking/consensus lists (if needed for feature selection)
4. 🎯 Final model evaluation on refined feature sets

**Next Action:** Start with Phase 1 (script updates), then run Phase 2 (initial training).

