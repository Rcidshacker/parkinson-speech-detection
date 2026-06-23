# Script Fixes & Changes

## 1. `prepare_egemaps_training.py` — CRITICAL

### Change 1: Remove Hard-Coded Assert (Line ~60)

**Current (WRONG):**
```python
assert len(feature_cols) == 88, f"Expected 88 features, got {len(feature_cols)}"
```

**Fix (Flexible):**
```python
if len(feature_cols) != 88:
    logger.warning(f"Expected 88 features, got {len(feature_cols)} — proceeding anyway")
```

---

### Change 2: Add CLI Argument for CSV Input

**Current (WRONG):**
```python
PCGITA_CSV = os.path.join(INPUT_DIR, "features_egemaps_pcgita.csv")
VOICED_CSV = os.path.join(INPUT_DIR, "features_egemaps_voiced.csv")
```

**Problem:** Script assumes separate `pcgita` and `voiced` CSVs. Your extraction produced a single `features_egemaps_8k.csv`.

**Fix (Option A — Recommended):**
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare eGeMAPS training datasets")
    parser.add_argument('--egemaps_csv', default='features/features_egemaps_8k.csv',
                        help='Path to the eGeMAPS features CSV')
    parser.add_argument('--output_dir', default='features/opensmile',
                        help='Output directory for training CSVs')
    args = parser.parse_args()

    print("=" * 65)
    print("  Prepare eGeMAPS Training Datasets")
    print("=" * 65)

    # 1. Load the single eGeMAPS CSV
    if not os.path.exists(args.egemaps_csv):
        print(f"[ERROR] Not found: {args.egemaps_csv}")
        sys.exit(1)

    df = pd.read_csv(args.egemaps_csv)
    
    print(f"\nLoaded eGeMAPS features: {len(df)} rows")
    print(f"  PD: {(df['label_binary']==1).sum()}")
    print(f"  HC: {(df['label_binary']==0).sum()}")

    # 2. Identify feature columns (same 88)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    if len(feature_cols) != 88:
        print(f"[WARNING] Expected 88 features, got {len(feature_cols)} — proceeding anyway")
    
    biomarker_eq, mfcc_feats, non_mfcc = categorise_features(feature_cols)

    # ... rest of the function remains the same ...
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Save outputs
    out_full = os.path.join(args.output_dir, "training_egemaps_full88f.csv")
    df[META_COLS + feature_cols].to_csv(out_full, index=False)
    print(f"\n[1] Saved: {out_full}")

    # ... etc for other outputs ...
```

**Fix (Option B — Simpler, No CLI):**
```python
# At the top, after BASE definition:
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

# Replace hard-coded paths:
EGEMAPS_CSV = os.path.join(BASE, "features", "features_egemaps_8k.csv")  # ← Single file

# In main():
if not os.path.exists(EGEMAPS_CSV):
    print(f"[ERROR] Not found: {EGEMAPS_CSV}")
    sys.exit(1)

df = pd.read_csv(EGEMAPS_CSV)  # ← Load single file, not two
```

**How to run (after Option A fix):**
```bash
python scripts/prepare_egemaps_training.py --egemaps_csv features/features_egemaps_8k.csv
```

---

## 2. `dataset_wise_analysis_v2.py` — BUG FIX #2

### Issue: Only First Fold Evaluated

**Current (WRONG):**
```python
# Lines ~250–300 (cross-dataset eval section)
fold_count = 0
for train_idx, test_idx in cv.split(X, y, groups):
    fold_count += 1
    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]
    
    # ... train models, evaluate ...
    
    if fold_count == 1:
        break  # ❌ EXITS AFTER FIRST FOLD!
```

**Fix:**
```python
# Collect results from all folds
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    print(f"  Fold {fold_idx+1}/{N_FOLDS}...")
    
    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]
    
    # ... train models, get results ...
    
    fold_results.append({
        'fold': fold_idx,
        'auc': auc_value,
        'f1': f1_value,
        'mcc': mcc_value,
        # ... all metrics ...
    })

# After all folds, compute mean ± std
if fold_results:
    auc_mean = np.mean([r['auc'] for r in fold_results])
    auc_std = np.std([r['auc'] for r in fold_results])
    f1_mean = np.mean([r['f1'] for r in fold_results])
    f1_std = np.std([r['f1'] for r in fold_results])
    
    logger.info(f"  Cross-fold AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    logger.info(f"  Cross-fold F1:  {f1_mean:.4f} ± {f1_std:.4f}")
```

---

## 3. `prepare_training_datasets.py` — NO CHANGES (Already Dynamic)

This script is **already well-designed** for your setup. It:
- ✅ Dynamically finds `results/final/ranking_*/merged_consensus_list.csv`
- ✅ Loads raw CSVs from `features/features_*.csv`
- ✅ Filters by consensus features + stability
- ✅ Generates training CSVs

**No changes needed**. Use it after `per_dataset_ranking.py` completes.

---

## 4. Other Scripts — Document Assumptions

### `dataset_wise_analysis_v2.py` — Docstring Update

**Current (Top of file):**
```python
"""
INPUTS (features/):
    features_sustained_a.csv         data file
    features_egemaps_8k.csv          data file
"""
```

**Update to reflect Bug #4 (RPDE/PPE):**
```python
"""
FEATURE NOTES:
    [FN4] RPDE & PPE use histogram entropy (not canonical Little et al.)
         This is acceptable for baseline comparisons but should be noted
         when publishing. Canonical algorithms (e.g. from the Praat papers)
         would require different math for f0-variability quantization.
"""
```

---

## Execution Order (Summary)

```
1. Fix prepare_egemaps_training.py (Option A or B)
2. Fix dataset_wise_analysis_v2.py (Bug #2)
3. Run: python scripts/prepare_egemaps_training.py
   → Outputs: training_egemaps_*.csv
4. Run: python scripts/dataset_wise_analysis_v2.py --data_file features/features_sustained_a.csv
   → Outputs: results/evaluation_sustained_a_baseline/
5. Run: python scripts/dataset_wise_analysis_v2.py --data_file features/features_egemaps_8k.csv
   → Outputs: results/evaluation_egemaps_baseline/
6. (Optional) Run: python scripts/per_dataset_ranking.py
7. (Optional) Run: python scripts/prepare_training_datasets.py
8. (Optional) Re-run analysis on training_*_STABLE.csv
```

---

## Quick Reference: What Each Script Does

| Script | Input | Output | Changes? |
|--------|-------|--------|----------|
| `extract_features_sustained_a.py` | WAVs | features_sustained_a.csv | ✅ None |
| `prepare_egemaps_training.py` | features_egemaps_*.csv | training_egemaps_*.csv | ⚠️ Fix #1, #2 |
| `dataset_wise_analysis_v2.py` | features_*.csv or training_*.csv | results/evaluation_*/ | ⚠️ Fix Bug #2 |
| `prepare_training_datasets.py` | ranking results + raw CSVs | training_*_{UNION,STABLE}.csv | ✅ None |
| `per_dataset_ranking.py` | features_*.csv | results/final/ranking_*/ | ✅ None (external) |

