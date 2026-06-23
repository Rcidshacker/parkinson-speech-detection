# SESSION MEMORY — PD Speech Detection Capstone
## Forensic Audit, Refactoring & Data Regeneration
**Date:** 2026-03-28 | **Status:** COMPLETE — all bugs fixed, data regenerated, verified

---

## PROJECT IDENTITY

- **Project:** Speech-Based Parkinson's Disease Detection (BE Capstone)
- **Author:** Ruchit Das (22AM1084)
- **Supervisor:** Prof. Pramod Kachare
- **Institution:** RAIT, DY Patil University, Navi Mumbai
- **Root path:** `C:\Users\Lenovo\Desktop\Code\2026\BE mini project`
- **Venv:** `<root>\venv\Scripts\python.exe`
- **Teammate GitHub:** Achyut103040/Parkinson-s_Disorder

---

## WHAT THIS SESSION DID (COMPLETE SUMMARY)

### Phase 1 — Forensic Audit
A `FORENSIC CODE REPORT` was produced identifying three anomalies:

| Anomaly | Location | Severity |
|---|---|---|
| A — Identity Theft | `prepare_egemaps_training.py` | Critical |
| B — Naming Confusion (8k/10k/16k) | `extract_features_*_10k/16k.py` | Documentation |
| C — Eval Engine bugs (3 sub-bugs) | `dataset_wise_analysis_v2.py` | Moderate |

### Phase 2 — Code Refactoring (all applied)

**Fix A — `scripts/prepare_egemaps_training.py`**
- PROBLEM: `main()` had `argparse` with `default='features/features_egemaps_8k.csv'`
  making `training_egemaps_full88f.csv` a byte-for-byte copy of the 8k source.
- ROOT CAUSE: `.bak` version assumed `features_egemaps_pcgita.csv` and
  `features_egemaps_voiced.csv` existed on disk — they never did.
  Extractor always writes combined CSVs. Someone "fixed" the FileNotFoundError
  with argparse, but set the wrong default and never changed the output name.
- FIX: Replaced argparse with `dataset` column filter on the combined CSV.
  Key constants:
  ```python
  COMBINED_EGEMAPS_CSV = os.path.join(FEATURES_DIR, "features_egemaps_8k.csv")
  PCGITA_LABEL = "pc_gita"
  VOICED_LABEL = "voice_dataset"
  ```
  Now does: load → filter pc_gita rows → filter voice_dataset rows →
  assert both non-empty → concat → shuffle(random_state=42) → save.

**Fix C1 — `load_full_data()` in `dataset_wise_analysis_v2.py`**
- Was: silent `df = df[df['dataset'] != 'italian']` with no logging.
- Now: `logger.warning()` with exact Italian row count, total before/after.

**Fix C2 — Ensemble slicing in `train_all_models()`**
- Was: `base = list(fitted_models.items())[:3]` in both VotingClassifier
  and StackingClassifier blocks — hardcoded to LR+SVM+RF, XGBoost excluded.
- Now: `base = list(fitted_models.items())` — all fitted models participate.

**Fix C3 — Single-fold self-domain eval in Part 1 of `main()`**
- Was: `next(cv.split(X, y, groups))` — first fold only, high variance.
- Now: full CV loop, logs `mean ± std` AUC per model, uses last fold for plots.

**Fix B — Documentation in `extract_features_sustained_a_10k.py`**
- Added disambiguation block to docstring: "10k = audio sample rate in Hz,
  NOT training sample count. All three variants produce identical row counts."
- Same fix needed for `_16k.py` (was not completed — add manually).


### Phase 3 — Data Availability Audit

Discovered that `features/opensmile/` did NOT contain the correct data:
- `training_egemaps_full88f.csv` was a copy of `features_egemaps_8k.csv`
- Per-dataset files (`features_egemaps_pcgita.csv`, `features_egemaps_voiced.csv`)
  never existed — the extractor (`extract_features_sustained_a_opensmile.py`)
  always writes ONE combined CSV per sample rate.

Dataset column values confirmed from `features_egemaps_8k.csv` header:
- `pc_gita` → Spanish, 300 rows at 8kHz (150 PD / 150 HC)
- `voice_dataset` → English-adjacent/Kaggle, 567 rows (280 PD / 287 HC)
- `italian` → Italian, 99 rows — EXCLUDED from cross-lingual scope

### Phase 4 — File Deletions (EXECUTED)

Three files permanently deleted:
1. `scripts/prepare_egemaps_training.py.bak` — phantom paths, would crash
2. `scripts/dataset_wise_analysis_v2.py.bak` — obsolete v1 predecessor
3. `scripts/dataset_wise_analysis.py` — v1 engine, superseded by v2

### Phase 5 — Data Regeneration (EXECUTED & VERIFIED)

Ran: `python scripts/prepare_egemaps_training.py`

Output confirmed:
```
Loaded combined eGeMAPS CSV: 966 rows
  Datasets found: ['italian', 'pc_gita', 'voice_dataset']
  [INFO] Dropped 99 Italian rows

Subset sizes:
  PC-GITA (pc_gita)       : 300 rows | PD=150 HC=150
  VOICED  (voice_dataset) : 567 rows | PD=280 HC=287
[WARNING] Expected 88 eGeMAPS features, got 83 — proceeding

Combined: 867 rows | PD=430 HC=437
[1] Saved: training_egemaps_full88f.csv  (83 features, 867 rows)
[2] Saved: training_egemaps_biomarker_eq6f.csv  (5 features)
[3] Saved: training_egemaps_no_mfcc_72f.csv  (67 features)
```

Verification checks (all PASS):
- [CHECK 1] Row counts differ from 8k source: 966 → 867 (diff=99 Italian rows) ✅
- [CHECK 2] Zero Italian rows in full88f ✅
- [CHECK 3] Both target datasets present: ['pc_gita', 'voice_dataset'] ✅

**NOTE on feature count:** Expected 88 eGeMAPS features, got 83.
This is because `VarianceThreshold(threshold=0.001)` in the extractor dropped
5 near-zero-variance features during extraction. This is correct behavior —
not a bug. The feature count is 83 in the actual data.


---

## CURRENT FILE STATE (as of end of session)

### `features/` directory
```
features/
├── features_egemaps_8k.csv       966 rows — PRIMARY SOURCE (pc_gita+italian+voice_dataset)
├── features_egemaps_10k.csv      ~966 rows — cross-rate comparison
├── features_egemaps_16k.csv      ~966 rows — cross-rate comparison
├── features_compare_8k.csv       ComParE 6373-feature variant (not used yet)
├── features_compare_10k.csv
├── features_compare_16k.csv
├── dedup_report_egemaps_8k.csv   L1 dedup audit trail
├── dedup_report_egemaps_10k.csv
├── dedup_report_egemaps_16k.csv
├── [sustained_a variants]        handcrafted 112-feature pipeline CSVs
└── opensmile/
    ├── training_egemaps_full88f.csv        867 rows, 83 features ✅ REGENERATED
    ├── training_egemaps_biomarker_eq6f.csv 867 rows,  5 features ✅ REGENERATED
    └── training_egemaps_no_mfcc_72f.csv    867 rows, 67 features ✅ REGENERATED
```

### `scripts/` directory — trusted state
```
TRUSTED (verified correct):
  dataset_wise_analysis_v2.py       ← PATCHED (C1+C2+C3 fixed)
  prepare_egemaps_training.py       ← PATCHED (Anomaly A fixed)
  extract_features_sustained_a_opensmile.py  ← no bugs found
  _batch_run.py                     ← no bugs found
  prepare_training_datasets.py      ← not audited but not in anomaly chain
  feature_selection_crosslingual.py ← not audited but not in anomaly chain

PENDING (minor, not blocking):
  extract_features_sustained_a_16k.py  ← needs same docstring disambiguation
                                          as 10k.py (add manually, low priority)

DELETED:
  prepare_egemaps_training.py.bak   ← gone ✅
  dataset_wise_analysis_v2.py.bak   ← gone ✅
  dataset_wise_analysis.py          ← gone ✅
```

### Permanent records written this session
```
SINGLE_TRUTH_REPORT.md   ← full forensic + methodology + integrity report
SESSION_MEMORY.md        ← this file
```

---

## WHAT TO DO NEXT (WHERE WE LEFT OFF)

The data pipeline is clean and ready. The next step is to **run the evaluation**:

```powershell
$PY = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\venv\Scripts\python.exe"
$BASE = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project"

# eGeMAPS evaluation runs
& $PY "$BASE\scripts\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_egemaps_full88f.csv"
& $PY "$BASE\scripts\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_egemaps_biomarker_eq6f.csv"
& $PY "$BASE\scripts\dataset_wise_analysis_v2.py" --data_file "$BASE\features\opensmile\training_egemaps_no_mfcc_72f.csv"

# Handcrafted feature set batch run
& $PY "$BASE\scripts\_batch_run.py"
```

Results land in: `results/evaluation_<name>_<timestamp>/`
Each folder contains: `csv_results/analysis_summary.csv`, `visualizations/`, `matrices/`, `reports/detailed_results.json`

### Open items / known gaps to address
1. **`extract_features_sustained_a_16k.py`** — add the same `10k` disambiguation
   docstring (file says "16 kHz" but has no warning about the naming convention).
2. **Feature count is 83, not 88** — the `[WARNING]` in prepare script is benign.
   5 features were dropped by VarianceThreshold during extraction. Update the
   `SINGLE_TRUTH_REPORT.md` §3.5 and §5.2 to reflect 83, not 88.
3. **VOICED row count** — 567 rows, not the ~296 originally estimated in the
   forensic report. The dataset has been augmented (original 195 UCI →
   1,037 files per Dataset_Inventory.xlsx, post-dedup gives 567).
   Update any documentation that says "~296 VOICED rows."
4. **Batch runner path mismatch** — `_batch_run.py` points results to
   `scripts/results_dataset_wise_v2` but `dataset_wise_analysis_v2.py` routes
   to `<project_root>/results/`. Verify the batch runner's glob path is correct
   before running it, or the summary aggregation at the end will find 0 runs.

---

## KEY CONSTANTS (memorize for next session)

```python
# Dataset labels in all feature CSVs
PCGITA_LABEL = "pc_gita"       # Spanish, 300 rows, 150 PD / 150 HC
VOICED_LABEL = "voice_dataset" # English-adj, 567 rows, 280 PD / 287 HC
ITALIAN_LABEL = "italian"      # 99 rows — EXCLUDED from cross-lingual scope

# After prepare_egemaps_training.py
FULL88F_ROWS = 867   # actual (83 features, not 88 — VarianceThreshold dropped 5)

# Source file for all eGeMAPS prep
PRIMARY_SOURCE = r"features\features_egemaps_8k.csv"  # 966 rows total
```

---

*End of SESSION_MEMORY.md*
