# Modification Log — 03_group_matrix.py

**Date:** 2026-04-06  
**Source:** `scripts/pipeline/dataset_matrix.py`  
**Output:** `stage_2_feature_grouping/scripts/03_group_matrix.py`

---

## AGENT 1 — Audit Report

### Data Flow
```
[CSV load] → [drop ALL_META cols] → [select_dtypes(numeric)] → [X_all_df DataFrame]
           → [feat_names list]
           → [GROUP FILTER injected here] ← NEW
           → [X_all = X_all_df.values]  ← numpy conversion after filter
           → [dataset splits by mask]
           → [run_within_cv / run_cross_dataset] → [Pipeline(imputer+scaler+model)]
```

### Feature Column Extraction (Primary Injection Point)
- **Lines 259–263 (original):** `X_all` is a DataFrame until `.values` — filter inserted between these two steps
- `drop_cols` uses `ALL_META` (14-column list) to exclude metadata; remaining numeric columns = features
- `feat_names = X_all_df.columns.tolist()` — filter narrows this list before numpy conversion

### Anti-Leakage Checkpoints (Preserved Unchanged)
1. **StratifiedGroupKFold** — `run_within_cv()`, `groups=groups_all[tr_mask]` (subject_id)
2. **StandardScaler** — inside `Pipeline` in `make_model()`, fitted per fold (CV) or per train set (cross-dataset); never on full data
3. **SimpleImputer** — also inside `Pipeline`, same guarantees
4. **No VarianceThreshold** — this script does not use it (only `dataset_wise_analysis_v2.py` does)

### Hidden Assumptions Checked
- [x] No hardcoded feature count — `len(feat_names)` is logged dynamically
- [x] Feature count log message uses `{len(feat_names)}` — correct after filtering
- [x] Sub-analysis (Section 6) uses `feat_names.index(f)` — safe after filter since `feat_subset` is derived from the already-filtered `feat_names`
- [x] `TIMESTAMP` and `RUN_DIR` were set at module level before args could exist — **moved to after argparse**

---

## AGENT 2 — Filter Design

### Data Source
`groups.json` (not `compare_group_columns.txt` — that file was not yet generated).  
Format: `{ "group_name": ["prefix1", "prefix2", ...] }`.  
Matching: case-insensitive `str.startswith()` on each feature column name.

### `parse_group_columns()` Signature
```python
def parse_group_columns(
    group_name: str, groups_json_path: str, all_feat_names: List[str]
) -> List[str]
```
- Raises `FileNotFoundError` if groups.json missing
- Raises `ValueError` with available groups list if group_name unknown
- Returns empty list (not an error) if prefixes match nothing — caller checks and exits with log message

### Anti-Leakage Proof
- Filter applied at column selection time (before DataFrame → numpy) ✓
- Filter applied before any Pipeline.fit() call ✓
- `subject_id`, `label`, `dataset` already excluded by `drop_cols` before filter ✓
- Scaler/imputer inside Pipeline — refitted per fold/per train set regardless of feature count ✓

---

## AGENT 3 — Changes Made

| # | Location | Change |
|---|----------|--------|
| 1 | Imports | Added `argparse`, `json`, `from typing import List` |
| 2 | CONFIG | `BASE_OUTDIR` → `stage_2_feature_grouping/results/` |
| 3 | CONFIG | Added `GROUPS_JSON_PATH` constant |
| 4 | After CONFIG | Added `parse_group_columns()` function |
| 5 | After parse_group_columns | Added `_load_group_choices()` + argparse block with `--feature_group` |
| 6 | Output setup | Moved `TIMESTAMP`/`RUN_DIR` to after argparse; `RUN_DIR` uses `group_suffix` |
| 7 | Section 1 LOAD | Changed `X_all = df.drop(...)` → `X_all_df = df.drop(...)` (keep as DataFrame) |
| 8 | Section 1 LOAD | Injected group filter block between `feat_names` extraction and `.values` |
| 9 | Section 6 | Dynamic `all_feats_label` (shows group name when filtered) |
| 10 | Section 6 | Dynamic `sub_colors` palette (avoids KeyError on non-standard subset names) |
| 11 | Plots (ax1, suptitle) | Added `group_title_str` to titles |
| 12 | Section 8 summary | `{sname:<22}` widened to `{sname:<30}` for longer dynamic labels |
| 13 | Docstring | Updated to describe `--feature_group` usage and output path |

**Not changed:** model hyperparameters, CV strategy, Pipeline structure, output CSV/plot format,
all logging statements, cascade thresholds, bootstrap CI, dataset split logic.

---

## AGENT 4 — Validation Checklist

- [x] `import argparse`, `import json`, `from typing import List` present
- [x] `parse_group_columns()` raises `FileNotFoundError` (groups.json missing) and `ValueError` (group not found) with informative messages including available group list
- [x] Zero-match guard: `if len(group_cols) == 0: log.error(...); sys.exit(1)`
- [x] `choices=_valid_groups` — argparse rejects invalid group names at startup
- [x] Output dir: `stage_2_feature_grouping/results/matrix_<group>_<TIMESTAMP>/`
- [x] Anti-leakage patterns untouched (StratifiedGroupKFold, Pipeline, scaler inside fit)
- [x] Sub-analysis `feat_idx` uses `feat_names.index(f)` — safe because `feat_subset` derived from already-filtered `feat_names`

### Test Invocation

```powershell
$PY  = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\venv\Scripts\python.exe"
$S2  = "C:\Users\Lenovo\Desktop\Code\2026\BE mini project\stage_2_feature_grouping"

# Smoke test — check help text
& $PY "$S2\scripts\03_group_matrix.py" --help

# Run with Voice Quality (jitter + shimmer groups)
& $PY "$S2\scripts\03_group_matrix.py" --feature_group jitter
& $PY "$S2\scripts\03_group_matrix.py" --feature_group shimmer

# Full run (all 6373+ features — same as original dataset_matrix.py)
& $PY "$S2\scripts\03_group_matrix.py"
```

### Expected Behaviour per Group

| Group | Expected columns | Output dir |
|-------|-----------------|------------|
| `mfcc` | ~hundreds (mfcc* prefix) | `matrix_mfcc_<ts>/` |
| `jitter` | ~dozens (jitter* prefix) | `matrix_jitter_<ts>/` |
| `shimmer` | ~dozens (shimmer* prefix) | `matrix_shimmer_<ts>/` |
| `voicing` | ~dozens (voicingFinalUnclipped*, F0finEnv*, F0raw*) | `matrix_voicing_<ts>/` |
| *(none)* | All 6373+ features | `matrix_full_<ts>/` |

Smaller groups run faster than full 6373-feature run. If group prefix matches zero columns,
the script logs `[ERROR]` and exits cleanly — no partial output directory left open.
