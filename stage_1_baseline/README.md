# Stage 1 - Baseline

This stage documents the existing work. No files were moved.

## What exists
- `scripts/` - feature extraction and ML pipeline scripts
- `features/` - extracted feature CSVs (PC-GITA + VOICED, 867 rows)
- `results/` - per-model evaluation outputs
- `Dataset/` - raw audio organized by dataset/class

## Key facts (as of March 2026)
- 867 rows: 300 PC-GITA (Spanish) + 567 VOICED (Italian)
- 430 PD / 437 HC, zero Italian rows excluded
- Feature count: 83 (after VarianceThreshold, not 88)
- Best classical result: Top-9 cross-lingual features + LR - mean AUC ~0.842
- Foundation model target: wav2vec 2.0 / HuBERT (>0.90 AUC)

## Open items
- [ ] Verify path mismatch in `_batch_run.py` results glob before running
- [ ] Add docstring disambiguation to `extract_features_sustained_a_16k.py`