# Project Stages

## Stage 1 - Baseline (DONE)
Existing pipeline: feature extraction (ComParE 88-feat), classical ML (SVM, RF, LR).
See `stage_1_baseline/README.md` for details and open items.

## Stage 2 - Feature Grouping (IN PROGRESS)
Split ComParE features into semantic groups (MFCC, energy, spectral, etc.).
Run 3-model matrix per group to identify which feature families drive PD detection.
See `stage_2_feature_grouping/README.md` for methodology.

## Stage 3 - Foundation Model (PLANNED)
Fine-tune wav2vec 2.0 / HuBERT for language-independent PD detection (target AUC >0.90).