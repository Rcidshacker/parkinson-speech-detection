# Speech-Based Parkinson's Disease Detection (BE Capstone)

## What It Does
This project investigates whether a machine learning model trained on Parkinson's Disease (PD) speech recordings in one language can detect PD in a completely different language — a cross-lingual generalization problem. It extracts acoustic features from sustained vowel recordings across three feature regimes (handcrafted phonatory biomarkers, eGeMAPS, and ComParE16 functionals), trains a battery of classical ML classifiers, then evaluates them in a 3x3 cross-dataset generalization matrix (Spanish PC-GITA vs. English Voice_Dataset) to measure how well each acoustic feature family transfers across linguistic and recording boundaries. Stage 2 applies LLD-level Cepstral Mean Normalization to strip microphone EQ differences while deliberately preserving frame-level variance as the vocal tremor signal. Stage 3 (planned) will fine-tune wav2vec 2.0 / HuBERT to push cross-lingual AUC above 0.90.

## Technical Architecture
**Data flow:**
```
Dataset/ (raw WAV) → audio preprocessing (resample, RMS-normalize)
    → OpenSMILE / librosa feature extraction → features/*.csv
    → prepare_*_training.py (dataset filter, dedup, ID neutralization)
    → dataset_wise_analysis_v2.py (7-model eval, 5-fold StratifiedGroupKFold)
    → dataset_matrix.py / 03_group_matrix.py (3x3 cross-dataset matrix)
    → results/<run>/ (CSVs, ROC curves, heatmaps, JSON reports, xlsx)
```

Three datasets: PC-GITA (Spanish, 300 rows, 150 PD / 150 HC), Voice_Dataset (English/Kaggle, 567 unique rows after deduplication of 470 exact duplicate pairs, 280 PD / 287 HC), Italian Parkinson's Voice (99 rows, excluded from cross-lingual analysis — scope mismatch and recording profile incompatibility).

Stage 1 trains 7 models (LR, SVM-RBF, RandomForest, DecisionTree, XGBoost, VotingClassifier, StackingClassifier) with 5-fold StratifiedGroupKFold on subject_id. All preprocessing (StandardScaler, SimpleImputer, optional SelectKBest) is fitted strictly inside CV folds — never on full data.

Stage 2 uses capacity-constrained models (L1-LR C=0.1, constrained RF max_depth=3, linear SVM C=0.05 with RobustScaler) to prevent noise-floor memorization on 52 CMN features. A three-stage cascade (L1-LR at 70% confidence → constrained RF at 65% → gradient boosting always-output) routes samples by model confidence, reporting percentage of test cases handled per stage.

CMN normalization applies StandardScaler(with_std=False) per utterance before computing MFCC functionals — strips static microphone EQ across recording environments but preserves frame-level variance, because std(MFCC) is precisely where vocal tremor manifests. The amean feature is excluded downstream because it is exactly 0.0 post-normalization by construction.

## Stack
- **Language:** Python 3.13 (Windows, isolated venv)
- **Audio processing:** librosa 0.11, soundfile, OpenSMILE 2.6 (eGeMAPS, ComParE_2016), scipy
- **ML:** scikit-learn (StratifiedGroupKFold, Pipeline, RobustScaler, SelectKBest, VotingClassifier, StackingClassifier), XGBoost 3.2
- **Numerical / analysis:** NumPy 2.4, pandas 2.3, SciPy (bootstrap confidence intervals, 1000 iterations)
- **Visualization:** matplotlib (Agg headless backend), seaborn
- **Parallelism:** joblib (Parallel/delayed for bootstrap and feature extraction)
- **Reporting:** xlsxwriter 3.2 (presentation-ready xlsx with embedded plots)
- **Datasets:** PC-GITA (Ibáñez et al. 2014), UCI/Kaggle Voice_Dataset (extended), Italian Parkinson's Voice and Speech corpus

## Most Impressive Parts

1. **Forensic audit trail with documented root-cause fixes.** The codebase underwent a three-phase audit identifying three critical bugs: `prepare_egemaps_training.py` silently output a byte-for-byte copy of the source CSV instead of filtering by dataset label (rendering all eGeMAPS cross-dataset comparisons methodologically void); the ensemble builder was hardcoded to `[:3]`, permanently excluding XGBoost from VotingClassifier and StackingClassifier; and self-domain evaluation used only the first CV fold instead of full cross-validation. All fixes are documented with root-cause chains, before/after code, and verification steps in `SINGLE_TRUTH_REPORT.md` — unusually rigorous for a BE capstone.

2. **Domain-informed per-utterance CMN that preserves the pathological signal.** Stage 2 applies mean-only normalization (not CMVN) per audio file — subtracting utterance mean to strip static microphone EQ, but deliberately skipping variance normalization because frame-level MFCC standard deviation is the vocal tremor biomarker. The amean column is then excluded from all downstream features with explicit justification (it is always exactly 0.0 post-normalization). This is a principled feature engineering decision grounded in understanding of both the acoustic pathology and the recording-condition confound.

3. **Parameterized 3x3 cross-dataset generalization matrix with feature-group sub-analysis.** The `03_group_matrix.py` script accepts a `--feature_group` CLI argument that filters the 6373-feature ComParE space to a named semantic group (MFCC, jitter, shimmer, spectral, voicing, energy, ZCR, structural CMN, temporal CMN) before any model fitting — with anti-leakage guarantees unchanged. Each of the 9 matrix cells (diagonal via 5-fold CV, off-diagonal as true zero-overlap cross-dataset) includes 1000-bootstrap CIs. This single parameterized design answers "which feature families generalize across languages?" across all groups in one batch run, with findings immediately actionable: MFCC features achieve within-dataset AUC up to 0.97 but degrade sharply cross-lingually, while voice-quality biomarkers generalize more consistently.

## Results / Metrics / Outputs

**Stage 1 Baseline (ComParE16, full ~6373 features, 8 kHz):**
- Within-dataset: PC-GITA (ES) self AUC 0.813, Voice_Dataset (EN) self AUC 0.986
- Cross-dataset: ES→EN AUC 0.642, EN→ES AUC 0.664
- Combined→combined AUC 0.928
- Best cross-lingual result reported (Top-9 features + LR): mean AUC ~0.842

**Stage 1 eGeMAPS (83 features) — best models within-dataset:**
- Voice_Dataset: SVM AUC 0.994, VotingEnsemble AUC 0.994, XGBoost AUC 0.978
- PC-GITA: XGBoost AUC 0.893, RandomForest AUC 0.863, StackingEnsemble AUC 0.817
- Cross-dataset (eGeMAPS): ES→EN AUC 0.604 (best Stacking), EN→ES AUC 0.679 (best SVM)

**Stage 2 CMN (52 features, capacity-constrained models):**
- Full CMN matrix: ES→EN AUC 0.752, EN→ES AUC 0.652
- Structural MFCC CMN group only: ES→EN AUC 0.733, EN→ES AUC 0.651
- Temporal ZCR CMN group only: ES→EN 0.622, EN→ES 0.633

**Stage 2 ComParE feature group comparison (cross-lingual AUC):**
- MFCC group: ES→EN 0.637, EN→ES 0.675
- Jitter group: ES→EN 0.614, EN→ES 0.575
- Spectral/MFCC sub-analysis: ES→EN 0.686, EN→ES 0.701
- Prosody/Voice Quality sub-analysis: ES→EN 0.685, EN→ES 0.620

**Dataset sizes:** 867 total training rows (300 PC-GITA + 567 Voice_Dataset after dedup), 430 PD / 437 HC. Voice_Dataset original: 1037 raw rows; 470 exact duplicate pairs removed via MD5 file hashing.

**Shipped artifacts:** Per-run timestamped directories containing ROC curves, confusion matrices, AUC heatmaps, per-model JSON reports, `comprehensive_analysis_report_v2.xlsx`, and `Comprehensive_CMN_Results_Report.xlsx` (presentation-ready with embedded plots, AUC matrices, cascade results, sub-analysis, methodology).

## Status
Working demo

## Estimated Timeline
Active development ran from approximately **March 2026** (Stage 1 baseline, full forensic audit documented 2026-03-28, initial evaluation runs dated 2026-03-27) through **April 2026** (Stage 2 CMN pipeline, feature group matrix runs dated 2026-04-06, extraction log dated 2026-04-06). Stage 3 (fine-tuning wav2vec 2.0 / HuBERT, target AUC >0.90) is planned but not started. Total active development approximately 5–6 weeks. Author: Ruchit Das (22AM1084), supervised by Prof. Pramod Kachare, RAIT, DY Patil University, Navi Mumbai.
