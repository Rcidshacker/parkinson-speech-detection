# Cross-Lingual Parkinson's Disease Detection

**Project:** Speech-Based Parkinson's Disease Detection (BE Capstone)  
**Author:** Ruchit Das (22AM1084)  
**Datasets:** PC-GITA (Spanish), VOICED (Italian)

---

## 📁 Project Structure

```
pd_analysis/
├── scripts/                    # Python analysis scripts
│   ├── balance_datasets_vowel_matched.py    # Create balanced /a/-only datasets
│   └── feature_type_analysis.py             # Feature category performance analysis
│
├── opensmile/                  # OpenSMILE feature extraction
│   └── extract_opensmile_egemaps.py         # Extract eGeMAPS (88 features)
│
├── data/                       # Balanced datasets (ready for experiments)
│   ├── features_pcgita_balanced_a_only.csv  # PC-GITA /a/ only (192 samples)
│   └── features_voiced_balanced_a_matched.csv # VOICED balanced (266 samples)
│
├── results/                    # Analysis results
│   ├── feature_type_summary.csv             # Performance by feature category
│   ├── feature_type_detailed_results.csv    # Full metrics
│   └── balanced_vowel_matched_summary.txt   # Dataset statistics
│
├── images/                     # Visualization plots
│   ├── feature_type_performance.png         # AUC/F1 by category
│   ├── confusion_matrices.png               # Cross-dataset confusion matrices
│   └── feature_count_vs_performance.png     # Features vs AUC scatter
│
└── README.md                   # This file
```

---

## 🎯 Key Findings

### Cross-Dataset Performance (ES↔IT)

| Feature Category | Features | ES→IT AUC | IT→ES AUC | Cross Avg |
|------------------|----------|-----------|-----------|-----------|
| **Phonatory (Biomarkers)** | 11 | **0.729** | **0.729** | **0.729** |
| Phonatory + F0 | 17 | 0.662 | 0.717 | 0.690 |
| All Features | 112 | 0.621 | 0.639 | 0.630 |
| MFCC | 26 | 0.412 | 0.458 | 0.435 |

### Key Insights

1. **Biomarkers outperform full feature set cross-lingually**
   - 11 features (jitter, shimmer, HNR) > 112 features
   - 17% better AUC (0.73 vs 0.63)

2. **MFCC features fail cross-lingually**
   - Within-dataset: 0.82-0.96 AUC (excellent)
   - Cross-dataset: 0.41-0.46 AUC (worse than random)

3. **More features ≠ Better cross-lingual performance**
   - Language-specific features add noise

---

## 📊 Dataset Summary

| Dataset | Vowel | Total | PD | HC | Speakers |
|---------|-------|-------|-----|-----|----------|
| PC-GITA (Spanish) | /a/ only | 192 | 96 | 96 | 64 (32 PD + 32 HC) |
| VOICED (Italian) | /a/ only | 266 | 133 | 133 | 266 |

Both datasets use vowel /a/ for controlled cross-lingual comparison.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn librosa soundfile parselmouth tqdm
```

### 1. Data Balancing (Already Done)
```bash
python scripts/balance_datasets_vowel_matched.py
```
Creates balanced /a-only datasets from original feature CSVs.

### 2. Feature Type Analysis
```bash
python scripts/feature_type_analysis.py
```
Analyzes performance by feature category (Phonatory, MFCC, Spectral, etc.)

### 3. OpenSMILE eGeMAPS Extraction (On Local Machine)
```bash
pip install opensmile
# Update paths in extract_opensmile_egemaps.py
python opensmile/extract_opensmile_egemaps.py
```
Extracts validated 88-feature eGeMAPS set from raw audio.

---

## 📋 Task Status (From Mentor Meeting 2026-03-23)

| # | Task | Status |
|---|------|--------|
| 1 | Generate balanced PC-GITA subset (vowel-matched) | ✅ Done |
| 2 | RF feature importance validation | ⏳ Pending |
| 3 | Feature type-wise performance analysis | ✅ Done |
| 4 | CNN with 11×10 feature matrix | ⏳ Pending |
| 5 | OpenSMILE eGeMAPS extraction | ✅ Script ready |
| 6 | Report writing | ⏳ Pending |

---

## 📚 References

- **PC-GITA:** Orozco et al., 2014 - Spanish PD speech corpus
- **VOICED:** Fabbri et al., 2021 - Italian PD speech corpus  
- **eGeMAPS:** Eyben et al., 2015 - Geneva Minimalistic Acoustic Parameter Set
- **OpenSMILE:** Eyben et al., 2010 - Speech and Music Interpretation toolkit

---

## 📝 Thesis Narrative

```
1. Extract features → works within-language ✓
        ↓
2. Feature selection → performance drops ✓
        ↓
3. Feature type analysis → biomarkers cross-lingual, spectral fails ✓
        ↓
4. Conclude: vocal fold features invariant, vocal tract features language-dependent ✓
        ↓
5. Next: OpenSMILE eGeMAPS or CNN approach
```

---

**Generated:** 2026-03-24
