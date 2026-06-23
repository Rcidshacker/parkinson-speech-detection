# Stage 2 - Feature Grouping

## Goal
Identify which ComParE feature groups (MFCC, energy, spectral, voicing, etc.) are most predictive of PD, independently of language.

## Pipeline
1. `01_inspect_groups.py` - Discover LLD prefixes from ComParE CSV, print unique prefix list
2. `02_prepare_group_csvs.py` - Split master CSV into per-group CSVs using `groups.json` config
3. `03_group_matrix.py` - Run SVM / RF / LR on each group CSV, output results to `results/`

## Config
`group_definitions/groups.json` maps group names to column prefix patterns. Edit this to add/remove groups without touching scripts.

## Findings Log
_(Fill in as experiments run)_