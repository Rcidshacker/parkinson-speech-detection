"""
Gender-Stratified Classifier — PC-GITA
========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

MOTIVATION:
    PC-GITA within-dataset AUC was only 0.69 in baseline.
    Literature shows gender confounds F0/jitter/shimmer significantly.
    PC-GITA is documented as balanced: 25M/25F PD + 25M/25F HC.
    Separate male/female models remove inter-gender variance.

APPROACH:
    - Use verified gender from dataset.csv (70/100 speakers have metadata)
    - Train separate RF + XGBoost models for male and female subsets
    - 5-fold GroupKFold per gender (no speaker leakage)
    - Compare against original combined baseline (AUC=0.69)

INPUT  : features/features_sv_modeling.csv
         datasets/final/audio_pd_gita_co/dataset.csv
OUTPUT : results/gender_stratified_<timestamp>/
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
FEATURES    = os.path.join(BASE, "features", "features_sv_modeling.csv")
METADATA    = os.path.join(BASE, "data", "active", "pc_gita", "dataset.csv")
RESULTS_DIR = os.path.join(BASE, "results",
              f"gender_stratified_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
LOG_DIR     = os.path.join(BASE, "logs")

N_FOLDS     = 5
N_BOOTSTRAP = 1000
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"gender_stratified_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("gender")

def section(title):
    log.info("")
    log.info("=" * 65)
    log.info(f"  {title}")
    log.info("=" * 65)

def bootstrap_auc(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng  = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def run_cv(df_subset, label, models):
    """
    5-fold StratifiedGroupKFold on a subset.
    Returns dict: model_name -> (auc, ci_lo, ci_hi, oof_probs)
    """
    X      = df_subset[feat_cols].values
    y      = df_subset["label"].values
    groups = df_subset["speaker_id"].values

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=SEED)
    results = {}

    for name, model in models.items():
        oof_probs = np.zeros(len(y))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   model)
        ])

        for train_idx, test_idx in cv.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            oof_probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

        auc        = roc_auc_score(y, oof_probs)
        ci_lo, ci_hi = bootstrap_auc(y, oof_probs)
        results[name] = (auc, ci_lo, ci_hi, oof_probs, y)

        log.info(f"    {name:<22}: AUC={auc:.4f}  "
                 f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
                 f"(n={len(y)}, {int(y.sum())}PD/{int((1-y).sum())}HC)")

    return results


# ── LOAD DATA ────────────────────────────────────────────────
section("1. Load Data")

df   = pd.read_csv(FEATURES)
pc   = df[df["dataset"] == "pc_gita"].copy()
log.info(f"  PC-GITA rows : {len(pc)}")
log.info(f"  Speakers     : {pc['speaker_id'].nunique()}")

# ── LOAD & MAP METADATA ──────────────────────────────────────
section("2. Map Gender from Metadata")

meta_raw = pd.read_csv(METADATA)
real_headers       = meta_raw.iloc[0].tolist()
meta_raw.columns   = real_headers
meta_raw           = meta_raw.iloc[1:].reset_index(drop=True)

meta = meta_raw[meta_raw["Participant code"].str.startswith(
    ("PD", "HC"))].copy()

def to_subject_id(code):
    prefix, num = code[:2], code[2:]
    num_padded = num.zfill(4)
    return (f"AVPEPUDEA{num_padded}"  if prefix == "PD"
            else f"AVPEPUDEAC{num_padded}")

meta["subject_id"]  = meta["Participant code"].apply(to_subject_id)
meta["age"]         = pd.to_numeric(meta["Age (years)"],  errors="coerce")
meta["updrs_total"] = pd.to_numeric(meta["UPDRS III total (-)"], errors="coerce")
meta_clean = meta[["subject_id", "Gender", "age", "updrs_total"]].copy()
meta_clean.columns = ["subject_id", "gender_verified", "age", "updrs_total"]

# Merge into PC-GITA rows
pc = pc.merge(meta_clean, on="subject_id", how="left")

known   = pc[pc["gender_verified"].notna()]
unknown = pc[pc["gender_verified"].isna()]

log.info(f"  Speakers with verified gender : "
         f"{known['speaker_id'].nunique()} / "
         f"{pc['speaker_id'].nunique()}")
log.info(f"  Rows with verified gender     : {len(known)} / {len(pc)}")
log.info(f"  Rows excluded (no metadata)   : {len(unknown)}")
log.info("")
log.info(f"  Gender breakdown (known speakers):")
log.info(f"    {known.groupby(['gender_verified','label']).size().to_string()}")

# ── FEATURE COLUMNS ──────────────────────────────────────────
section("3. Feature Columns")

META_COLS = ["dataset","subject_id","language","gender","speech_type",
             "disease_label","label_binary","multiclass_label",
             "updrs_total","moca_score","meds_status","file","label",
             "speaker_id","gender_verified","age"]

feat_cols = [c for c in pc.columns
             if c not in META_COLS
             and pd.api.types.is_numeric_dtype(pc[c])]

log.info(f"  Feature columns: {len(feat_cols)}")

# ── MODELS ───────────────────────────────────────────────────
models = {
    "Random Forest"    : RandomForestClassifier(
                            n_estimators=300, random_state=SEED, n_jobs=-1),
    "XGBoost"          : XGBClassifier(
                            n_estimators=300, random_state=SEED,
                            eval_metric="logloss", verbosity=0),
    "Gradient Boosting": GradientBoostingClassifier(
                            n_estimators=200, random_state=SEED),
}

# ── BASELINE: ALL PC-GITA (no stratification) ────────────────
section("4. Baseline — All PC-GITA (known gender subset, no stratification)")
log.info("  This is the apples-to-apples comparison for gender stratification")

baseline_results = run_cv(known, "ALL", models)

# ── GENDER STRATIFIED ────────────────────────────────────────
section("5. Gender-Stratified Results")

male_df   = known[known["gender_verified"] == "M"].copy()
female_df = known[known["gender_verified"] == "F"].copy()

log.info(f"  Male   subset: {male_df['speaker_id'].nunique()} speakers  "
         f"{len(male_df)} rows  "
         f"PD={int(male_df['label'].sum())}  "
         f"HC={int((1-male_df['label']).sum())}")
log.info(f"  Female subset: {female_df['speaker_id'].nunique()} speakers  "
         f"{len(female_df)} rows  "
         f"PD={int(female_df['label'].sum())}  "
         f"HC={int((1-female_df['label']).sum())}")

log.info("")
log.info("  ── Male models ──")
male_results = run_cv(male_df, "MALE", models)

log.info("")
log.info("  ── Female models ──")
female_results = run_cv(female_df, "FEMALE", models)

# ── COMBINED PREDICTION (merge M+F back) ─────────────────────
section("6. Combined Gender-Stratified AUC (M+F merged)")
log.info("  Train male model on male data, female model on female data,")
log.info("  merge OOF predictions, compute single AUC on all known speakers")
log.info("")

combined_rows = []
for name, model in models.items():
    # Male OOF probs
    _, _, _, m_probs, m_y = male_results[name]
    # Female OOF probs
    _, _, _, f_probs, f_y = female_results[name]

    # Combine
    all_probs = np.concatenate([m_probs, f_probs])
    all_y     = np.concatenate([m_y,     f_y])

    auc           = roc_auc_score(all_y, all_probs)
    ci_lo, ci_hi  = bootstrap_auc(all_y, all_probs)

    # Baseline AUC for same subset
    base_auc = baseline_results[name][0]
    delta    = auc - base_auc

    log.info(f"  {name:<22}: AUC={auc:.4f}  "
             f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
             f"vs baseline={base_auc:.4f}  Δ={delta:+.4f}")

    combined_rows.append({
        "model"        : name,
        "stratified_auc" : round(auc, 4),
        "ci_lo"        : round(ci_lo, 4),
        "ci_hi"        : round(ci_hi, 4),
        "baseline_auc" : round(base_auc, 4),
        "delta"        : round(delta, 4),
        "male_auc"     : round(male_results[name][0],   4),
        "female_auc"   : round(female_results[name][0], 4),
    })

# ── SAVE ─────────────────────────────────────────────────────
section("7. Save Results")

results_df = pd.DataFrame(combined_rows)
out_path   = os.path.join(RESULTS_DIR, "gender_stratified_results.csv")
results_df.to_csv(out_path, index=False)
log.info(f"  Saved: {out_path}")
log.info("")
log.info("  Summary table:")
log.info(f"  {'Model':<22}  {'Baseline':>8}  "
         f"{'Stratified':>10}  {'Delta':>7}  "
         f"{'Male AUC':>9}  {'Female AUC':>11}")
log.info("  " + "─" * 75)
for _, row in results_df.iterrows():
    log.info(f"  {row['model']:<22}  {row['baseline_auc']:>8.4f}  "
             f"{row['stratified_auc']:>10.4f}  "
             f"{row['delta']:>+7.4f}  "
             f"{row['male_auc']:>9.4f}  "
             f"{row['female_auc']:>11.4f}")

section("Done")
log.info(f"  Results : {RESULTS_DIR}")
log.info(f"  Log     : {LOG_FILE}")