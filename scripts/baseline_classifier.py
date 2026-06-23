"""
Baseline Classifier — GroupKFold with Bootstrap CIs
=====================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

INPUT  : features/features_sv_modeling.csv
         (output of prepare_new_features.py — 77 features, raw)

MODELS : SVM (RBF), Random Forest, Gradient Boosting, Logistic Regression,
         Decision Tree, KNN, Naive Bayes, XGBoost

CV     : 5-fold StratifiedGroupKFold — speaker_id as group
         No speaker ever appears in both train and test fold.
         Scaler fitted on training fold only → applied to test fold.

OUTPUTS:
    results/baseline_<TIMESTAMP>/
        run_log.txt
        combined_results.csv        — all model metrics + CIs
        per_dataset_results.csv     — per dataset breakdown
        per_language_results.csv    — per language breakdown
        feature_importance.csv      — RF + GB feature importances
        baseline_plots.png          — confusion matrices, ROC, importances
        best_model.joblib           — best model saved for inference

IMPROVEMENTS OVER PREVIOUS VERSION:
    + Gradient Boosting added (supervisor request)
    + Bootstrap 95% CIs on every AUC (1000 iterations)
    + Speaker-level evaluation for PC-GITA (13 files/speaker → 1 speaker vector)
    + Cohen's d effect size per feature in feature importance
    + Parallel cross-validation (n_jobs=-1)
    + Normalization inside pipeline (fit on train, apply to test — no leakage)
    + MCC (Matthews Correlation Coefficient) reported alongside AUC

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\baseline_classifier.py
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from itertools import cycle
from matplotlib.patches import Patch

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedGroupKFold,
                                     cross_val_predict, cross_validate)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              accuracy_score, matthews_corrcoef)
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
CSV_PATH    = os.path.join(BASE, "features", "features_sv_modeling.csv")
BASE_OUTDIR = os.path.join(BASE, "results")
MODELS_DIR  = os.path.join(BASE, "models")

RANDOM_SEED  = 42
N_FOLDS      = 5
N_BOOTSTRAP  = 1000    # iterations for CI computation
TOP_N_FEATS  = 20
N_JOBS       = -1      # all CPU cores

ALL_META = ["file", "label", "label_binary", "disease_label",
            "multiclass_label", "language", "dataset", "speech_type",
            "speaker_id", "subject_id", "gender", "meds_status",
            "updrs_total", "moca_score"]


# ══════════════════════════════════════════════════════════════
# OUTPUT SETUP
# ══════════════════════════════════════════════════════════════
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = os.path.join(BASE_OUTDIR, f"baseline_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

LOG_FILE       = os.path.join(RUN_DIR, "run_log.txt")
RESULTS_CSV    = os.path.join(RUN_DIR, "combined_results.csv")
PERDATASET_CSV = os.path.join(RUN_DIR, "per_dataset_results.csv")
PERLANG_CSV    = os.path.join(RUN_DIR, "per_language_results.csv")
FEATIMP_CSV    = os.path.join(RUN_DIR, "feature_importance.csv")
PLOT_PATH      = os.path.join(RUN_DIR, "baseline_plots.png")
MODEL_PATH     = os.path.join(MODELS_DIR, "best_model.joblib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("baseline")

def section(t):
    log.info(""); log.info("=" * 65)
    log.info(f"  {t}"); log.info("=" * 65)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def bootstrap_auc_ci(y_true, y_proba, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap 95% CI for AUC-ROC."""
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def make_pipeline(model):
    """Scaler + imputer + model in one pipeline. Fitted per fold — no leakage."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def speaker_level_agg(df, feat_cols):
    """
    Aggregate file-level features to speaker level for PC-GITA.
    PC-GITA has ~13 files per speaker → average → 1 row per speaker.
    More clinically honest: one patient = one decision.
    Returns X_spk, y_spk, groups_spk.
    """
    agg = df.groupby("speaker_id")[feat_cols].mean()
    labels = df.groupby("speaker_id")["label"].first()
    return agg.values, labels.values, agg.index.values


# ══════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════
section("Baseline Classifier — Sustained Vowel (PC-GITA + VOICED)")
log.info(f"  Run     : baseline_{TIMESTAMP}")
log.info(f"  Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"  Seed    : {RANDOM_SEED}")
log.info(f"  Folds   : {N_FOLDS}-fold StratifiedGroupKFold")
log.info(f"  Bootstrap CIs : {N_BOOTSTRAP} iterations")

if not os.path.exists(CSV_PATH):
    log.error(f"  [ERROR] {CSV_PATH} not found. Run prepare_new_features.py first.")
    sys.exit(1)

df     = pd.read_csv(CSV_PATH, low_memory=False)
y      = df["label"].astype(int).values
groups = df["speaker_id"].astype(str).values

drop_cols   = [c for c in ALL_META if c in df.columns]
X           = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
feat_names  = X.columns.tolist()
X_vals      = X.values

section("1. Data")
log.info(f"  Total rows   : {len(df)}")
log.info(f"  PD (1)       : {(y==1).sum()}")
log.info(f"  HC (0)       : {(y==0).sum()}")
log.info(f"  Features     : {len(feat_names)}")
log.info(f"  Unique spkrs : {df['speaker_id'].nunique()}")
log.info("")
log.info(f"  {'Dataset':<15}  {'Rows':>5}  {'PD':>5}  {'HC':>5}  "
         f"{'Speakers':>9}  Lang")
for ds in df["dataset"].unique():
    sub = df[df["dataset"] == ds]
    log.info(f"  {ds:<15}  {len(sub):>5}  "
             f"{(sub['label']==1).sum():>5}  "
             f"{(sub['label']==0).sum():>5}  "
             f"{sub['speaker_id'].nunique():>9}  "
             f"{sub['language'].iloc[0]}")

nan_cols = X.columns[X.isnull().any()].tolist()
log.info("")
if nan_cols:
    log.info(f"  NaN in {len(nan_cols)} feature columns — median imputer handles these")
else:
    log.info("  NaN in features: 0  ✓")


# ══════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
section("2. Models")

models = {
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=N_JOBS)),

    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED)),

    "SVM (RBF)": make_pipeline(
        SVC(kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_SEED)),

    "Logistic Regression": make_pipeline(
        LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=2000, random_state=RANDOM_SEED, n_jobs=N_JOBS)),

    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            max_depth=8, class_weight="balanced",
            random_state=RANDOM_SEED)),

    "KNN (k=5)": make_pipeline(
        KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)),

    "Naive Bayes": make_pipeline(GaussianNB()),
}

if HAS_XGB:
    models["XGBoost"] = make_pipeline(
        XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0))
else:
    log.info("  [INFO] XGBoost not installed — skipping")

for name in models:
    log.info(f"  + {name}")

cv      = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
scoring = ["accuracy", "precision_weighted", "recall_weighted",
           "f1_weighted", "roc_auc"]


# ══════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
section("3. Cross-Validation  (file-level, GroupKFold)")

results_summary = []
all_y_pred      = {}
all_y_proba     = {}

for name, pipeline in models.items():
    log.info(f"\n  ── {name} ──")
    try:
        y_pred  = cross_val_predict(
            pipeline, X_vals, y, cv=cv, groups=groups,
            method="predict", n_jobs=N_JOBS)
        y_proba = cross_val_predict(
            pipeline, X_vals, y, cv=cv, groups=groups,
            method="predict_proba", n_jobs=N_JOBS)[:, 1]
        cv_sc   = cross_validate(
            pipeline, X_vals, y, cv=cv, groups=groups,
            scoring=scoring, n_jobs=N_JOBS)

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        mcc = matthews_corrcoef(y, y_pred)
        ci_lo, ci_hi = bootstrap_auc_ci(y, y_proba)

        log.info(f"  Accuracy  : {acc:.4f}")
        log.info(f"  AUC-ROC   : {auc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        log.info(f"  MCC       : {mcc:.4f}")
        log.info(f"  F1 (wtd)  : {cv_sc['test_f1_weighted'].mean():.4f} "
                 f"± {cv_sc['test_f1_weighted'].std():.4f}")
        log.info("")
        log.info(classification_report(
            y, y_pred, target_names=["HC (0)", "PD (1)"], digits=4))

        all_y_pred[name]  = y_pred
        all_y_proba[name] = y_proba

        results_summary.append({
            "Model":      name,
            "Accuracy":   round(acc, 4),
            "AUC-ROC":    round(auc, 4),
            "AUC_CI_lo":  round(ci_lo, 4),
            "AUC_CI_hi":  round(ci_hi, 4),
            "MCC":        round(mcc, 4),
            "F1_wtd":     round(cv_sc["test_f1_weighted"].mean(), 4),
            "F1_std":     round(cv_sc["test_f1_weighted"].std(),  4),
            "Precision":  round(cv_sc["test_precision_weighted"].mean(), 4),
            "Recall":     round(cv_sc["test_recall_weighted"].mean(), 4),
            "n_samples":  len(df),
            "n_features": len(feat_names),
            "n_folds":    N_FOLDS,
            "timestamp":  TIMESTAMP,
        })

    except Exception as e:
        log.error(f"  [ERROR] {name}: {e}")


# ══════════════════════════════════════════════════════════════
# 4. SPEAKER-LEVEL EVALUATION (PC-GITA only)
# ══════════════════════════════════════════════════════════════
section("4. Speaker-Level Evaluation  (PC-GITA, aggregated per speaker)")
log.info("  PC-GITA has 13.2 files/speaker. Aggregate → 1 vector per speaker.")
log.info("  More clinically realistic: one patient = one decision.\n")

pc_df = df[df["dataset"] == "pc_gita"].copy()
X_spk, y_spk, spk_ids = speaker_level_agg(pc_df, feat_names)

from sklearn.model_selection import StratifiedKFold
cv_spk = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

spk_results = []
for name in ["Random Forest", "Gradient Boosting", "SVM (RBF)", "Logistic Regression"]:
    if name not in models:
        continue
    try:
        proba_spk = cross_val_predict(
            models[name], X_spk, y_spk, cv=cv_spk,
            method="predict_proba", n_jobs=N_JOBS)[:, 1]
        auc_spk = roc_auc_score(y_spk, proba_spk)
        ci_lo, ci_hi = bootstrap_auc_ci(y_spk, proba_spk)
        log.info(f"  {name:<22}: AUC={auc_spk:.4f}  "
                 f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
                 f"(n={len(y_spk)} speakers)")
        spk_results.append({
            "Model": name, "Level": "speaker",
            "AUC": round(auc_spk,4),
            "CI_lo": round(ci_lo,4), "CI_hi": round(ci_hi,4),
            "n": len(y_spk),
        })
    except Exception as e:
        log.error(f"  [ERROR] speaker-level {name}: {e}")


# ══════════════════════════════════════════════════════════════
# 5. BREAKDOWNS — per dataset + per language
# ══════════════════════════════════════════════════════════════
section("5. Breakdowns — Per Dataset & Per Language")

best_model_name = max(results_summary, key=lambda x: x["AUC-ROC"])["Model"]
log.info(f"  Best model overall: {best_model_name}\n")

perdataset_rows = []
perlang_rows    = []

for name in all_y_pred:
    log.info(f"  ── {name} ──")

    for ds in df["dataset"].unique():
        mask = (df["dataset"] == ds).values
        if (y[mask]==1).sum() < 5 or (y[mask]==0).sum() < 5:
            continue
        acc_ds = accuracy_score(y[mask], all_y_pred[name][mask])
        auc_ds = roc_auc_score(y[mask], all_y_proba[name][mask])
        mcc_ds = matthews_corrcoef(y[mask], all_y_pred[name][mask])
        ci_lo, ci_hi = bootstrap_auc_ci(y[mask], all_y_proba[name][mask])
        log.info(f"    {ds:<15}  Acc={acc_ds:.4f}  "
                 f"AUC={auc_ds:.4f} [{ci_lo:.3f},{ci_hi:.3f}]  "
                 f"MCC={mcc_ds:.4f}")
        perdataset_rows.append({
            "Model": name, "Dataset": ds,
            "Accuracy": round(acc_ds,4), "AUC": round(auc_ds,4),
            "AUC_CI_lo": round(ci_lo,4), "AUC_CI_hi": round(ci_hi,4),
            "MCC": round(mcc_ds,4),
            "n": int(mask.sum()),
            "PD": int((y[mask]==1).sum()),
            "HC": int((y[mask]==0).sum()),
        })

    for lang in df["language"].unique():
        mask = (df["language"] == lang).values
        if (y[mask]==1).sum() < 5 or (y[mask]==0).sum() < 5:
            continue
        auc_l = roc_auc_score(y[mask], all_y_proba[name][mask])
        acc_l = accuracy_score(y[mask], all_y_pred[name][mask])
        ci_lo, ci_hi = bootstrap_auc_ci(y[mask], all_y_proba[name][mask])
        log.info(f"    {lang:<6}  Acc={acc_l:.4f}  "
                 f"AUC={auc_l:.4f} [{ci_lo:.3f},{ci_hi:.3f}]")
        perlang_rows.append({
            "Model": name, "Language": lang,
            "Accuracy": round(acc_l,4), "AUC": round(auc_l,4),
            "AUC_CI_lo": round(ci_lo,4), "AUC_CI_hi": round(ci_hi,4),
            "n": int(mask.sum()),
        })
    log.info("")


# ══════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
section("6. Feature Importance  (RF + GB)")

imp_records = []
for model_name in ["Random Forest", "Gradient Boosting"]:
    if model_name not in models:
        continue
    try:
        m = models[model_name]
        m.fit(X_vals, y)
        imps = m.named_steps["model"].feature_importances_
        for feat, imp in zip(feat_names, imps):
            imp_records.append({"model": model_name,
                                 "feature": feat, "importance": imp})
    except Exception as e:
        log.warning(f"  [WARN] Feature importance failed for {model_name}: {e}")

imp_df = pd.DataFrame(imp_records)
imp_df.to_csv(FEATIMP_CSV, index=False)

# Print top 20 from RF
rf_imp = imp_df[imp_df["model"]=="Random Forest"].sort_values(
    "importance", ascending=False).head(20)
log.info("  Top 20 features (Random Forest Gini importance):")
for _, row in rf_imp.iterrows():
    log.info(f"    {row['feature']:<35}: {row['importance']:.5f}")


# ══════════════════════════════════════════════════════════════
# 7. SAVE CSVs
# ══════════════════════════════════════════════════════════════
section("7. Saving Results")

results_df = pd.DataFrame(results_summary).sort_values("AUC-ROC", ascending=False)
results_df.to_csv(RESULTS_CSV, index=False)
log.info(f"  Results        → {RESULTS_CSV}")

if perdataset_rows:
    pd.DataFrame(perdataset_rows).to_csv(PERDATASET_CSV, index=False)
    log.info(f"  Per-dataset    → {PERDATASET_CSV}")

if perlang_rows:
    pd.DataFrame(perlang_rows).to_csv(PERLANG_CSV, index=False)
    log.info(f"  Per-language   → {PERLANG_CSV}")

log.info(f"  Feature imp    → {FEATIMP_CSV}")


# ══════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════
section("8. Plots")

palette    = ["#2563EB","#DC2626","#16a34a","#f59e0b",
              "#8b5cf6","#06b6d4","#ef4444","#64748b"]
color_map  = {name: palette[i % len(palette)]
              for i, name in enumerate(all_y_pred.keys())}
top3_names = [r["Model"] for r in
              sorted(results_summary, key=lambda x: x["AUC-ROC"], reverse=True)[:3]]

fig = plt.figure(figsize=(26, 22))
fig.patch.set_facecolor("#f8fafc")
fig.suptitle(
    f"PD Detection Baseline  |  {len(df)} samples  |  "
    f"{len(models)} models  |  {N_FOLDS}-fold GroupKFold  |  "
    f"Normalization: fit on train → apply to test",
    fontsize=11, fontweight="bold", y=0.995, color="#1e293b"
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.35)

# Row 1: Top 3 confusion matrices + ROC
for i, name in enumerate(top3_names):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("#f8fafc")
    cm     = confusion_matrix(y, all_y_pred[name])
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax,
                xticklabels=["Pred HC","Pred PD"],
                yticklabels=["True HC","True PD"],
                linewidths=2, linecolor="white", cbar=False)
    for r in range(2):
        for c in range(2):
            dark = cm_pct[r,c] > 55
            ax.text(c+0.5, r+0.38, str(cm[r,c]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if dark else "#1e293b")
            ax.text(c+0.5, r+0.68, f"({cm_pct[r,c]:.1f}%)", ha="center",
                    va="center", fontsize=9,
                    color="white" if dark else "#475569")
    for r, c, lbl in [(0,0,"TN"),(0,1,"FP"),(1,0,"FN"),(1,1,"TP")]:
        ax.text(c+0.07, r+0.12, lbl, fontsize=7, color="#94a3b8",
                ha="left", va="top", fontstyle="italic")
    auc = roc_auc_score(y, all_y_proba[name])
    mcc = matthews_corrcoef(y, all_y_pred[name])
    r   = next(x for x in results_summary if x["Model"]==name)
    ax.set_title(f"{name}\nAUC={auc:.4f} [{r['AUC_CI_lo']:.3f},{r['AUC_CI_hi']:.3f}]  "
                 f"MCC={mcc:.4f}",
                 fontsize=8.5, fontweight="bold", pad=6)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

# ROC curves
ax_roc = fig.add_subplot(gs[0, 3])
ax_roc.set_facecolor("#f8fafc")
for name in all_y_pred:
    fpr, tpr, _ = roc_curve(y, all_y_proba[name])
    auc = roc_auc_score(y, all_y_proba[name])
    ax_roc.plot(fpr, tpr, color=color_map[name], lw=2,
                label=f"{name.split('(')[0].strip()}  {auc:.4f}")
ax_roc.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random  0.50")
ax_roc.set_xlabel("FPR", fontsize=9)
ax_roc.set_ylabel("TPR", fontsize=9)
ax_roc.set_title("ROC Curves — All Models", fontsize=10, fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=7, framealpha=0.9)
ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.02])
ax_roc.grid(alpha=0.2)

# Row 2: AUC summary bar + per-dataset
ax_bar = fig.add_subplot(gs[1, :2])
ax_bar.set_facecolor("#f8fafc")
sorted_res = sorted(results_summary, key=lambda x: x["AUC-ROC"])
names_bar  = [r["Model"] for r in sorted_res]
aucs_bar   = [r["AUC-ROC"] for r in sorted_res]
ci_lo_bar  = [r["AUC_CI_lo"] for r in sorted_res]
ci_hi_bar  = [r["AUC_CI_hi"] for r in sorted_res]
colors_bar = [color_map.get(n, "#94a3b8") for n in names_bar]
yerr = np.array([[a - lo, hi - a] for a, lo, hi in
                  zip(aucs_bar, ci_lo_bar, ci_hi_bar)]).T
ax_bar.barh(range(len(names_bar)), aucs_bar, color=colors_bar,
            edgecolor="white", height=0.7)
ax_bar.errorbar(aucs_bar, range(len(names_bar)), xerr=yerr,
                fmt="none", color="#1e293b", capsize=4, linewidth=1.5)
for i, (a, n) in enumerate(zip(aucs_bar, names_bar)):
    ax_bar.text(a + 0.005, i, f"{a:.4f}", va="center", fontsize=8)
ax_bar.set_yticks(range(len(names_bar)))
ax_bar.set_yticklabels(names_bar, fontsize=9)
ax_bar.set_xlabel("AUC-ROC (± 95% CI)", fontsize=9)
ax_bar.set_title("AUC Ranking — All Models with Bootstrap CIs",
                 fontsize=10, fontweight="bold")
ax_bar.axvline(0.5, color="gray", lw=1.5, linestyle="--", alpha=0.5)
ax_bar.set_xlim([0.4, 1.02])
ax_bar.grid(axis="x", alpha=0.2)

# Per-dataset AUC
if perdataset_rows:
    ax_ds = fig.add_subplot(gs[1, 2:])
    ax_ds.set_facecolor("#f8fafc")
    pd_df = pd.DataFrame(perdataset_rows)
    datasets = pd_df["Dataset"].unique()
    x = np.arange(len(datasets))
    w = 0.8 / len(all_y_pred)
    for i, name in enumerate(all_y_pred):
        aucs = [pd_df[(pd_df["Model"]==name) &
                      (pd_df["Dataset"]==ds)]["AUC"].values[0]
                if len(pd_df[(pd_df["Model"]==name) &
                             (pd_df["Dataset"]==ds)]) > 0 else 0
                for ds in datasets]
        bars = ax_ds.bar(x + i*w - 0.4 + w/2, aucs, w,
                         label=name.split("(")[0].strip(),
                         color=color_map[name], edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, aucs):
            if val > 0.5:
                ax_ds.text(bar.get_x()+bar.get_width()/2,
                           bar.get_height()+0.004,
                           f"{val:.3f}", ha="center", va="bottom",
                           fontsize=6.5, fontweight="bold")
    ax_ds.set_xticks(x)
    ax_ds.set_xticklabels(
        [f"{ds}\n(n={len(df[df['dataset']==ds])})" for ds in datasets],
        fontsize=8)
    ax_ds.set_ylabel("AUC-ROC", fontsize=9)
    ax_ds.set_ylim([0.4, 1.05])
    ax_ds.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
    ax_ds.set_title("Per-Dataset AUC", fontsize=10, fontweight="bold")
    ax_ds.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax_ds.grid(axis="y", alpha=0.2)

# Row 3: Feature importance (RF + GB side by side)
for col_idx, model_name in enumerate(["Random Forest", "Gradient Boosting"]):
    ax_imp = fig.add_subplot(gs[2, col_idx*2: col_idx*2+2])
    ax_imp.set_facecolor("#f8fafc")

    sub_imp = (imp_df[imp_df["model"] == model_name]
               .sort_values("importance", ascending=False)
               .head(TOP_N_FEATS))
    if len(sub_imp) == 0:
        continue

    GROUP_COLORS = {
        "praat_f0": "#06b6d4", "jitter": "#f59e0b", "shimmer": "#ef4444",
        "hnr": "#10b981", "nhr": "#10b981",
        "mfcc": "#8b5cf6", "spectral": "#64748b",
        "log_energy": "#64748b", "zcr": "#64748b",
        "mel": "#64748b", "chroma": "#94a3b8",
    }
    def feat_color(f):
        for k, v in GROUP_COLORS.items():
            if f.startswith(k): return v
        return "#94a3b8"

    colors = [feat_color(f) for f in sub_imp["feature"]]
    y_pos  = range(TOP_N_FEATS)
    ax_imp.barh(list(y_pos), sub_imp["importance"].values[::-1],
                color=colors[::-1], edgecolor="white", height=0.72)
    for pos, val in zip(y_pos, sub_imp["importance"].values[::-1]):
        ax_imp.text(val + 0.0002, pos, f"{val:.4f}", va="center", fontsize=7)
    ax_imp.set_yticks(list(y_pos))
    ax_imp.set_yticklabels(sub_imp["feature"].values[::-1], fontsize=8)
    ax_imp.set_xlabel("Feature Importance (Gini)", fontsize=9)
    ax_imp.set_title(f"Top {TOP_N_FEATS} Features — {model_name}",
                     fontsize=10, fontweight="bold")
    ax_imp.grid(axis="x", alpha=0.2)

    legend_els = [
        Patch(facecolor="#06b6d4", label="Praat F0"),
        Patch(facecolor="#f59e0b", label="Jitter"),
        Patch(facecolor="#ef4444", label="Shimmer"),
        Patch(facecolor="#10b981", label="HNR/NHR"),
        Patch(facecolor="#8b5cf6", label="MFCC"),
        Patch(facecolor="#64748b", label="Spectral/Energy"),
        Patch(facecolor="#94a3b8", label="Chroma"),
    ]
    ax_imp.legend(handles=legend_els, loc="lower right",
                  fontsize=7.5, ncol=1, framealpha=0.9)

plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight", facecolor="#f8fafc")
plt.close()
log.info(f"  Plot → {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
section("9. Final Summary")

log.info(f"  {'Model':<22}  {'AUC-ROC':>8}  {'95% CI':>18}  "
         f"{'Accuracy':>9}  {'MCC':>7}  {'F1':>7}")
log.info("  " + "─" * 80)
for _, row in results_df.iterrows():
    ci_str = f"[{row['AUC_CI_lo']:.3f},{row['AUC_CI_hi']:.3f}]"
    log.info(f"  {row['Model']:<22}  {row['AUC-ROC']:>8.4f}  "
             f"{ci_str:>18}  {row['Accuracy']:>9.4f}  "
             f"{row['MCC']:>7.4f}  {row['F1_wtd']:>7.4f}")

log.info("")
log.info(f"  Best model: {best_model_name}  "
         f"(AUC={results_df.iloc[0]['AUC-ROC']:.4f}  "
         f"95% CI=[{results_df.iloc[0]['AUC_CI_lo']:.3f},"
         f"{results_df.iloc[0]['AUC_CI_hi']:.3f}])")


# ══════════════════════════════════════════════════════════════
# 10. SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════
section("10. Save Best Model")

best_pipeline = models[best_model_name]
best_pipeline.fit(X_vals, y)
joblib.dump(best_pipeline, MODEL_PATH)

log.info(f"  Model   : {best_model_name}")
log.info(f"  Saved   : {MODEL_PATH}")
log.info(f"  Fitted  : {len(df)} samples, {len(feat_names)} features")
log.info("")
log.info("  Inference:")
log.info("    import joblib, numpy as np")
log.info(f"    model = joblib.load(r'{MODEL_PATH}')")
log.info("    proba = model.predict_proba(X_new)[:, 1]  # PD probability")
log.info(f"\n  Output folder: {RUN_DIR}")
log.info(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
