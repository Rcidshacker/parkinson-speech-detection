"""
3×3 Cross-Dataset Generalization Matrix + Cascade Pipeline
===========================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

INPUT  : features/opensmile/training_compare_8k_full.csv
         (output of prepare_compare_training.py — ComParE_2016, 6373+ features, 8kHz)

THE CORE RESEARCH QUESTION:
    "Does a model trained on PC-GITA (Spanish) PD speech detect PD in
     Voice_Dataset (English) speech it has never seen — and vice versa?"

3×3 MATRIX:
    Rows (Train) : D1=PC-GITA(ES),  D2=VOICE_DATASET(EN),  D1+D2=Combined
    Cols (Test)  : D1=PC-GITA(ES),  D2=VOICE_DATASET(EN),  D1+D2=Combined

    Diagonal     → Within-dataset: 5-fold StratifiedGroupKFold (CV)
    Off-diagonal → Cross-dataset: train on full set, test on full other set
                   (true zero-overlap cross-dataset evaluation)

NORMALIZATION (per supervisor instruction):
    "Always normalize the training set and apply to the testing set."
    StandardScaler fitted on training data → transformed on test data.
    No pre-normalization. Scaler is INSIDE every pipeline.

CASCADE PIPELINE:
    Stage 1: Logistic Regression   (threshold=0.70)
    Stage 2: Random Forest         (threshold=0.65)
    Stage 3: Gradient Boosting     (always outputs — final decision)

    Easy cases (high-confidence) caught at Stage 1 (fast, cheap).
    Hard cases escalate to Stage 3 (most powerful).
    Reports what % of cases each stage handles and its AUC on those cases.

FEATURE SUB-ANALYSES (ComParE_2016 functional groups):
    Run 1: Prosody features       (F0, energy, duration functionals)
    Run 2: Spectral/MFCC features (MFCCs, LSP, spectral functionals)
    Run 3: All features combined  (6373+ ComParE functionals)
    Answers: which ComParE feature group drives the cross-dataset signal?

BOOTSTRAP CIs: 1000 iterations on every matrix cell.

OUTPUTS:
    results/matrices/matrix_<TIMESTAMP>/
        run_log.txt
        matrix_auc.csv              — 3×3 AUC matrix
        matrix_auc_ci.csv           — 3×3 CI bounds
        cascade_results.csv         — cascade stage breakdown
        subanalysis_results.csv     — ComParE group sub-analysis
        matrix_plots.png            — heatmaps + cascade + sub-analysis

HOW TO RUN:
    venv\\Scripts\\activate
    python scripts\\pipeline\\dataset_matrix.py
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
CSV_PATH = os.path.join(BASE, "features", "opensmile", "training_compare_8k_full.csv")
BASE_OUTDIR = os.path.join(BASE, "results")

RANDOM_SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 1000
N_JOBS = -1

# Cascade confidence thresholds
CASCADE_THRESH = {
    "Logistic Regression": 0.70,
    "Random Forest": 0.65,
    # Gradient Boosting always outputs (Stage 3, no threshold)
}

ALL_META = [
    "file",
    "label",
    "label_binary",
    "disease_label",
    "multiclass_label",
    "language",
    "dataset",
    "speech_type",
    "speaker_id",
    "subject_id",
    "gender",
    "meds_status",
    "updrs_total",
    "moca_score",
]


# ══════════════════════════════════════════════════════════════
# OUTPUT SETUP
# ══════════════════════════════════════════════════════════════
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_OUTDIR, "matrices", f"matrix_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUN_DIR, "run_log.txt")
MATRIX_AUC_CSV = os.path.join(RUN_DIR, "matrix_auc.csv")
MATRIX_CI_CSV = os.path.join(RUN_DIR, "matrix_auc_ci.csv")
CASCADE_CSV = os.path.join(RUN_DIR, "cascade_results.csv")
SUBANALYSIS_CSV = os.path.join(RUN_DIR, "subanalysis_results.csv")
PLOT_PATH = os.path.join(RUN_DIR, "matrix_plots.png")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("matrix")


def section(t):
    log.info("")
    log.info("=" * 65)
    log.info(f"  {t}")
    log.info("=" * 65)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def bootstrap_auc_ci(y_true, y_proba, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap 95% confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def make_model(name):
    """Return a fresh unfitted pipeline for the given model name."""
    base = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        ),
    }[name]
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", base),
        ]
    )


def run_within_cv(X, y, groups, model_name):
    """
    Within-dataset evaluation: 5-fold StratifiedGroupKFold.
    Scaler fitted on each training fold separately.
    Returns (auc, ci_lo, ci_hi, y_proba).
    """
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pipe = make_model(model_name)
    proba = cross_val_predict(
        pipe, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=N_JOBS
    )[:, 1]
    auc = roc_auc_score(y, proba)
    ci_lo, ci_hi = bootstrap_auc_ci(y, proba)
    return auc, ci_lo, ci_hi, proba


def run_cross_dataset(X_train, y_train, X_test, y_test, model_name):
    """
    Cross-dataset evaluation: train on full train set → test on full test set.
    Scaler fitted on training data only → applied to test data.
    This is enforced by Pipeline(scaler+model).fit(X_train).predict(X_test).
    Returns (auc, ci_lo, ci_hi, y_proba).
    """
    pipe = make_model(model_name)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ci_lo, ci_hi = bootstrap_auc_ci(y_test, proba)
    return auc, ci_lo, ci_hi, proba


# ══════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════
section("3×3 Cross-Dataset Matrix — ComParE_2016 8kHz PD Detection")
log.info(f"  Run   : matrix_{TIMESTAMP}")
log.info(f"  Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"  Normalization: fit on train → apply to test (supervisor instruction)")
log.info(f"  Bootstrap CIs : {N_BOOTSTRAP} iterations per cell")

if not os.path.exists(CSV_PATH):
    log.error(f"  [ERROR] {CSV_PATH} not found. Run prepare_compare_training.py first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, low_memory=False)
y_all = df["label_binary"].astype(int).values

drop_cols = [c for c in ALL_META if c in df.columns]
X_all = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
feat_names = X_all.columns.tolist()
X_all = X_all.values
groups_all = df["subject_id"].astype(str).values

section("1. Data")
log.info(f"  Total rows  : {len(df)}")
log.info(f"  Features    : {len(feat_names)}")
for ds in df["dataset"].unique():
    sub = df[df["dataset"] == ds]
    log.info(
        f"  {ds:<15}: {len(sub):>5} rows  "
        f"PD={(sub['label_binary'] == 1).sum()}  "
        f"HC={(sub['label_binary'] == 0).sum()}  "
        f"lang={sub['language'].iloc[0]}"
    )


# ══════════════════════════════════════════════════════════════
# 2. BUILD DATASET SPLITS
# ══════════════════════════════════════════════════════════════
section("2. Dataset Splits")

D1_mask = (df["dataset"] == "pc_gita").values
D2_mask = (df["dataset"] == "voice_dataset").values

splits = {
    "PC-GITA (ES)": {"mask": D1_mask, "label": "D1"},
    "VOICE_DATASET (EN)": {"mask": D2_mask, "label": "D2"},
    "Combined": {"mask": np.ones(len(df), dtype=bool), "label": "D1+D2"},
}

for name, info in splits.items():
    m = info["mask"]
    log.info(
        f"  {name:<15}: {m.sum():>5} rows  "
        f"PD={(y_all[m] == 1).sum()}  HC={(y_all[m] == 0).sum()}"
    )


# ══════════════════════════════════════════════════════════════
# 3. PRIMARY CLASSIFIER — Random Forest for main matrix
# ══════════════════════════════════════════════════════════════
MAIN_MODEL = "Random Forest"

section(f"3. 3×3 AUC Matrix  (primary model: {MAIN_MODEL})")
log.info("  Diagonal   = 5-fold GroupKFold within-dataset CV")
log.info("  Off-diag   = full train → full test cross-lingual")
log.info("")

split_names = list(splits.keys())
matrix_auc = pd.DataFrame(index=split_names, columns=split_names, dtype=float)
matrix_ci_lo = pd.DataFrame(index=split_names, columns=split_names, dtype=float)
matrix_ci_hi = pd.DataFrame(index=split_names, columns=split_names, dtype=float)
matrix_proba = {}  # store probas for cascade use

for train_name, train_info in splits.items():
    for test_name, test_info in splits.items():
        tr_mask = train_info["mask"]
        te_mask = test_info["mask"]

        X_tr = X_all[tr_mask]
        y_tr = y_all[tr_mask]
        X_te = X_all[te_mask]
        y_te = y_all[te_mask]
        g_tr = groups_all[tr_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            matrix_auc.loc[train_name, test_name] = float("nan")
            continue

        if train_name == test_name:
            # Diagonal: GroupKFold CV
            auc, ci_lo, ci_hi, proba = run_within_cv(X_tr, y_tr, g_tr, MAIN_MODEL)
            cell_type = "CV"
        else:
            # Off-diagonal: cross-dataset
            auc, ci_lo, ci_hi, proba = run_cross_dataset(
                X_tr, y_tr, X_te, y_te, MAIN_MODEL
            )
            cell_type = "cross"

        matrix_auc.loc[train_name, test_name] = round(auc, 4)
        matrix_ci_lo.loc[train_name, test_name] = round(ci_lo, 4)
        matrix_ci_hi.loc[train_name, test_name] = round(ci_hi, 4)
        matrix_proba[(train_name, test_name)] = (proba, y_te)

        log.info(
            f"  [{train_name:<15} → {test_name:<15}]  "
            f"{cell_type}  AUC={auc:.4f}  "
            f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
        )

# Print matrix
log.info("")
log.info("  ── AUC Matrix ──")
log.info(f"  {'Train \\ Test':<20}" + "".join(f"{n:>18}" for n in split_names))
for train_name in split_names:
    row = f"  {train_name:<20}"
    for test_name in split_names:
        val = matrix_auc.loc[train_name, test_name]
        ci_lo = matrix_ci_lo.loc[train_name, test_name]
        ci_hi = matrix_ci_hi.loc[train_name, test_name]
        cell = f"{val:.4f}[{ci_lo:.3f},{ci_hi:.3f}]"
        row += f"{cell:>18}"
    log.info(row)

matrix_auc.to_csv(MATRIX_AUC_CSV)
matrix_ci_lo.to_csv(MATRIX_CI_CSV.replace(".csv", "_lo.csv"))
matrix_ci_hi.to_csv(MATRIX_CI_CSV.replace(".csv", "_hi.csv"))
log.info(f"\n  Saved → {MATRIX_AUC_CSV}")


# ══════════════════════════════════════════════════════════════
# 4. ALL MODELS CROSS-LINGUAL (off-diagonal only)
# ══════════════════════════════════════════════════════════════
section("4. All Models — Cross-Lingual Off-Diagonal")

cross_pairs = [
    ("PC-GITA (ES)", "VOICE_DATASET (EN)"),
    ("VOICE_DATASET (EN)", "PC-GITA (ES)"),
]
all_model_names = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "SVM (RBF)",
]
cross_rows = []

for train_name, test_name in cross_pairs:
    log.info(f"\n  ── Train: {train_name}  →  Test: {test_name} ──")
    tr_mask = splits[train_name]["mask"]
    te_mask = splits[test_name]["mask"]
    X_tr = X_all[tr_mask]
    y_tr = y_all[tr_mask]
    X_te = X_all[te_mask]
    y_te = y_all[te_mask]

    for mname in all_model_names:
        try:
            auc, ci_lo, ci_hi, _ = run_cross_dataset(X_tr, y_tr, X_te, y_te, mname)
            log.info(
                f"    {mname:<22}: AUC={auc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
            )
            cross_rows.append(
                {
                    "train": train_name,
                    "test": test_name,
                    "model": mname,
                    "AUC": round(auc, 4),
                    "CI_lo": round(ci_lo, 4),
                    "CI_hi": round(ci_hi, 4),
                }
            )
        except Exception as e:
            log.error(f"    [ERROR] {mname}: {e}")


# ══════════════════════════════════════════════════════════════
# 5. CASCADE PIPELINE
# ══════════════════════════════════════════════════════════════
section("5. Cascade Pipeline  (LR → RF → GB)")

log.info("  Stage 1: Logistic Regression  (confidence ≥ 0.70 → done)")
log.info("  Stage 2: Random Forest        (confidence ≥ 0.65 → done)")
log.info("  Stage 3: Gradient Boosting    (always outputs)")
log.info("")

cascade_rows = []

for train_name, test_name in cross_pairs:
    tr_mask = splits[train_name]["mask"]
    te_mask = splits[test_name]["mask"]
    X_tr = X_all[tr_mask]
    y_tr = y_all[tr_mask]
    X_te = X_all[te_mask]
    y_te = y_all[te_mask]
    n_test = len(y_te)

    log.info(f"  ── {train_name} → {test_name} ──")

    # Train all three models on training set
    stage_models = {}
    for mname in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        pipe = make_model(mname)
        pipe.fit(X_tr, y_tr)
        stage_models[mname] = pipe

    # Cascade inference
    final_proba = np.zeros(n_test)
    handled_by = np.full(n_test, "", dtype=object)
    stage_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    thresholds = [0.70, 0.65, None]  # None = always output

    remaining_idx = np.arange(n_test)

    for stage_name, thresh in zip(stage_names, thresholds):
        if len(remaining_idx) == 0:
            break

        proba_stage = stage_models[stage_name].predict_proba(X_te[remaining_idx])[:, 1]
        confidence = np.maximum(proba_stage, 1 - proba_stage)

        if thresh is not None:
            confident_mask = confidence >= thresh
            confident_idx = remaining_idx[confident_mask]
            not_confident_idx = remaining_idx[~confident_mask]
        else:
            confident_idx = remaining_idx
            not_confident_idx = np.array([], dtype=int)

        final_proba[confident_idx] = (
            proba_stage[confident_mask] if thresh is not None else proba_stage
        )
        handled_by[confident_idx] = stage_name

        pct = len(confident_idx) / n_test * 100
        if len(confident_idx) > 0:
            auc_stage = roc_auc_score(y_te[confident_idx], final_proba[confident_idx])
            ci_lo, ci_hi = bootstrap_auc_ci(
                y_te[confident_idx], final_proba[confident_idx]
            )
        else:
            auc_stage = float("nan")
            ci_lo = ci_hi = float("nan")

        log.info(
            f"    Stage {stage_names.index(stage_name) + 1} "
            f"({stage_name:<22}): handled "
            f"{len(confident_idx):>3}/{n_test} ({pct:.0f}%)  "
            f"AUC={auc_stage:.4f} [{ci_lo:.3f},{ci_hi:.3f}]"
        )

        cascade_rows.append(
            {
                "train": train_name,
                "test": test_name,
                "stage": stage_name,
                "stage_num": stage_names.index(stage_name) + 1,
                "n_handled": len(confident_idx),
                "pct_handled": round(pct, 1),
                "AUC": round(auc_stage, 4),
                "CI_lo": round(ci_lo, 4),
                "CI_hi": round(ci_hi, 4),
            }
        )

        remaining_idx = not_confident_idx

    # Full cascade AUC
    auc_cascade = roc_auc_score(y_te, final_proba)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, final_proba)
    log.info(
        f"    FULL CASCADE AUC: {auc_cascade:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
    )
    log.info("")

    cascade_rows.append(
        {
            "train": train_name,
            "test": test_name,
            "stage": "FULL_CASCADE",
            "stage_num": 0,
            "n_handled": n_test,
            "pct_handled": 100.0,
            "AUC": round(auc_cascade, 4),
            "CI_lo": round(ci_lo, 4),
            "CI_hi": round(ci_hi, 4),
        }
    )

pd.DataFrame(cascade_rows).to_csv(CASCADE_CSV, index=False)
log.info(f"  Saved → {CASCADE_CSV}")


# ══════════════════════════════════════════════════════════════
# 6. FEATURE SUB-ANALYSIS
# ══════════════════════════════════════════════════════════════
section("6. Feature Sub-Analysis  (ComParE_2016 functional groups)")

log.info("  Answers: which ComParE feature group drives the cross-dataset signal?")
log.info("")

# ComParE_2016 functional group splits based on OpenSMILE naming conventions:
#   F0-related  : functionals computed on the F0 contour (pitch)
#   Energy/Loudness: functionals on loudness / RMS energy contours  
#   MFCC/Spectral : functionals on MFCCs, LSP, spectral features
# All names are lowercase substrings found inside the ComParE feature column names.
prosody_feats = [
    f for f in feat_names
    if any(x in f.lower() for x in ["f0", "pitch", "jitter", "shimmer", "hnr", "nhr"])
]
spectral_feats = [
    f for f in feat_names
    if any(x in f.lower() for x in ["mfcc", "lsp", "spectral", "zcr", "energy", "loudness"])
]
all_feats = feat_names

# Guard: if name-based matching yields nothing (feature names are opaque indices),
# fall back gracefully to equal thirds of the feature list as proxy groups.
if len(prosody_feats) == 0 and len(spectral_feats) == 0:
    log.warning("  ComParE feature names are opaque indices — splitting into equal thirds as proxy groups.")
    n = len(all_feats)
    prosody_feats  = all_feats[:n // 3]
    spectral_feats = all_feats[n // 3: 2 * n // 3]

subsets = {
    "Prosody/Voice Quality": prosody_feats,
    "Spectral/MFCC":         spectral_feats,
    "All ComParE features":  all_feats,
}

log.info(f"  Prosody/Voice Quality features : {len(prosody_feats)}")
log.info(f"  Spectral/MFCC features         : {len(spectral_feats)}")
log.info(f"  All ComParE features           : {len(all_feats)}")
log.info("")

sub_rows = []

for subset_name, feat_subset in subsets.items():
    feat_idx = [feat_names.index(f) for f in feat_subset]
    log.info(f"  ── {subset_name} ({len(feat_subset)} features) ──")

    for train_name, test_name in cross_pairs:
        tr_mask = splits[train_name]["mask"]
        te_mask = splits[test_name]["mask"]
        X_tr_s = X_all[tr_mask][:, feat_idx]
        y_tr_s = y_all[tr_mask]
        X_te_s = X_all[te_mask][:, feat_idx]
        y_te_s = y_all[te_mask]

        # Run RF as standard comparison model
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=N_JOBS,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr_s, y_tr_s)
        proba = pipe.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(y_te_s, proba)
        ci_lo, ci_hi = bootstrap_auc_ci(y_te_s, proba)

        log.info(
            f"    {train_name:<15} → {test_name:<15}  "
            f"AUC={auc:.4f}  [{ci_lo:.4f},{ci_hi:.4f}]"
        )
        sub_rows.append(
            {
                "subset": subset_name,
                "n_feats": len(feat_subset),
                "train": train_name,
                "test": test_name,
                "AUC": round(auc, 4),
                "CI_lo": round(ci_lo, 4),
                "CI_hi": round(ci_hi, 4),
            }
        )
    log.info("")

pd.DataFrame(sub_rows).to_csv(SUBANALYSIS_CSV, index=False)
log.info(f"  Saved → {SUBANALYSIS_CSV}")


# ══════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════
section("7. Plots")

fig = plt.figure(figsize=(24, 20))
fig.patch.set_facecolor("#f8fafc")
fig.suptitle(
    f"Cross-Dataset Generalization Matrix  |  "
    f"PC-GITA (ES) + VOICE_DATASET (EN)  |  "
    f"Primary: {MAIN_MODEL}  |  "
    f"Normalization: fit on train → apply to test",
    fontsize=11,
    fontweight="bold",
    y=0.995,
    color="#1e293b",
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: AUC Heatmap ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor("#f8fafc")

auc_vals = matrix_auc.astype(float).values
im = ax1.imshow(auc_vals, cmap="RdYlGn", vmin=0.50, vmax=1.00, aspect="auto")
plt.colorbar(im, ax=ax1, label="AUC-ROC", shrink=0.8)

for i in range(len(split_names)):
    for j in range(len(split_names)):
        val = auc_vals[i, j]
        ci_lo = matrix_ci_lo.values[i, j]
        ci_hi = matrix_ci_hi.values[i, j]
        if not np.isnan(val):
            txt_color = "white" if val < 0.62 else "#1e293b"
            ax1.text(
                j,
                i - 0.12,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=txt_color,
            )
            ax1.text(
                j,
                i + 0.22,
                f"[{ci_lo:.3f},{ci_hi:.3f}]",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

diag_patch = plt.Rectangle(
    (-0.5, -0.5), 1, 1, fill=False, edgecolor="#1e293b", linewidth=2.5, linestyle="--"
)
ax1.add_patch(diag_patch)

ax1.set_xticks(range(len(split_names)))
ax1.set_yticks(range(len(split_names)))
ax1.set_xticklabels(["Test: " + n for n in split_names], fontsize=9, fontweight="bold")
ax1.set_yticklabels(["Train: " + n for n in split_names], fontsize=9, fontweight="bold")
ax1.set_title(
    f"3×3 AUC Matrix ({MAIN_MODEL})\n"
    f"Diagonal = within-dataset CV  |  Off-diagonal = cross-lingual",
    fontsize=10,
    fontweight="bold",
)

# ── Plot 2: Cross-lingual comparison across models ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#f8fafc")

cross_df = pd.DataFrame(cross_rows)
colors = {
    "Logistic Regression": "#2563EB",
    "Random Forest": "#DC2626",
    "Gradient Boosting": "#16a34a",
    "SVM (RBF)": "#f59e0b",
}
pair_labels = ["ES→IT", "IT→ES"]
x = np.arange(2)
w = 0.8 / len(all_model_names)

for i, mname in enumerate(all_model_names):
    sub = cross_df[cross_df["model"] == mname]
    aucs = sub["AUC"].values if len(sub) >= 2 else [0, 0]
    cilo = sub["CI_lo"].values if len(sub) >= 2 else [0, 0]
    cihi = sub["CI_hi"].values if len(sub) >= 2 else [0, 0]
    yerr = np.array([[a - lo, hi - a] for a, lo, hi in zip(aucs, cilo, cihi)]).T
    bars = ax2.bar(
        x + i * w - 0.4 + w / 2,
        aucs,
        w,
        label=mname.split("(")[0].strip(),
        color=colors.get(mname, "#94a3b8"),
        edgecolor="white",
        alpha=0.9,
    )
    ax2.errorbar(
        x + i * w - 0.4 + w / 2,
        aucs,
        yerr=yerr,
        fmt="none",
        color="#1e293b",
        capsize=3,
        linewidth=1.2,
    )
    for bar, val in zip(bars, aucs):
        if val > 0.5:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

ax2.set_xticks(x)
ax2.set_xticklabels(pair_labels, fontsize=10, fontweight="bold")
ax2.set_ylabel("AUC-ROC", fontsize=9)
ax2.set_ylim([0.4, 1.05])
ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
ax2.set_title(
    "Cross-Lingual AUC\n(all models, ± 95% CI)", fontsize=10, fontweight="bold"
)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.grid(axis="y", alpha=0.2)

# ── Plot 3: Sub-analysis grouped bar ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor("#f8fafc")

sub_df = pd.DataFrame(sub_rows)
subset_nms = list(subsets.keys())
pair_nms = [
    f"{tr.split('(')[0].strip()} → {te.split('(')[0].strip()}" for tr, te in cross_pairs
]
x2 = np.arange(len(pair_nms))
w2 = 0.8 / len(subset_nms)
sub_colors = {
    "Prosody/Voice Quality": "#ef4444",
    "Spectral/MFCC":         "#8b5cf6",
    "All ComParE features":  "#2563EB",
}

for i, sname in enumerate(subset_nms):
    sub_s = sub_df[sub_df["subset"] == sname]
    aucs = [
        sub_s[sub_s["train"] == tr][sub_s["test"] == te]["AUC"].values[0]
        if len(sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]) > 0
        else 0.0
        for tr, te in cross_pairs
    ]
    cilo = [
        sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]["CI_lo"].values[0]
        if len(sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]) > 0
        else 0.0
        for tr, te in cross_pairs
    ]
    cihi = [
        sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]["CI_hi"].values[0]
        if len(sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]) > 0
        else 0.0
        for tr, te in cross_pairs
    ]
    n_f = subsets[sname]
    yerr = np.array([[a - lo, hi - a] for a, lo, hi in zip(aucs, cilo, cihi)]).T
    bars = ax3.bar(
        x2 + i * w2 - 0.4 + w2 / 2,
        aucs,
        w2,
        label=f"{sname} (n={len(n_f)})",
        color=sub_colors[sname],
        edgecolor="white",
        alpha=0.9,
    )
    ax3.errorbar(
        x2 + i * w2 - 0.4 + w2 / 2,
        aucs,
        yerr=yerr,
        fmt="none",
        color="#1e293b",
        capsize=3,
        linewidth=1.2,
    )
    for bar, val in zip(bars, aucs):
        if val > 0.5:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

ax3.set_xticks(x2)
ax3.set_xticklabels(pair_nms, fontsize=10, fontweight="bold")
ax3.set_ylabel("AUC-ROC", fontsize=9)
ax3.set_ylim([0.4, 1.05])
ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
ax3.set_title(
    "ComParE Feature Group Sub-Analysis: Which Group Drives Cross-Dataset Signal?\n"
    "(Random Forest, ± 95% CI)",
    fontsize=10,
    fontweight="bold",
)
ax3.legend(fontsize=9, framealpha=0.9)
ax3.grid(axis="y", alpha=0.2)

# ── Plot 4: Cascade stage breakdown ──
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor("#f8fafc")

casc_df = pd.DataFrame(cascade_rows)
casc_df = casc_df[casc_df["stage"] != "FULL_CASCADE"]
stage_colors = {
    "Logistic Regression": "#2563EB",
    "Random Forest": "#DC2626",
    "Gradient Boosting": "#16a34a",
}

for pair_idx, (tr, te) in enumerate(cross_pairs):
    sub_c = casc_df[(casc_df["train"] == tr) & (casc_df["test"] == te)]
    bottom = pair_idx * 1.2
    left = 0.0
    for _, row in sub_c.iterrows():
        w_bar = row["pct_handled"] / 100.0
        ax4.barh(
            bottom,
            w_bar,
            left=left,
            height=0.8,
            color=stage_colors.get(row["stage"], "#94a3b8"),
            edgecolor="white",
            alpha=0.9,
        )
        if w_bar > 0.1:
            ax4.text(
                left + w_bar / 2,
                bottom,
                f"{row['pct_handled']:.0f}%\n{row['AUC']:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="white",
            )
        left += w_bar

    pair_label = f"ES→IT" if "GITA" in tr else "IT→ES"
    ax4.text(
        -0.02,
        bottom,
        pair_label,
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

legend_els = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=stage_colors["Logistic Regression"],
        label=f"Stage 1: LR (≥70%)",
    ),
    plt.Rectangle(
        (0, 0), 1, 1, fc=stage_colors["Random Forest"], label=f"Stage 2: RF (≥65%)"
    ),
    plt.Rectangle(
        (0, 0), 1, 1, fc=stage_colors["Gradient Boosting"], label=f"Stage 3: GB (all)"
    ),
]
ax4.legend(handles=legend_els, loc="lower right", fontsize=8, framealpha=0.9)
ax4.set_xlim([0, 1])
ax4.set_ylim([-0.3, 2.8])
ax4.set_xlabel("Fraction of Test Set", fontsize=9)
ax4.set_title(
    "Cascade Pipeline\n% handled per stage + AUC", fontsize=10, fontweight="bold"
)
ax4.set_yticks([])
ax4.grid(axis="x", alpha=0.2)

# ── Plot 5: AUC degradation bar (diagonal vs off-diagonal) ──
ax5 = fig.add_subplot(gs[2, :])
ax5.set_facecolor("#f8fafc")

# Collect within vs cross AUC for each direction
comparison_data = []
for mname in all_model_names:
    sub_m = cross_df[cross_df["model"] == mname]
    # PC-GITA within
    pc_within = matrix_auc.loc["PC-GITA (ES)", "PC-GITA (ES)"]
    voi_within = matrix_auc.loc["VOICE_DATASET (EN)", "VOICE_DATASET (EN)"]

    # Cross
    pc_to_voi = sub_m[
        (sub_m["train"] == "PC-GITA (ES)") & (sub_m["test"] == "VOICE_DATASET (EN)")
    ]["AUC"].values
    voi_to_pc = sub_m[
        (sub_m["train"] == "VOICE_DATASET (EN)") & (sub_m["test"] == "PC-GITA (ES)")
    ]["AUC"].values

    if len(pc_to_voi) > 0:
        comparison_data.append(
            {
                "label": f"{mname}\nPC-GITA within → Voice_Dataset cross",
                "within": pc_within,
                "cross": pc_to_voi[0],
                "color": colors.get(mname, "#94a3b8"),
            }
        )

x3 = np.arange(len(comparison_data))
within_vals = [d["within"] for d in comparison_data]
cross_vals = [d["cross"] for d in comparison_data]
labels3 = [d["label"] for d in comparison_data]
bar_colors3 = [d["color"] for d in comparison_data]

bars_w = ax5.bar(
    x3 - 0.2,
    within_vals,
    0.35,
    label="Within-dataset (CV)",
    color=bar_colors3,
    edgecolor="white",
    alpha=0.9,
)
bars_c = ax5.bar(
    x3 + 0.2,
    cross_vals,
    0.35,
    label="Cross-lingual (ES→IT)",
    color=bar_colors3,
    edgecolor="white",
    alpha=0.5,
    hatch="//",
)

for bar, val in zip(bars_w, within_vals):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
for bar, val in zip(bars_c, cross_vals):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

for i, (w, c) in enumerate(zip(within_vals, cross_vals)):
    drop = w - c
    ax5.annotate(
        f"Δ={drop:.3f}",
        xy=(i + 0.2, c),
        xytext=(i, max(w, c) + 0.06),
        fontsize=8,
        color="#475569",
        arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=1),
    )

ax5.set_xticks(x3)
ax5.set_xticklabels(labels3, fontsize=8.5)
ax5.set_ylabel("AUC-ROC", fontsize=9)
ax5.set_ylim([0.4, 1.05])
ax5.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
ax5.set_title(
    "Within-Dataset vs Cross-Lingual AUC Degradation\n"
    "Solid = within-dataset (CV)  |  Hatched = cross-lingual (ES→IT)",
    fontsize=10,
    fontweight="bold",
)
ax5.legend(fontsize=9, framealpha=0.9)
ax5.grid(axis="y", alpha=0.2)

plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight", facecolor="#f8fafc")
plt.close()
log.info(f"  Plot → {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("8. Final Summary")

log.info("  ── Cross-Lingual AUC (PC-GITA ES → VOICE_DATASET EN) ──")
for mname in all_model_names:
    sub = cross_df[
        (cross_df["model"] == mname)
        & (cross_df["train"] == "PC-GITA (ES)")
        & (cross_df["test"] == "VOICE_DATASET (EN)")
    ]
    if len(sub) > 0:
        row = sub.iloc[0]
        log.info(
            f"  {mname:<22}: AUC={row['AUC']:.4f}  "
            f"95% CI=[{row['CI_lo']:.4f},{row['CI_hi']:.4f}]"
        )

log.info("")
log.info("  ── Cross-Lingual AUC (VOICE_DATASET EN → PC-GITA ES) ──")
for mname in all_model_names:
    sub = cross_df[
        (cross_df["model"] == mname)
        & (cross_df["train"] == "VOICE_DATASET (EN)")
        & (cross_df["test"] == "PC-GITA (ES)")
    ]
    if len(sub) > 0:
        row = sub.iloc[0]
        log.info(
            f"  {mname:<22}: AUC={row['AUC']:.4f}  "
            f"95% CI=[{row['CI_lo']:.4f},{row['CI_hi']:.4f}]"
        )

log.info("")
log.info("  ── Feature Sub-Analysis Summary ──")
for sname in subsets:
    sub_s = pd.DataFrame(sub_rows)
    sub_s = sub_s[sub_s["subset"] == sname]
    log.info(f"  {sname:<22}:")
    for tr, te in cross_pairs:
        row = sub_s[(sub_s["train"] == tr) & (sub_s["test"] == te)]
        if len(row) > 0:
            r = row.iloc[0]
            dir_lbl = "ES→IT" if "GITA" in tr else "IT→ES"
            log.info(
                f"    {dir_lbl}: AUC={r['AUC']:.4f}  "
                f"[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]"
            )

log.info(f"\n  Output folder: {RUN_DIR}")
log.info(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
