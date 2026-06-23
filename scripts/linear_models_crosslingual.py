"""
Linear Models Cross-Lingual Comparison
========================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

MOTIVATION:
    LR with top-9 consistent features achieved best cross-lingual AUC = 0.842 mean
    (ES->IT: 0.929, IT->ES: 0.755). Can any advanced linear model beat it?

MODELS TESTED:
    1. Logistic Regression          (baseline, our current best)
    2. Ridge Classifier             (stronger L2 regularization)
    3. Elastic Net LR               (L1+L2, automatic feature selection)
    4. Linear Discriminant Analysis (maximizes between-class separation)
    5. SGD Classifier               (online LR with elastic net)

APPROACH:
    - All models use same top-9 consistent features
    - ES->IT and IT->ES cross-lingual evaluation
    - All 5 models run in parallel (joblib)
    - 1000-iteration bootstrap CIs
    - Full comparison plot + CSV output

INPUT  : features/features_sv_modeling.csv
OUTPUT : results/final/linear_models_<timestamp>/
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
FEATURES    = os.path.join(BASE, "features", "features_sv_modeling.csv")
RESULTS_DIR = os.path.join(BASE, "results", "final",
              f"linear_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
LOG_DIR     = os.path.join(BASE, "logs")

N_BOOTSTRAP = 1000
SEED        = 42

# Top-9 consistent features from feature selection experiment
OPT_FEATURES = [
    "spectral_flux_mean",
    "shimmer_apq11",
    "shimmer_local",
    "spectral_bandwidth_std",
    "jitter_ppq5",
    "jitter_local",
    "shimmer_apq5",
    "jitter_rap",
    "mfcc_04_std",
]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"linear_models_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("linear_models")


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
    return np.percentile(np.array(aucs), [2.5, 97.5])


def make_pipe(model, needs_calibration=False):
    """Build Pipeline with imputer + scaler + model.
    RidgeClassifier doesn't have predict_proba natively,
    so we wrap it with CalibratedClassifierCV."""
    if needs_calibration:
        return Pipeline([
            ("imputer",    SimpleImputer(strategy="median")),
            ("scaler",     StandardScaler()),
            ("model",      CalibratedClassifierCV(model, cv=5, method="sigmoid")),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def evaluate_model(model_name, model, train_X, train_y,
                   test_X, test_y, direction, needs_calibration=False):
    """Train on train set, evaluate on test set. Returns result dict."""
    pipe = make_pipe(model, needs_calibration)
    pipe.fit(train_X, train_y)
    probs        = pipe.predict_proba(test_X)[:, 1]
    auc          = roc_auc_score(test_y, probs)
    ci_lo, ci_hi = bootstrap_auc(test_y, probs)
    return {
        "model"    : model_name,
        "direction": direction,
        "auc"      : round(auc,   4),
        "ci_lo"    : round(ci_lo, 4),
        "ci_hi"    : round(ci_hi, 4),
        "probs"    : probs,
        "y_true"   : test_y,
    }


# ── MODELS ───────────────────────────────────────────────────
MODELS = [
    ("Logistic Regression",
     LogisticRegression(max_iter=2000, random_state=SEED, C=1.0),
     False),

    ("Ridge Classifier",
     RidgeClassifier(alpha=1.0, random_state=SEED),
     True),   # needs calibration for probabilities

    ("Elastic Net LR",
     LogisticRegression(
         penalty="elasticnet", solver="saga",
         l1_ratio=0.5, C=1.0,
         max_iter=2000, random_state=SEED
     ),
     False),

    ("LDA",
     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
     False),

    ("SGD Classifier",
     SGDClassifier(
         loss="log_loss", penalty="elasticnet",
         l1_ratio=0.5, alpha=0.001,
         max_iter=2000, random_state=SEED
     ),
     False),
]


# ── LOAD DATA ────────────────────────────────────────────────
section("1. Load Data")

df  = pd.read_csv(FEATURES)
es  = df[df["dataset"] == "pc_gita"].copy()
it  = df[df["dataset"] == "voiced"].copy()

log.info(f"  PC-GITA (ES): {len(es)} rows  "
         f"PD={int(es['label'].sum())}  HC={int((1-es['label']).sum())}")
log.info(f"  VOICED  (IT): {len(it)} rows  "
         f"PD={int(it['label'].sum())}  HC={int((1-it['label']).sum())}")
log.info(f"  Features    : {OPT_FEATURES}")

X_es = es[OPT_FEATURES].values
y_es = es["label"].values
X_it = it[OPT_FEATURES].values
y_it = it["label"].values


# ── PARALLEL EVALUATION ──────────────────────────────────────
section("2. Parallel Cross-Lingual Evaluation (all 5 models × 2 directions)")

log.info("  Running 10 jobs in parallel (5 models × ES->IT + IT->ES)...")
log.info("")

# Build all jobs
jobs = []
for model_name, model, needs_cal in MODELS:
    # ES -> IT
    jobs.append(delayed(evaluate_model)(
        model_name, model,
        X_es, y_es, X_it, y_it,
        "ES->IT", needs_cal
    ))
    # IT -> ES
    jobs.append(delayed(evaluate_model)(
        model_name, model,
        X_it, y_it, X_es, y_es,
        "IT->ES", needs_cal
    ))

# Run all in parallel
all_results = Parallel(n_jobs=-1)(jobs)

# Log results grouped by model
for model_name, _, _ in MODELS:
    es_it = next(r for r in all_results
                 if r["model"] == model_name and r["direction"] == "ES->IT")
    it_es = next(r for r in all_results
                 if r["model"] == model_name and r["direction"] == "IT->ES")
    mean_auc = (es_it["auc"] + it_es["auc"]) / 2
    log.info(f"  {model_name:<25}  "
             f"ES->IT={es_it['auc']:.4f}[{es_it['ci_lo']:.3f},{es_it['ci_hi']:.3f}]  "
             f"IT->ES={it_es['auc']:.4f}[{it_es['ci_lo']:.3f},{it_es['ci_hi']:.3f}]  "
             f"mean={mean_auc:.4f}")


# ── SUMMARY TABLE ────────────────────────────────────────────
section("3. Full Comparison Summary")

rows = []
for model_name, _, _ in MODELS:
    es_it = next(r for r in all_results
                 if r["model"] == model_name and r["direction"] == "ES->IT")
    it_es = next(r for r in all_results
                 if r["model"] == model_name and r["direction"] == "IT->ES")
    mean_auc = round((es_it["auc"] + it_es["auc"]) / 2, 4)
    rows.append({
        "model"      : model_name,
        "es_it_auc"  : es_it["auc"],
        "es_it_ci_lo": es_it["ci_lo"],
        "es_it_ci_hi": es_it["ci_hi"],
        "it_es_auc"  : it_es["auc"],
        "it_es_ci_lo": it_es["ci_lo"],
        "it_es_ci_hi": it_es["ci_hi"],
        "mean_auc"   : mean_auc,
    })

results_df = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)

log.info(f"  {'Rank':<5} {'Model':<25} {'ES->IT':>7} "
         f"{'IT->ES':>7} {'Mean':>7}")
log.info("  " + "─" * 55)
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    marker = " ← BEST" if i == 1 else ""
    log.info(f"  {i:<5} {row['model']:<25} "
             f"{row['es_it_auc']:>7.4f} "
             f"{row['it_es_auc']:>7.4f} "
             f"{row['mean_auc']:>7.4f}{marker}")

best = results_df.iloc[0]
log.info("")
log.info(f"  Best model    : {best['model']}")
log.info(f"  ES->IT AUC    : {best['es_it_auc']:.4f}  "
         f"[{best['es_it_ci_lo']:.4f}, {best['es_it_ci_hi']:.4f}]")
log.info(f"  IT->ES AUC    : {best['it_es_auc']:.4f}  "
         f"[{best['it_es_ci_lo']:.4f}, {best['it_es_ci_hi']:.4f}]")
log.info(f"  Mean cross-lingual AUC: {best['mean_auc']:.4f}")


# ── SAVE CSV ─────────────────────────────────────────────────
section("4. Save Results")

csv_path = os.path.join(RESULTS_DIR, "linear_models_results.csv")
results_df.to_csv(csv_path, index=False)
log.info(f"  Saved: {csv_path}")


# ── PLOTS ────────────────────────────────────────────────────
section("5. Plots")

model_names  = [r["model"]    for r in rows]
es_it_aucs   = [r["es_it_auc"] for r in rows]
it_es_aucs   = [r["it_es_auc"] for r in rows]
es_it_ci_lo  = [r["es_it_ci_lo"] for r in rows]
es_it_ci_hi  = [r["es_it_ci_hi"] for r in rows]
it_es_ci_lo  = [r["it_es_ci_lo"] for r in rows]
it_es_ci_hi  = [r["it_es_ci_hi"] for r in rows]
mean_aucs    = [r["mean_auc"]  for r in rows]

# Short names for x-axis
short_names = {
    "Logistic Regression": "LR",
    "Ridge Classifier"   : "Ridge",
    "Elastic Net LR"     : "ElasticNet\nLR",
    "LDA"                : "LDA",
    "SGD Classifier"     : "SGD",
}
x_labels = [short_names.get(n, n) for n in model_names]

# Color: gold for best, blue for others
sorted_mean = sorted(mean_aucs, reverse=True)
bar_colors  = ["#FFD700" if m == sorted_mean[0]
               else "#2196F3" for m in mean_aucs]

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Linear Models Cross-Lingual Comparison\n"
    "Top-9 Consistent Features | PC-GITA (ES) + VOICED (IT)",
    fontsize=14, fontweight="bold", y=0.98
)

x    = np.arange(len(model_names))
w    = 0.30

# ── Plot 1: ES->IT AUC with CIs ──────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
bars = ax1.bar(x, es_it_aucs, color=bar_colors, alpha=0.85,
               edgecolor="white", width=0.5)
# Error bars
for i in range(len(model_names)):
    ax1.errorbar(x[i], es_it_aucs[i],
                 yerr=[[es_it_aucs[i]-es_it_ci_lo[i]],
                       [es_it_ci_hi[i]-es_it_aucs[i]]],
                 fmt="none", color="black", capsize=5, linewidth=1.5)
ax1.axhline(0.929, color="gold",  linestyle="--", alpha=0.8,
            label="LR benchmark (0.929)")
ax1.axhline(0.55,  color="red",   linestyle="--", alpha=0.5,
            label="RF baseline (0.55)")
ax1.axhline(0.5,   color="gray",  linestyle=":",  alpha=0.4,
            label="Random (0.5)")
for i, v in enumerate(es_it_aucs):
    ax1.text(x[i], v + 0.015, f"{v:.4f}", ha="center",
             fontsize=8, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(x_labels, fontsize=9)
ax1.set_ylim(0.3, 1.1)
ax1.set_title("ES → IT AUC (PC-GITA train → VOICED test)",
              fontweight="bold")
ax1.set_ylabel("AUC-ROC")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

# ── Plot 2: IT->ES AUC with CIs ──────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
bars2 = ax2.bar(x, it_es_aucs, color=bar_colors, alpha=0.85,
                edgecolor="white", width=0.5)
for i in range(len(model_names)):
    ax2.errorbar(x[i], it_es_aucs[i],
                 yerr=[[it_es_aucs[i]-it_es_ci_lo[i]],
                       [it_es_ci_hi[i]-it_es_aucs[i]]],
                 fmt="none", color="black", capsize=5, linewidth=1.5)
ax2.axhline(0.755, color="gold",  linestyle="--", alpha=0.8,
            label="LR benchmark (0.755)")
ax2.axhline(0.584, color="red",   linestyle="--", alpha=0.5,
            label="RF baseline (0.584)")
ax2.axhline(0.5,   color="gray",  linestyle=":",  alpha=0.4,
            label="Random (0.5)")
for i, v in enumerate(it_es_aucs):
    ax2.text(x[i], v + 0.015, f"{v:.4f}", ha="center",
             fontsize=8, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(x_labels, fontsize=9)
ax2.set_ylim(0.3, 1.1)
ax2.set_title("IT → ES AUC (VOICED train → PC-GITA test)",
              fontweight="bold")
ax2.set_ylabel("AUC-ROC")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

# ── Plot 3: Mean Cross-Lingual AUC ───────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
bars3 = ax3.bar(x, mean_aucs, color=bar_colors, alpha=0.85,
                edgecolor="white", width=0.5)
ax3.axhline(0.842, color="gold",  linestyle="--", alpha=0.8,
            label="LR benchmark (0.842)")
ax3.axhline(0.567, color="red",   linestyle="--", alpha=0.5,
            label="RF baseline (0.567)")
ax3.axhline(0.5,   color="gray",  linestyle=":",  alpha=0.4,
            label="Random (0.5)")
for i, v in enumerate(mean_aucs):
    ax3.text(x[i], v + 0.01, f"{v:.4f}", ha="center",
             fontsize=8, fontweight="bold")
ax3.set_xticks(x); ax3.set_xticklabels(x_labels, fontsize=9)
ax3.set_ylim(0.3, 1.1)
ax3.set_title("Mean Cross-Lingual AUC (ES↔IT average)",
              fontweight="bold")
ax3.set_ylabel("AUC-ROC")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Grouped bar — both directions side by side ───────
ax4 = fig.add_subplot(2, 2, 4)
x4  = np.arange(len(model_names))
w4  = 0.35
b1  = ax4.bar(x4 - w4/2, es_it_aucs, w4,
              label="ES→IT", color="#2196F3", alpha=0.85)
b2  = ax4.bar(x4 + w4/2, it_es_aucs, w4,
              label="IT→ES", color="#FF9800", alpha=0.85)

# Highlight best model bars
best_idx = mean_aucs.index(max(mean_aucs))
b1[best_idx].set_edgecolor("gold"); b1[best_idx].set_linewidth(3)
b2[best_idx].set_edgecolor("gold"); b2[best_idx].set_linewidth(3)

ax4.axhline(0.5, color="gray", linestyle=":", alpha=0.4,
            label="Random (0.5)")
for i, (v1, v2) in enumerate(zip(es_it_aucs, it_es_aucs)):
    ax4.text(x4[i]-w4/2, v1+0.01, f"{v1:.3f}",
             ha="center", fontsize=7)
    ax4.text(x4[i]+w4/2, v2+0.01, f"{v2:.3f}",
             ha="center", fontsize=7)
ax4.set_xticks(x4); ax4.set_xticklabels(x_labels, fontsize=9)
ax4.set_ylim(0.3, 1.1)
ax4.set_title("Both Directions Side by Side\n(gold border = best mean AUC)",
              fontweight="bold")
ax4.set_ylabel("AUC-ROC")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(RESULTS_DIR, "linear_models_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Plot saved: {plot_path}")

section("Done")
log.info(f"  Results folder : {RESULTS_DIR}")
log.info(f"  CSV            : {csv_path}")
log.info(f"  Plot           : {plot_path}")
log.info(f"  Log            : {LOG_FILE}")
log.info("")
log.info(f"  Best model     : {best['model']}")
log.info(f"  Mean cross-lingual AUC: {best['mean_auc']:.4f}")
log.info(f"  vs LR baseline : 0.8421")
delta = best["mean_auc"] - 0.8421
log.info(f"  Delta vs LR    : {delta:+.4f}")
