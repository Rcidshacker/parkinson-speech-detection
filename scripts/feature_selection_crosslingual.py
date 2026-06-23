"""
Cross-Lingual Feature Selection
================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

MOTIVATION:
    Baseline cross-lingual AUC with all 77 features : ~0.55 (near random)
    Biomarkers only (9 features)                    : ~0.75
    → Spectral features hurt cross-lingual generalization
    → Find optimal feature subset that maximizes cross-lingual AUC

APPROACH:
    1. Rank features by cross-lingual discriminative power
       (Mann-Whitney U between PD/HC, computed per language,
        keep features where direction is CONSISTENT across both languages)
    2. Try top-N subsets: N = 3,5,7,9,12,15,20,25,30,40,50,77
    3. Evaluate each subset: ES→IT and IT→ES AUC with RF + LR
    4. Find optimal N and feature set

INPUT  : features/features_sv_modeling.csv
OUTPUT : results/final/feature_selection_<timestamp>/
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
BASE        = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
FEATURES    = os.path.join(BASE, "features", "features_sv_modeling.csv")
RESULTS_DIR = os.path.join(BASE, "results", "final",
              f"feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
LOG_DIR     = os.path.join(BASE, "logs")

N_BOOTSTRAP = 1000
SEED        = 42

# Feature subsets to try
N_SUBSETS = [3, 5, 7, 9, 12, 15, 20, 25, 30, 40, 50, 77]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = os.path.join(LOG_DIR, f"feature_selection_{TIMESTAMP}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("featsel")

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

def cross_lingual_auc(train_df, test_df, features, model):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model)
    ])
    pipe.fit(train_df[features], train_df["label"])
    probs = pipe.predict_proba(test_df[features])[:, 1]
    auc = roc_auc_score(test_df["label"], probs)
    ci_lo, ci_hi = bootstrap_auc(
        test_df["label"].values, probs)
    return auc, ci_lo, ci_hi


# ── LOAD DATA ────────────────────────────────────────────────
section("1. Load Data")

df  = pd.read_csv(FEATURES)
es  = df[df["dataset"] == "pc_gita"].copy()  # Spanish
it  = df[df["dataset"] == "voiced"].copy()   # Italian

log.info(f"  PC-GITA (ES): {len(es)} rows  "
         f"PD={es['label'].sum()}  HC={(1-es['label']).sum()}")
log.info(f"  VOICED  (IT): {len(it)} rows  "
         f"PD={it['label'].sum()}  HC={(1-it['label']).sum()}")

# ── FEATURE COLUMNS ──────────────────────────────────────────
META_COLS = ["dataset","subject_id","language","gender","speech_type",
             "disease_label","label_binary","multiclass_label",
             "updrs_total","moca_score","meds_status","file",
             "label","speaker_id"]

feat_cols = [c for c in df.columns
             if c not in META_COLS
             and pd.api.types.is_numeric_dtype(df[c])]

log.info(f"  Total features: {len(feat_cols)}")

# ── RANK FEATURES BY CROSS-LINGUAL CONSISTENCY ───────────────
section("2. Feature Ranking — Cross-Lingual Consistency")

log.info("  Method: Mann-Whitney U per feature per language")
log.info("  Score : mean AUC across ES and IT")
log.info("  Filter: keep features where PD>HC or PD<HC direction")
log.info("          is CONSISTENT across both languages")
log.info("")

ranking = []

for feat in feat_cols:
    # ES stats
    es_pd = es[es["label"]==1][feat].dropna()
    es_hc = es[es["label"]==0][feat].dropna()
    if len(es_pd) < 5 or len(es_hc) < 5:
        continue
    _, es_p = stats.mannwhitneyu(es_pd, es_hc, alternative="two-sided")
    es_dir = "PD>HC" if es_pd.mean() > es_hc.mean() else "PD<HC"
    es_auc = roc_auc_score(
        es[es[feat].notna()]["label"],
        es[es[feat].notna()][feat] if es_dir == "PD>HC"
        else -es[es[feat].notna()][feat]
    )

    # IT stats
    it_pd = it[it["label"]==1][feat].dropna()
    it_hc = it[it["label"]==0][feat].dropna()
    if len(it_pd) < 5 or len(it_hc) < 5:
        continue
    _, it_p = stats.mannwhitneyu(it_pd, it_hc, alternative="two-sided")
    it_dir = "PD>HC" if it_pd.mean() > it_hc.mean() else "PD<HC"
    it_auc = roc_auc_score(
        it[it[feat].notna()]["label"],
        it[it[feat].notna()][feat] if it_dir == "PD>HC"
        else -it[it[feat].notna()][feat]
    )

    # Consistency check
    consistent = (es_dir == it_dir)
    mean_auc   = (es_auc + it_auc) / 2
    min_auc    = min(es_auc, it_auc)

    ranking.append({
        "feature"    : feat,
        "es_auc"     : round(es_auc, 4),
        "it_auc"     : round(it_auc, 4),
        "mean_auc"   : round(mean_auc, 4),
        "min_auc"    : round(min_auc, 4),
        "es_dir"     : es_dir,
        "it_dir"     : it_dir,
        "consistent" : consistent,
        "es_p"       : round(es_p, 6),
        "it_p"       : round(it_p, 6),
    })

rank_df = pd.DataFrame(ranking).sort_values(
    "min_auc", ascending=False).reset_index(drop=True)

# Show top 20
log.info(f"  {'Rank':<5} {'Feature':<25} {'ES AUC':>7} "
         f"{'IT AUC':>7} {'Min AUC':>8} {'Consistent':>10}")
log.info("  " + "─" * 65)
for i, row in rank_df.head(20).iterrows():
    consistent_str = "✓" if row["consistent"] else "✗"
    log.info(f"  {i+1:<5} {row['feature']:<25} "
             f"{row['es_auc']:>7.4f} {row['it_auc']:>7.4f} "
             f"{row['min_auc']:>8.4f} {consistent_str:>10}")

# Consistent features only
consistent_df = rank_df[rank_df["consistent"]].reset_index(drop=True)
log.info(f"\n  Total features     : {len(rank_df)}")
log.info(f"  Consistent direction: {len(consistent_df)} features")
log.info(f"  Inconsistent        : {len(rank_df) - len(consistent_df)} features")

# Save ranking
rank_path = os.path.join(RESULTS_DIR, "feature_ranking.csv")
rank_df.to_csv(rank_path, index=False)
log.info(f"  Saved: {rank_path}")

# ── TOP-N SUBSET EVALUATION ───────────────────────────────────
section("3. Top-N Subset Cross-Lingual Evaluation")

log.info("  Models: Logistic Regression + Random Forest")
log.info("  Direction: ES→IT and IT→ES")
log.info("")

models = {
    "LR" : LogisticRegression(max_iter=1000, random_state=SEED),
    "RF"  : RandomForestClassifier(
               n_estimators=300, random_state=SEED, n_jobs=-1),
    "SVM" : SVC(kernel="rbf", probability=True, random_state=SEED),
}

# Use consistent features ranked by min_auc
ranked_features = consistent_df["feature"].tolist()

# If we don't have enough consistent features, fall back to all ranked
if len(ranked_features) < 20:
    log.info("  [WARN] Few consistent features — using all ranked features")
    ranked_features = rank_df["feature"].tolist()

subset_results = []

for n in N_SUBSETS:
    if n > len(ranked_features):
        n = len(ranked_features)

    top_feats = ranked_features[:n]

    for model_name, model in models.items():
        # ES → IT
        auc_es_it, ci_lo_es_it, ci_hi_es_it = cross_lingual_auc(
            es, it, top_feats, model)

        # IT → ES
        auc_it_es, ci_lo_it_es, ci_hi_es_it2 = cross_lingual_auc(
            it, es, top_feats, model)

        mean_cross = (auc_es_it + auc_it_es) / 2

        log.info(f"  N={n:<3}  {model_name:<4}  "
                 f"ES→IT={auc_es_it:.4f}[{ci_lo_es_it:.3f},{ci_hi_es_it:.3f}]  "
                 f"IT→ES={auc_it_es:.4f}[{ci_lo_it_es:.3f},{ci_hi_es_it2:.3f}]  "
                 f"mean={mean_cross:.4f}")

        subset_results.append({
            "n_features"  : n,
            "model"       : model_name,
            "es_it_auc"   : round(auc_es_it,  4),
            "it_es_auc"   : round(auc_it_es,  4),
            "mean_cross"  : round(mean_cross, 4),
            "ci_lo_es_it" : round(ci_lo_es_it, 4),
            "ci_hi_es_it" : round(ci_hi_es_it, 4),
            "ci_lo_it_es" : round(ci_lo_it_es, 4),
            "ci_hi_it_es" : round(ci_hi_es_it2, 4),
            "top_features": ", ".join(top_feats[:5]) + "..."
        })

# ── FIND OPTIMAL N ────────────────────────────────────────────
section("4. Optimal Feature Subset")

results_df = pd.DataFrame(subset_results)
best_idx   = results_df["mean_cross"].idxmax()
best       = results_df.loc[best_idx]

log.info(f"  Best N          : {int(best['n_features'])} features")
log.info(f"  Best model      : {best['model']}")
log.info(f"  ES→IT AUC       : {best['es_it_auc']:.4f}")
log.info(f"  IT→ES AUC       : {best['it_es_auc']:.4f}")
log.info(f"  Mean cross-lingual AUC: {best['mean_cross']:.4f}")
log.info("")

# Show optimal feature list
n_best    = int(best["n_features"])
opt_feats = ranked_features[:n_best]
log.info(f"  Optimal {n_best} features:")
for i, f in enumerate(opt_feats, 1):
    row = rank_df[rank_df["feature"] == f].iloc[0]
    log.info(f"    {i:2d}. {f:<25} "
             f"ES={row['es_auc']:.4f}  IT={row['it_auc']:.4f}  "
             f"dir={row['es_dir']}")

# ── COMPARE AGAINST BASELINES ────────────────────────────────
section("5. Comparison vs Baselines")

log.info(f"  {'Approach':<35} {'ES→IT':>7} {'IT→ES':>7} {'Mean':>7}")
log.info("  " + "─" * 60)
log.info(f"  {'All 77 features (RF baseline)':<35} {'0.5500':>7} {'0.5838':>7} {'0.5669':>7}")
log.info(f"  {'Biomarkers only - 9 features (RF)':<35} {'0.7500':>7} {'0.6796':>7} {'0.7148':>7}")

for _, row in results_df[results_df["model"]=="RF"].iterrows():
    marker = " ← BEST" if row["n_features"] == n_best and row["model"] == best["model"] else ""
    log.info(f"  {f'Top-{int(row.n_features)} consistent features (RF)':<35} "
             f"{row.es_it_auc:>7.4f} {row.it_es_auc:>7.4f} "
             f"{row.mean_cross:>7.4f}{marker}")

# ── SAVE ─────────────────────────────────────────────────────
section("6. Save")

results_path = os.path.join(RESULTS_DIR, "subset_results.csv")
results_df.to_csv(results_path, index=False)
log.info(f"  Subset results : {results_path}")

# Save optimal feature list
opt_df = rank_df[rank_df["feature"].isin(opt_feats)].copy()
opt_path = os.path.join(RESULTS_DIR, "optimal_features.csv")
opt_df.to_csv(opt_path, index=False)
log.info(f"  Optimal features: {opt_path}")
log.info(f"  Log             : {LOG_FILE}")


# ── PLOTS ─────────────────────────────────────────────────────
section("7. Plots")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

results_df_plot = pd.read_csv(results_path)
rank_df_plot2   = pd.read_csv(rank_path)

fig = plt.figure(figsize=(20, 18))
fig.suptitle("Cross-Lingual Feature Selection — PD Detection\nPC-GITA (ES) + VOICED (IT) | Sustained Vowel /a/",
             fontsize=14, fontweight="bold", y=0.98)

colors = {"LR": "#2196F3", "RF": "#4CAF50", "SVM": "#FF5722"}
styles = {"LR": "-o", "RF": "-s", "SVM": "-^"}

ax1 = fig.add_subplot(3, 3, 1)
for model in results_df_plot["model"].unique():
    sub = results_df_plot[results_df_plot["model"]==model].drop_duplicates("n_features")
    ax1.plot(sub["n_features"], sub["es_it_auc"], styles.get(model,"-o"),
             color=colors.get(model,"gray"), label=model, linewidth=2, markersize=6)
ax1.axhline(0.55,  color="red",  linestyle="--", alpha=0.5, label="Baseline RF (0.55)")
ax1.axhline(0.5,   color="gray", linestyle=":",  alpha=0.5, label="Random (0.50)")
ax1.set_title("ES→IT AUC vs N Features", fontweight="bold")
ax1.set_xlabel("N Features"); ax1.set_ylabel("AUC-ROC")
ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(3, 3, 2)
for model in results_df_plot["model"].unique():
    sub = results_df_plot[results_df_plot["model"]==model].drop_duplicates("n_features")
    ax2.plot(sub["n_features"], sub["it_es_auc"], styles.get(model,"-o"),
             color=colors.get(model,"gray"), label=model, linewidth=2, markersize=6)
ax2.axhline(0.584, color="red",  linestyle="--", alpha=0.5, label="Baseline RF (0.584)")
ax2.axhline(0.5,   color="gray", linestyle=":",  alpha=0.5, label="Random (0.50)")
ax2.set_title("IT→ES AUC vs N Features", fontweight="bold")
ax2.set_xlabel("N Features"); ax2.set_ylabel("AUC-ROC")
ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(3, 3, 3)
for model in results_df_plot["model"].unique():
    sub = results_df_plot[results_df_plot["model"]==model].drop_duplicates("n_features")
    ax3.plot(sub["n_features"], sub["mean_cross"], styles.get(model,"-o"),
             color=colors.get(model,"gray"), label=model, linewidth=2, markersize=6)
ax3.axhline(0.567, color="red",  linestyle="--", alpha=0.5, label="Baseline RF (0.567)")
ax3.axhline(0.5,   color="gray", linestyle=":",  alpha=0.5, label="Random (0.50)")
best_row = results_df_plot.loc[results_df_plot["mean_cross"].idxmax()]
ax3.scatter(best_row["n_features"], best_row["mean_cross"], s=200, zorder=5,
            color="gold", edgecolors="black",
            label=f"Best: N={int(best_row.n_features)} {best_row.model} ({best_row.mean_cross:.3f})")
ax3.set_title("Mean Cross-Lingual AUC vs N Features", fontweight="bold")
ax3.set_xlabel("N Features"); ax3.set_ylabel("AUC-ROC")
ax3.set_ylim(0, 1.05); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(3, 3, (4, 5))
top20 = rank_df_plot2.head(20).copy()
x = np.arange(len(top20))
w = 0.35
for i, (_, row) in enumerate(top20.iterrows()):
    alpha = 0.9 if row["consistent"] else 0.35
    ax4.bar(x[i]-w/2, row["es_auc"], w, color="#2196F3", alpha=alpha)
    ax4.bar(x[i]+w/2, row["it_auc"], w, color="#FF9800", alpha=alpha)
ax4.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels([f[:15] for f in top20["feature"]], rotation=45, ha="right", fontsize=7)
ax4.set_title("Top-20 Features: Individual AUC per Language\n(faded = inconsistent direction)", fontweight="bold")
ax4.set_ylabel("AUC-ROC"); ax4.set_ylim(0.4, 1.0)
p1 = mpatches.Patch(color="#2196F3", alpha=0.9,  label="ES AUC (consistent)")
p2 = mpatches.Patch(color="#2196F3", alpha=0.35, label="ES AUC (inconsistent)")
p3 = mpatches.Patch(color="#FF9800", alpha=0.9,  label="IT AUC")
ax4.legend(handles=[p1,p2,p3], fontsize=8); ax4.grid(True, alpha=0.3, axis="y")

ax5 = fig.add_subplot(3, 3, 6)
cc = rank_df_plot2["consistent"].sum()
ic = len(rank_df_plot2) - cc
ax5.pie([cc, ic], labels=[f"Consistent\n({cc})", f"Inconsistent\n({ic})"],
        colors=["#4CAF50","#F44336"], autopct="%1.0f%%", startangle=90,
        textprops={"fontsize":10})
ax5.set_title("Feature Direction Consistency\nacross ES and IT", fontweight="bold")

ax6 = fig.add_subplot(3, 1, 3)
approaches = ["All 77\n(RF)","Biomarkers\n9f (RF)","Top-3\n(LR)","Top-5\n(LR)",
              "Top-7\n(LR)","Top-9\n(LR) BEST","Top-12\n(LR)","Top-15\n(LR)","Top-25\n(LR)"]
es_vals = [0.550, 0.750, 0.8766, 0.9287, 0.9278, 0.9293, 0.8965, 0.8977, 0.9215]
it_vals = [0.584, 0.680, 0.7374, 0.7550, 0.7545, 0.7549, 0.7423, 0.7404, 0.7382]
xb = np.arange(len(approaches))
bars1 = ax6.bar(xb-w/2, es_vals, w, label="ES->IT", color="#2196F3", alpha=0.85)
bars2 = ax6.bar(xb+w/2, it_vals, w, label="IT->ES", color="#FF9800", alpha=0.85)
bars1[5].set_edgecolor("gold"); bars1[5].set_linewidth(3)
bars2[5].set_edgecolor("gold"); bars2[5].set_linewidth(3)
ax6.axhline(0.5,  color="gray",  linestyle=":", alpha=0.5, label="Random")
ax6.axhline(0.75, color="green", linestyle="--", alpha=0.4, label="0.75 ref")
for i,(es,it) in enumerate(zip(es_vals,it_vals)):
    fw = "bold" if i==5 else "normal"
    ax6.text(xb[i]-w/2, es+0.01, f"{es:.2f}", ha="center", fontsize=7, fontweight=fw)
    ax6.text(xb[i]+w/2, it+0.01, f"{it:.2f}", ha="center", fontsize=7, fontweight=fw)
ax6.set_xticks(xb); ax6.set_xticklabels(approaches, fontsize=9)
ax6.set_ylabel("AUC-ROC"); ax6.set_ylim(0, 1.1)
ax6.set_title("Cross-Lingual AUC Comparison — All Approaches", fontweight="bold")
ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(RESULTS_DIR, "feature_selection_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
log.info(f"  Plot saved: {plot_path}")



section("Done")
log.info(f"  Results folder: {RESULTS_DIR}")