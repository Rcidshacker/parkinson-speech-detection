import os, sys, warnings, logging, numpy as np, pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

BASE        = r'C:\Users\Lenovo\Desktop\Code\2026\BE mini project'
FEATURES    = os.path.join(BASE, 'features', 'features_sv_modeling.csv')
RESULTS_DIR = os.path.join(BASE, 'results', 'final',
              f'stacking_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
LOG_DIR     = os.path.join(BASE, 'logs')
N_BOOTSTRAP = 1000
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = os.path.join(LOG_DIR, f'stacking_{TIMESTAMP}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('stacking')

def section(title):
    log.info('')
    log.info('=' * 65)
    log.info(f'  {title}')
    log.info('=' * 65)

def bootstrap_auc(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def make_pipe(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model)
    ])

# Optimal 9 features from feature selection
OPT_FEATURES = [
    'spectral_flux_mean', 'shimmer_apq11', 'shimmer_local',
    'spectral_bandwidth_std', 'jitter_ppq5', 'jitter_local',
    'shimmer_apq5', 'jitter_rap', 'mfcc_04_std'
]

# Base models
BASE_MODELS = [
    ('RF',  RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ('GB',  GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
    ('LR',  LogisticRegression(max_iter=1000, random_state=SEED)),
]

section('1. Load Data')
df  = pd.read_csv(FEATURES)
es  = df[df['dataset']=='pc_gita'].copy()
it  = df[df['dataset']=='voiced'].copy()
log.info(f'  PC-GITA (ES): {len(es)} rows  PD={es["label"].sum()}  HC={(1-es["label"]).sum()}')
log.info(f'  VOICED  (IT): {len(it)} rows  PD={it["label"].sum()}  HC={(1-it["label"]).sum()}')
log.info(f'  Features used: {OPT_FEATURES}')

X_es = es[OPT_FEATURES].values
y_es = es['label'].values
X_it = it[OPT_FEATURES].values
y_it = it['label'].values

# ── OPTION A: Meta trained on ES OOF predictions ─────────────
section('2. Option A — Meta-Learner Trained on Spanish OOF')
log.info('  Train base models on ES with 5-fold CV → collect OOF probs')
log.info('  Train meta-learner (LR) on ES OOF probs')
log.info('  Test full stack on IT (cross-lingual)')
log.info('')

def option_a_single_direction(train_X, train_y, test_X, test_y,
                               direction_label, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Step 1: collect OOF predictions from base models
    oof_probs = np.zeros((len(train_y), len(BASE_MODELS)))
    for j, (name, model) in enumerate(BASE_MODELS):
        pipe = make_pipe(model)
        oof_probs[:, j] = cross_val_predict(
            pipe, train_X, train_y,
            cv=skf, method='predict_proba', n_jobs=-1
        )[:, 1]

    # Step 2: train meta-learner on OOF
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    meta.fit(oof_probs, train_y)

    # Step 3: generate test predictions from base models (fit on full train)
    test_probs = np.zeros((len(test_y), len(BASE_MODELS)))
    for j, (name, model) in enumerate(BASE_MODELS):
        pipe = make_pipe(model)
        pipe.fit(train_X, train_y)
        test_probs[:, j] = pipe.predict_proba(test_X)[:, 1]

    # Step 4: meta predict
    final_probs = meta.predict_proba(test_probs)[:, 1]
    auc         = roc_auc_score(test_y, final_probs)
    ci_lo, ci_hi = bootstrap_auc(test_y, final_probs)

    return {
        'direction': direction_label,
        'option'   : 'A',
        'auc'      : round(auc,   4),
        'ci_lo'    : round(ci_lo, 4),
        'ci_hi'    : round(ci_hi, 4),
        'oof_probs': oof_probs,
        'final_probs': final_probs,
        'y_true'   : test_y,
    }

# Run ES->IT and IT->ES in parallel
results_a = Parallel(n_jobs=2)([
    delayed(option_a_single_direction)(
        X_es, y_es, X_it, y_it, 'ES->IT'
    ),
    delayed(option_a_single_direction)(
        X_it, y_it, X_es, y_es, 'IT->ES'
    )
])

for r in results_a:
    log.info(f'  {r["direction"]}  AUC={r["auc"]:.4f}  '
             f'95% CI=[{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]')

# ── OPTION B: Meta trained on small IT hold-out ───────────────
section('3. Option B — Meta-Learner Trained on Italian Hold-out')
log.info('  Train base models on full ES')
log.info('  Split IT: 30% meta-train, 70% meta-test')
log.info('  Train meta-learner on IT 30% predictions')
log.info('  Test full stack on IT 70%')
log.info('')

from sklearn.model_selection import train_test_split

def option_b_es_it(seed=SEED):
    # Split Italian
    it_idx     = np.arange(len(y_it))
    meta_idx, test_idx = train_test_split(
        it_idx, test_size=0.7, random_state=seed,
        stratify=y_it
    )

    # Fit base models on full Spanish
    base_pipes = []
    for name, model in BASE_MODELS:
        pipe = make_pipe(model)
        pipe.fit(X_es, y_es)
        base_pipes.append(pipe)

    # Generate probs on IT meta-train split
    it_meta_probs = np.column_stack([
        p.predict_proba(X_it[meta_idx])[:, 1]
        for p in base_pipes
    ])

    # Train meta on IT 30%
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    meta.fit(it_meta_probs, y_it[meta_idx])

    # Test on IT 70%
    it_test_probs = np.column_stack([
        p.predict_proba(X_it[test_idx])[:, 1]
        for p in base_pipes
    ])
    final_probs = meta.predict_proba(it_test_probs)[:, 1]
    auc         = roc_auc_score(y_it[test_idx], final_probs)
    ci_lo, ci_hi = bootstrap_auc(y_it[test_idx], final_probs)

    return {
        'direction'  : 'ES->IT',
        'option'     : 'B',
        'auc'        : round(auc,   4),
        'ci_lo'      : round(ci_lo, 4),
        'ci_hi'      : round(ci_hi, 4),
        'meta_size'  : len(meta_idx),
        'test_size'  : len(test_idx),
        'final_probs': final_probs,
        'y_true'     : y_it[test_idx],
    }

def option_b_it_es(seed=SEED):
    # Split Spanish
    es_idx     = np.arange(len(y_es))
    meta_idx, test_idx = train_test_split(
        es_idx, test_size=0.7, random_state=seed,
        stratify=y_es
    )

    # Fit base models on full Italian
    base_pipes = []
    for name, model in BASE_MODELS:
        pipe = make_pipe(model)
        pipe.fit(X_it, y_it)
        base_pipes.append(pipe)

    # Generate probs on ES meta-train split
    es_meta_probs = np.column_stack([
        p.predict_proba(X_es[meta_idx])[:, 1]
        for p in base_pipes
    ])

    # Train meta on ES 30%
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    meta.fit(es_meta_probs, y_es[meta_idx])

    # Test on ES 70%
    es_test_probs = np.column_stack([
        p.predict_proba(X_es[test_idx])[:, 1]
        for p in base_pipes
    ])
    final_probs = meta.predict_proba(es_test_probs)[:, 1]
    auc         = roc_auc_score(y_es[test_idx], final_probs)
    ci_lo, ci_hi = bootstrap_auc(y_es[test_idx], final_probs)

    return {
        'direction'  : 'IT->ES',
        'option'     : 'B',
        'auc'        : round(auc,   4),
        'ci_lo'      : round(ci_lo, 4),
        'ci_hi'      : round(ci_hi, 4),
        'meta_size'  : len(meta_idx),
        'test_size'  : len(test_idx),
        'final_probs': final_probs,
        'y_true'     : y_es[test_idx],
    }

# Run both directions in parallel
results_b = Parallel(n_jobs=2)([
    delayed(option_b_es_it)(),
    delayed(option_b_it_es)()
])

for r in results_b:
    log.info(f'  {r["direction"]}  AUC={r["auc"]:.4f}  '
             f'95% CI=[{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]  '
             f'(meta_train={r["meta_size"]}, test={r["test_size"]})')

# ── COMPARISON ────────────────────────────────────────────────
section('4. Comparison vs All Approaches')

all_results = results_a + results_b

log.info(f'  {"Approach":<35} {"ES->IT":>7} {"IT->ES":>7} {"Mean":>7}')
log.info('  ' + '-'*60)

# Previous baselines
baselines = [
    ('All 77 features + RF',         0.5500, 0.5838),
    ('Biomarkers 9f + RF',           0.7500, 0.6796),
    ('Top-9 consistent + LR (BEST)', 0.9293, 0.7549),
]
for name, es_it, it_es in baselines:
    log.info(f'  {name:<35} {es_it:>7.4f} {it_es:>7.4f} {(es_it+it_es)/2:>7.4f}')

log.info('  ' + '-'*60)

# Option A results
a_es_it = next(r for r in results_a if r['direction']=='ES->IT')
a_it_es = next(r for r in results_a if r['direction']=='IT->ES')
log.info(f'  {"Stacking Option A (OOF meta)":<35} '
         f'{a_es_it["auc"]:>7.4f} {a_it_es["auc"]:>7.4f} '
         f'{(a_es_it["auc"]+a_it_es["auc"])/2:>7.4f}')

# Option B results
b_es_it = next(r for r in results_b if r['direction']=='ES->IT')
b_it_es = next(r for r in results_b if r['direction']=='IT->ES')
log.info(f'  {"Stacking Option B (IT holdout meta)":<35} '
         f'{b_es_it["auc"]:>7.4f} {b_it_es["auc"]:>7.4f} '
         f'{(b_es_it["auc"]+b_it_es["auc"])/2:>7.4f}')

# ── SAVE ──────────────────────────────────────────────────────
section('5. Save')

rows = []
for r in all_results:
    rows.append({
        'option'   : r['option'],
        'direction': r['direction'],
        'auc'      : r['auc'],
        'ci_lo'    : r['ci_lo'],
        'ci_hi'    : r['ci_hi'],
    })
out = pd.DataFrame(rows)
out_path = os.path.join(RESULTS_DIR, 'stacking_results.csv')
out.to_csv(out_path, index=False)
log.info(f'  Saved: {out_path}')

# ── PLOTS ─────────────────────────────────────────────────────
section('6. Plots')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Stacking Cross-Lingual Evaluation\nTop-9 Consistent Features | PC-GITA (ES) + VOICED (IT)',
             fontsize=13, fontweight='bold')

approaches = [
    'All 77\n+RF', 'Bio 9f\n+RF', 'Top-9\n+LR\n(BEST)',
    'Stack\nOpt-A', 'Stack\nOpt-B'
]
es_it_vals = [0.5500, 0.7500, 0.9293,
              a_es_it['auc'], b_es_it['auc']]
it_es_vals = [0.5838, 0.6796, 0.7549,
              a_it_es['auc'], b_it_es['auc']]
colors_bar = ['#90A4AE','#90A4AE','#2196F3','#FF9800','#4CAF50']

for ax, vals, title, baseline in zip(
    axes,
    [es_it_vals, it_es_vals],
    ['ES → IT (PC-GITA train → VOICED test)',
     'IT → ES (VOICED train → PC-GITA test)'],
    [0.55, 0.584]
):
    bars = ax.bar(approaches, vals, color=colors_bar, alpha=0.85, edgecolor='white')
    bars[2].set_edgecolor('gold'); bars[2].set_linewidth(3)
    ax.axhline(baseline, color='red',  linestyle='--', alpha=0.5, label=f'RF baseline ({baseline})')
    ax.axhline(0.5,      color='gray', linestyle=':',  alpha=0.5, label='Random (0.5)')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('AUC-ROC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'stacking_plots.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
log.info(f'  Plot: {plot_path}')

section('Done')
log.info(f'  Results: {RESULTS_DIR}')
log.info(f'  Log    : {LOG_FILE}')
