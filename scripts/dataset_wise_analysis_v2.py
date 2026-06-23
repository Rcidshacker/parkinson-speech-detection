#!/usr/bin/env python3
"""
Comprehensive Dataset-Wise Analysis — Dynamic Evaluation Engine
================================================================
Project : Speech-Based Parkinson's Disease Detection (BE Capstone)
Author  : Ruchit Das (22AM1084)

CHANGES FROM PREVIOUS VERSIONS:
    1. Dynamic CLI: Pass the specific training CSV via argparse.
    2. Dynamic Outputs: Auto-routes to results/evaluation_<dataset>_<timestamp>/.
    3. Plot Consolidation: Generates one Overlaid ROC curve per scenario.
    4. Plot Pruning: Generates a Confusion Matrix ONLY for the Top 1 model.
    5. Focused Heatmaps: Only plots AUC and MCC (clinical standards).
    6. Cross-Dataset Logic: Automatically detects remaining datasets for testing.

CORE PRESERVED LOGIC:
    - 7 models (LR, SVM, RF, XGB, DT, Voting, Stacking)
    - 3-part structure (self / cross-dataset / combined)
    - Pipeline scaling & imputation to prevent data leakage.
    - StratifiedGroupKFold on subject_id to prevent speaker leakage.
"""

import os
import sys
import json
import logging
import argparse
import warnings
import gc

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.ioff()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc,
    roc_auc_score, matthews_corrcoef
)

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION & SETUP
# ================================================================================
RANDOM_STATE = 42
N_FOLDS      = 5
N_BOOTSTRAP  = 1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_engine")

# ================================================================================
# BOOTSTRAP CI & PIPELINE HELPERS
# ================================================================================
def bootstrap_auc_ci(y_true, y_proba, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Bootstrap 95% CI for AUC-ROC."""
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(np.array(y_true)[idx])) < 2:
            continue
        aucs.append(roc_auc_score(np.array(y_true)[idx], np.array(y_proba)[idx]))
    if not aucs:
        return 0.0, 0.0
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def make_pipeline(model):
    """Wrap any model in an imputer+scaler pipeline. Fitted per fold — no leakage."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])

# ================================================================================
# DATA LOADING
# ================================================================================
def load_full_data(data_file):
    # FIX-C1: Log the Italian drop explicitly. Previously dropped silently with no
    # row count, no warning, and no way to detect from the log that data was excluded.
    logger.info(f'Loading data from {data_file}...')
    df = pd.read_csv(data_file)
    n_before = len(df)
    italian_rows = (df['dataset'] == 'italian').sum()
    if italian_rows > 0:
        logger.warning(
            f"  EXCLUDING {italian_rows} Italian rows from {n_before} total "
            f"(cross-lingual scope: PC-GITA + VOICED only). "
            f"Set df filter to include 'italian' if needed."
        )
    df = df[df['dataset'] != 'italian']
    logger.info(f'  Rows after exclusion: {len(df)}  (dropped {n_before - len(df)})')
    logger.info(f'  PD={df["label_binary"].sum()}  HC={(df["label_binary"]==0).sum()}')
    logger.info(f'  Datasets Detected: {list(df["dataset"].unique())}')
    return df

def get_feature_cols(df):
    return [col for col in df.columns
            if col not in ['dataset', 'subject_id', 'language', 'label_binary', 'disease_label', 'speech_type', 'file']]

# ================================================================================
# MODEL TRAINING ENGINE
# ================================================================================
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 7 models. Pipeline handles imputation+scaling strictly on train."""
    results = {}
    fitted_models = {}

    models_config = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1, random_state=RANDOM_STATE, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=10, probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=RANDOM_STATE, class_weight='balanced')
    }

    # 1-4. Train Standard Models
    for name, model in models_config.items():
        try:
            pipe = make_pipeline(model)
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_proba)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'pred': y_pred, 'proba': y_proba,
            }
            fitted_models[name] = pipe
        except Exception as e:
            logger.error(f'{name} failed: {e}')

    # 5. Train XGBoost (GPU -> CPU fallback)
    try:
        neg = (np.array(y_train) == 0).sum()
        pos = (np.array(y_train) == 1).sum()
        spw = neg / (pos + 1e-6)
        
        xgb_params = dict(n_estimators=150, max_depth=7, learning_rate=0.05,
                      random_state=RANDOM_STATE, eval_metric='logloss',
                      verbosity=0, scale_pos_weight=spw)
                      
        try:
            # Try GPU first
            pipe = make_pipeline(xgb.XGBClassifier(**xgb_params, tree_method='hist', device='cuda'))
            pipe.fit(X_train, y_train)
        except:
            # Fallback to CPU
            pipe = make_pipeline(xgb.XGBClassifier(**xgb_params))
            pipe.fit(X_train, y_train)
            
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_proba)
        
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'pred': y_pred, 'proba': y_proba,
        }
        fitted_models['XGBoost'] = pipe
    except Exception as e:
        logger.error(f'XGBoost failed: {e}')

    # 6. Voting Ensemble
    try:
        if len(fitted_models) >= 3:
            # FIX-C2: Was [:3] — hardcoded to LR+SVM+RF only. Now uses ALL fitted
            # models so XGBoost and DecisionTree contribute to ensemble predictions.
            base = list(fitted_models.items())
            voting = VotingClassifier(base, voting='soft', n_jobs=-1)
            voting.fit(X_train, y_train)
            y_pred  = voting.predict(X_test)
            y_proba = voting.predict_proba(X_test)[:, 1]
            ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_proba)
            results['VotingEnsemble'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'pred': y_pred, 'proba': y_proba,
            }
    except Exception as e:
        logger.error(f'Voting Ensemble failed: {e}')

    # 7. Stacking Ensemble
    try:
        if len(fitted_models) >= 3:
            # FIX-C2: Was [:3] — same fix as VotingEnsemble. All fitted models used.
            base = list(fitted_models.items())
            stacking = StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
                cv=5, n_jobs=-1)
            stacking.fit(X_train, y_train)
            y_pred  = stacking.predict(X_test)
            y_proba = stacking.predict_proba(X_test)[:, 1]
            ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_proba)
            results['StackingEnsemble'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'pred': y_pred, 'proba': y_proba,
            }
    except Exception as e:
        logger.error(f'Stacking Ensemble failed: {e}')

    return results, y_test

# ================================================================================
# STREAMLINED VISUALIZATIONS
# ================================================================================
def plot_overlaid_roc(results_dict, y_true, name, filepath):
    """Plots a single, clean image containing ROC curves for all evaluated models."""
    plt.figure(figsize=(10, 8))
    
    # Pre-defined professional color palette for 7 models
    colors = ['#DC2626', '#2563EB', '#059669', '#D97706', '#7C3AED', '#DB2777', '#4B5563']
    
    for (model_name, result), color in zip(results_dict.items(), colors):
        if 'proba' in result:
            fpr, tpr, _ = roc_curve(y_true, result['proba'])
            auc_val = result.get('auc', 0)
            plt.plot(fpr, tpr, lw=2.5, color=color, label=f'{model_name} (AUC = {auc_val:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Overlaid ROC Curves: {name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, name, filepath):
    """Plots CM only for the absolute best model to avoid plot bloat."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 14})
    plt.title(f"Best Model CM: {name}", fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    # 1. Setup Argparse for Dynamic Inputs
    parser = argparse.ArgumentParser(description="Evaluate PD Detection Models on specific Feature CSVs.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the training CSV (e.g., ../features/training_sustained_a_STABLE.csv)")
    args = parser.parse_args()
    
    DATA_FILE = args.data_file
    if not os.path.exists(DATA_FILE):
        logger.error(f"Cannot find data file: {DATA_FILE}")
        sys.exit(1)

    # 2. Setup Dynamic Output Directories
    base_name = os.path.basename(DATA_FILE).replace('.csv', '')
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # We navigate up one directory because this script is executed from inside 'scripts/'
    base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(base_project_dir, 'results', f'evaluation_{base_name}_{TIMESTAMP}')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for subdir in ['matrices', 'visualizations', 'reports', 'csv_results']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    logger.info('='*80)
    logger.info(f'EVALUATION ENGINE STARTED')
    logger.info(f'Input Data  : {DATA_FILE}')
    logger.info(f'Output Path : {OUTPUT_DIR}')
    logger.info('='*80)

    # 3. Load Data & Prepare Structures
    df = load_full_data(DATA_FILE)
    datasets = list(df['dataset'].unique())
    feature_cols = get_feature_cols(df)

    all_results  = {}
    summary_data = []

    # ============================================================================
    # PART 1: INDIVIDUAL DATASET ANALYSIS (Speaker-Safe)
    # ============================================================================
    logger.info('\n' + '='*80)
    logger.info('PART 1: INDIVIDUAL DATASET ANALYSIS (Self-Test)')
    logger.info('='*80)

    for dataset_name in datasets:
        logger.info(f'\nProcessing dataset: {dataset_name}')
        sub = df[df['dataset'] == dataset_name].copy()
        
        X = sub[feature_cols].values
        y = sub['label_binary'].values
        groups = sub['subject_id'].values

        n_unique = len(np.unique(groups))
        n_splits = min(N_FOLDS, n_unique // 2)
        
        if n_splits < 2:
            logger.warning(f'  {dataset_name}: not enough subjects for GroupKFold — skipping')
            continue

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        logger.info(f'  Samples={len(sub)} | Subjects={n_unique} | Folds={n_splits}')

        # FIX-B: Was next(cv.split()) — used only the FIRST fold, making all AUC
        # comparisons between 8k/10k/16k high-variance single-fold noise rather
        # than a stable estimate. Now runs full CV and aggregates mean ± std.
        # The final fold is retained for plot generation (CM + ROC).
        all_fold_results_self = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            fold_res, fold_y = train_all_models(X_tr, X_te, y_tr, y_te)
            all_fold_results_self.append({'results': fold_res, 'y_test': fold_y})

        # Aggregate: mean AUC across folds for logging
        model_names_self = list(all_fold_results_self[0]['results'].keys())
        logger.info(f'  Self-domain CV AUC (mean ± std across {n_splits} folds):')
        for m in model_names_self:
            auc_vals = [f['results'][m].get('auc', 0) for f in all_fold_results_self
                        if m in f['results']]
            if auc_vals:
                logger.info(f'    {m}: {np.mean(auc_vals):.4f} ± {np.std(auc_vals):.4f}')

        # Use last fold for visualization and summary_data reporting
        results   = all_fold_results_self[-1]['results']
        y_test_eval = all_fold_results_self[-1]['y_test']

        all_results[f'{dataset_name}_self'] = results

        # Identify the BEST model to plot its CM
        best_model_name = max(results, key=lambda k: results[k].get('auc', 0))
        logger.info(f'  🏆 Best Model: {best_model_name} (AUC={results[best_model_name].get("auc", 0):.4f})')
        
        scenario_name = f'{dataset_name}_self'
        
        plot_confusion_matrix(y_test_eval, results[best_model_name]['pred'], 
            f'{best_model_name} - {dataset_name}', 
            f'{OUTPUT_DIR}/matrices/{scenario_name}_Best_{best_model_name}_cm.png')
            
        plot_overlaid_roc(results, y_test_eval, f"{dataset_name} (Self-Test)", 
            f'{OUTPUT_DIR}/visualizations/{scenario_name}_Overlaid_ROC.png')

        for model_name, result in results.items():
            summary_data.append({
                'Analysis': f'{dataset_name} (Self)',
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1': result['f1'],
                'AUC': result.get('auc', None),
                'AUC_CI_lo': result.get('ci_lo', None),
                'AUC_CI_hi': result.get('ci_hi', None),
                'MCC': result.get('mcc', None),
            })

    gc.collect(); plt.close('all')

    # ============================================================================
    # PART 2: CROSS-DATASET ANALYSIS
    # ============================================================================
    if len(datasets) > 1:
        logger.info('\n' + '='*80)
        logger.info('PART 2: CROSS-DATASET EVALUATION (Train on A, Test on B)')
        logger.info('='*80)

        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset: continue

                logger.info(f'\nTrain: {train_dataset}  →  Test: {test_dataset}')

                train_df = df[df['dataset'] == train_dataset]
                test_df  = df[df['dataset'] == test_dataset]

                X_train = train_df[feature_cols].values
                y_train = train_df['label_binary'].values
                X_test  = test_df[feature_cols].values
                y_test  = test_df['label_binary'].values

                logger.info(f'  Train={len(X_train)}  Test={len(X_test)}')

                results, y_test_eval = train_all_models(X_train, X_test, y_train, y_test)
                scenario_name = f'{train_dataset}_to_{test_dataset}'
                all_results[scenario_name] = results

                # Identify Best Model
                best_model_name = max(results, key=lambda k: results[k].get('auc', 0))
                logger.info(f'  🏆 Best Model: {best_model_name} (AUC={results[best_model_name].get("auc", 0):.4f})')
                
                plot_confusion_matrix(y_test_eval, results[best_model_name]['pred'], 
                    f'{best_model_name} - {train_dataset} → {test_dataset}', 
                    f'{OUTPUT_DIR}/matrices/{scenario_name}_Best_{best_model_name}_cm.png')
                    
                plot_overlaid_roc(results, y_test_eval, f"{train_dataset} → {test_dataset}", 
                    f'{OUTPUT_DIR}/visualizations/{scenario_name}_Overlaid_ROC.png')

                for model_name, result in results.items():
                    summary_data.append({
                        'Analysis': f'{train_dataset}→{test_dataset}',
                        'Model': model_name,
                        'Accuracy': result['accuracy'],
                        'F1': result['f1'],
                        'AUC': result.get('auc', None),
                        'AUC_CI_lo': result.get('ci_lo', None),
                        'AUC_CI_hi': result.get('ci_hi', None),
                        'MCC': result.get('mcc', None),
                    })

        gc.collect(); plt.close('all')

    # ============================================================================
    # PART 3: COMBINED DATASET ANALYSIS
    # ============================================================================
    logger.info('\n' + '='*80)
    logger.info('PART 3: COMBINED DATASET ANALYSIS (Speaker-Safe GroupKFold)')
    logger.info('='*80)

    X_all      = df[feature_cols].values
    y_all      = df['label_binary'].values
    groups_all = df['subject_id'].values

    n_unique_all = len(np.unique(groups_all))
    n_splits_all = min(N_FOLDS, n_unique_all // 2)
    logger.info(f'  Combined Data: {len(df)} rows | Subjects={n_unique_all} | Folds={n_splits_all}')

    cv_all = StratifiedGroupKFold(n_splits=n_splits_all, shuffle=True, random_state=RANDOM_STATE)
    
    # Collect results from ALL folds, not just first
    all_fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_all.split(X_all, y_all, groups_all)):
        logger.info(f'  Fold {fold_idx+1}/{n_splits_all}...')
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        logger.info(f'    Train={len(X_train)}  Test={len(X_test)}')
        
        results, y_test_eval = train_all_models(X_train, X_test, y_train, y_test)
        
        all_fold_results.append({
            'fold': fold_idx,
            'results': results,
            'y_test': y_test_eval
        })
    
    # After all folds, aggregate metrics
    if all_fold_results:
        # For visualization, use the last fold
        results = all_fold_results[-1]['results']
        y_test_eval = all_fold_results[-1]['y_test']
        
        # Compute mean AUC across folds for logging
        model_names = list(results.keys())
        all_aucs = {m: [] for m in model_names}
        
        for fold_data in all_fold_results:
            for model_name in model_names:
                auc_val = fold_data['results'][model_name].get('auc', 0)
                all_aucs[model_name].append(auc_val)
        
        logger.info(f'\n  Cross-fold AUC (mean ± std):')
        for model_name in model_names:
            auc_mean = np.mean(all_aucs[model_name])
            auc_std = np.std(all_aucs[model_name])
            logger.info(f'    {model_name}: {auc_mean:.4f} ± {auc_std:.4f}')
    else:
        logger.error("No folds completed!")
        sys.exit(1)
    all_results['Combined'] = results

    # Identify Best Model
    best_model_name = max(results, key=lambda k: results[k].get('auc', 0))
    logger.info(f'  🏆 Best Model: {best_model_name} (AUC={results[best_model_name].get("auc", 0):.4f})')
    
    plot_confusion_matrix(y_test_eval, results[best_model_name]['pred'], 
        f'{best_model_name} - Combined Datasets', 
        f'{OUTPUT_DIR}/matrices/Combined_Best_{best_model_name}_cm.png')
        
    plot_overlaid_roc(results, y_test_eval, "Combined Datasets", 
        f'{OUTPUT_DIR}/visualizations/Combined_Overlaid_ROC.png')

    for model_name, result in results.items():
        summary_data.append({
            'Analysis': 'Combined (All)',
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'F1': result['f1'],
            'AUC': result.get('auc', None),
            'AUC_CI_lo': result.get('ci_lo', None),
            'AUC_CI_hi': result.get('ci_hi', None),
            'MCC': result.get('mcc', None),
        })

    gc.collect(); plt.close('all')

    # ============================================================================
    # FINAL REPORTS & FOCUSED HEATMAPS
    # ============================================================================
    logger.info('\n' + '='*80)
    logger.info('GENERATING FINAL REPORTS')
    logger.info('='*80)

    summary_df = pd.DataFrame(summary_data)
    csv_file   = os.path.join(OUTPUT_DIR, 'csv_results', 'analysis_summary.csv')
    summary_df.to_csv(csv_file, index=False)

    json_results = {}
    for key, val in all_results.items():
        json_results[key] = {
            k: {m: float(v) for m, v in d.items() if m not in ('pred', 'proba') and v is not None}
            for k, d in val.items()
        }
    with open(os.path.join(OUTPUT_DIR, 'reports', 'detailed_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    # ── Heatmap 1: AUC ──
    pivot_auc = summary_df.pivot_table(values='AUC', index='Analysis', columns='Model', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis', vmin=0.5, vmax=1.0, cbar_kws={'label': 'AUC-ROC'})
    plt.title(f'Clinical Discrimination (AUC) - {base_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'heatmap_AUC.png'), dpi=300)
    plt.close()

    # ── Heatmap 2: MCC ──
    pivot_mcc = summary_df.pivot_table(values='MCC', index='Analysis', columns='Model', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_mcc, annot=True, fmt='.3f', cmap='magma', vmin=-0.2, vmax=1.0, cbar_kws={'label': 'MCC'})
    plt.title(f'Imbalance-Robust Accuracy (MCC) - {base_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'heatmap_MCC.png'), dpi=300)
    plt.close()

    logger.info('✓ Saved focused heatmaps (AUC & MCC)')
    logger.info('\n' + '='*80)
    logger.info(f'✓ EVALUATION COMPLETED SUCCESSFULLY')
    logger.info(f'  All files routed to: {OUTPUT_DIR}')
    logger.info('='*80)

if __name__ == '__main__':
    main()