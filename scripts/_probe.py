import pandas as pd, glob, os

BASE = r'C:\Users\Lenovo\Desktop\Code\2026\BE mini project'

df = pd.read_csv(f'{BASE}/features/training_top112_113features.csv')
feat_cols = [c for c in df.columns if c not in {'dataset','subject_id','language','label_binary'}]
print(f'training_top112 : {len(df)} rows  {len(feat_cols)} features')

rf_dir = sorted(glob.glob(f'{BASE}/results/final/rf_importance_comparison_*'))[-1]
print(f'RF dir          : {os.path.basename(rf_dir)}')

rf_es  = pd.read_csv(f'{rf_dir}/rf_full_ranking_es.csv')
rf_it  = pd.read_csv(f'{rf_dir}/rf_full_ranking_it.csv')
master = pd.read_csv(f'{rf_dir}/master_comparison_all112.csv')

print(f'RF ES  top-14   : {list(rf_es.head(14)["feature"])}')
print(f'RF IT  top-14   : {list(rf_it.head(14)["feature"])}')
print(f'Bottom 2 (CNN)  : {list(master.tail(2)["feature"])}')

mfcc = [f for f in feat_cols if any(f.startswith(p) for p in ['mfcc_','dmfcc_','d2mfcc_'])]
print(f'MFCC family     : {len(mfcc)} features')

BIOMARKERS = {'jitter_local','jitter_rap','jitter_ppq5','jitter_ddp',
              'shimmer_local','shimmer_apq3','shimmer_apq5','shimmer_apq11','shimmer_dda','hnr','nhr'}
bio = sorted([f for f in feat_cols if f in BIOMARKERS])
print(f'Biomarkers      : {len(bio)} -> {bio}')

# F0 features
f0 = [f for f in feat_cols if any(f.startswith(p) for p in ['praat_f0','pyin_f0'])]
print(f'F0 features     : {len(f0)} -> {f0}')

no_mfcc = [f for f in feat_cols if f not in mfcc]
print(f'No-MFCC set     : {len(no_mfcc)} features')

no_bottom2 = [f for f in feat_cols if f not in {'dmfcc_13_mean','dmfcc_07_mean'}]
print(f'110f (CNN-ready): {len(no_bottom2)} features')

print('\nExisting training CSVs:')
for f in sorted(glob.glob(f'{BASE}/features/training_*.csv')):
    print(f'  {os.path.basename(f)}')
