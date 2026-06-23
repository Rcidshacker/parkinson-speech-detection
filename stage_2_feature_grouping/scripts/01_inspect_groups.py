import pandas as pd
import os

# Paths to your extracted feature files
COMPARE_8K = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project\features\features_compare_8k.csv"
MPOWER_CSV = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project\Dataset\mPower_Merged\mPower_final_labeled_data.csv"

def inspect():
    frames = []
    
    # 1. Load the OpenSMILE extracted features (Neurovoz, PC-GITA, VOICED)
    if os.path.exists(COMPARE_8K):
        df_c = pd.read_csv(COMPARE_8K, usecols=['dataset', 'label_binary'])
        df_c = df_c.rename(columns={'label_binary': 'label'})
        frames.append(df_c)
    
    # 2. Load the Merged mPower features
    if os.path.exists(MPOWER_CSV):
        df_m = pd.read_csv(MPOWER_CSV, usecols=['label'])
        df_m['dataset'] = 'mPower'
        frames.append(df_m)

    if not frames:
        print("No feature files found. Please check your paths.")
        return

    df = pd.concat(frames)

    # 3. Generate the Group Summary Table
    summary = df.groupby('dataset')['label'].value_counts().unstack(fill_value=0)
    summary.columns = ['HC (0)', 'PD (1)']
    summary['Total'] = summary['HC (0)'] + summary['PD (1)']
    summary['PD %'] = (summary['PD (1)'] / summary['Total'] * 100).round(2)

    print("\n" + "="*60)
    print("      PROJECT DATASET SUMMARY (Sustained Vowel /a/)")
    print("="*60)
    print(summary)
    print("="*60)
    print(f"GRAND TOTAL SAMPLES: {summary['Total'].sum():,}")
    print("="*60 + "\n")

if __name__ == "__main__":
    inspect()