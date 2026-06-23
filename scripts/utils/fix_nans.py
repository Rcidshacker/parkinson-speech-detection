import os
import pandas as pd

features_dir = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project\features"

# Toggle this! 
# 'columns' = drop the feature for everyone
# 'rows' = drop the single audio file that failed
DROP_TARGET = 'columns' 

print("Scanning for NaNs...")

for file in os.listdir(features_dir):
    if file.endswith(".csv") and "features_" in file:
        filepath = os.path.join(features_dir, file)
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
            
        nan_count = df.isna().sum().sum()
        
        if nan_count > 0:
            print(f"\n--- Fixing {file} ---")
            print(f"Total NaNs found: {nan_count}")
            
            if DROP_TARGET == 'columns':
                # Identify which columns are the culprits
                nan_cols = df.columns[df.isna().any()].tolist()
                print(f"Dropping Columns: {nan_cols}")
                df_clean = df.dropna(axis=1)
                
            elif DROP_TARGET == 'rows':
                # Identify which rows are the culprits
                nan_rows = df[df.isna().any(axis=1)]['file'].tolist()
                print(f"Dropping Rows (Audio files): {nan_rows}")
                df_clean = df.dropna(axis=0)

            print(f"Shape before: {df.shape} | Shape after: {df_clean.shape}")
            
            # Overwrite the file with the clean data
            df_clean.to_csv(filepath, index=False)
            print("File cleaned and overwritten successfully.")

print("\nDone! All datasets are now ML-ready.")