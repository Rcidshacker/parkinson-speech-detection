import os
import pandas as pd

features_dir = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project\features"

for file in os.listdir(features_dir):
    if file.endswith(".csv") and "features_" in file:
        filepath = os.path.join(features_dir, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"--- {file} ---")
        try:
            # low_memory=False prevents warnings on huge ComParE files
            df = pd.read_csv(filepath, low_memory=False) 
            print(f"Size:   {size_mb:.2f} MB")
            print(f"Shape:  {df.shape[0]} rows, {df.shape[1]} columns")
            if 'label_binary' in df.columns:
                print(f"Labels: {df['label_binary'].value_counts().to_dict()}")
            print(f"NaNs:   {df.isna().sum().sum()}")
        except Exception as e:
            print(f"Error reading file: {e}")
        print("-" * 40)