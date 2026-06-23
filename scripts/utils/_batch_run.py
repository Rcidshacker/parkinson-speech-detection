"""
Batch runner for dataset_wise_analysis_v2.py across all knowledge-driven feature sets.
Runs sequentially, then prints unified cross-dataset AUC summary.
"""
import subprocess, sys, os, glob, json
from datetime import datetime

BASE    = r"C:\Users\Lenovo\Desktop\Code\2026\BE mini project"
PYTHON  = os.path.join(BASE, "venv", "Scripts", "python.exe")
SCRIPT  = os.path.join(BASE, "scripts", "dataset_wise_analysis_v2.py")

SETS = [
    "training_biomarker_f0_23f.csv",
    "training_no_mfcc_60f.csv",
    "training_rf_top14_es.csv",
    "training_rf_top14_it.csv",
    "training_cnn_ready_110f.csv",
]

print(f"Starting batch at {datetime.now().strftime('%H:%M:%S')}")
print(f"Running {len(SETS)} feature sets\n")

for fname in SETS:
    data_file = os.path.join(BASE, "features", fname)
    label = fname.replace("training_", "").replace(".csv", "")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {label} ...")
    result = subprocess.run(
        [PYTHON, SCRIPT, "--data_file", data_file],
        capture_output=True, text=True
    )
    # Print last 10 lines of output (summary)
    lines = result.stdout.strip().splitlines()
    for line in lines[-12:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-300:]}")
    print()

print(f"\nBatch complete at {datetime.now().strftime('%H:%M:%S')}")
print("\nCollecting cross-dataset results...")

# Collect ES->IT and IT->ES AUC from all runs (best model per direction)
results_dir = os.path.join(BASE, "scripts", "results_dataset_wise_v2")
all_runs = {}

for run_dir in sorted(glob.glob(f"{results_dir}/*")):
    csv_path = os.path.join(run_dir, "csv_results", "analysis_summary.csv")
    if not os.path.exists(csv_path):
        continue
    import pandas as pd
    df = pd.read_csv(csv_path)
    label = os.path.basename(run_dir).rsplit("_20", 1)[0]  # strip timestamp

    es_it = df[df["Analysis"].str.contains("pc_gita.*voiced", regex=True, na=False)]
    it_es = df[df["Analysis"].str.contains("voiced.*pc_gita", regex=True, na=False)]

    best_es_it = es_it["AUC"].max() if len(es_it) > 0 else None
    best_it_es = it_es["AUC"].max() if len(it_es) > 0 else None
    best_model_es = es_it.loc[es_it["AUC"].idxmax(), "Model"] if len(es_it) > 0 else "—"
    best_model_it = it_es.loc[it_es["AUC"].idxmax(), "Model"] if len(it_es) > 0 else "—"

    all_runs[label] = {
        "es_it": best_es_it, "es_it_model": best_model_es,
        "it_es": best_it_es, "it_es_model": best_model_it,
    }

print(f"\n{'Label':<30} {'ES→IT AUC':>10} {'Best Model':<22} {'IT→ES AUC':>10} {'Best Model':<22}")
print("-" * 96)
for label, r in sorted(all_runs.items()):
    es = f"{r['es_it']:.4f}" if r['es_it'] else "—"
    it = f"{r['it_es']:.4f}" if r['it_es'] else "—"
    print(f"{label:<30} {es:>10} {r['es_it_model']:<22} {it:>10} {r['it_es_model']:<22}")
