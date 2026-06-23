# Quick Reference for Future Sessions

## GitHub Repository
**URL:** https://github.com/Rcidshacker/pd-crosslingual-analysis

## How to Push Updates

### Option 1: Bash Script
```bash
cd /home/z/my-project/download/pd_analysis
./git_push.sh "Your commit message"
```

### Option 2: Python
```python
import sys
sys.path.append('/home/z/my-project/download/pd_analysis')
from git_helper import push_to_github

push_to_github("Your commit message")
```

### Option 3: Direct Git Commands
```bash
cd /home/z/my-project/download/pd_analysis
git add .
git commit -m "Your message"
git push origin main
```

---

## Project Status (2026-03-24)

### Completed Tasks
| # | Task | Output |
|---|------|--------|
| 1 | Balanced datasets (vowel-matched /a/) | `data/features_*_balanced_*.csv` |
| 3 | Feature type-wise analysis | `results/feature_type_*.csv`, `images/*.png` |

### Pending Tasks
| # | Task | Notes |
|---|------|-------|
| 2 | RF feature importance validation | Optional - low priority |
| 4 | CNN with 11×10 feature matrix | Needs implementation |
| 5 | OpenSMILE eGeMAPS | Script ready, needs raw audio |
| 6 | Report writing | After experiments |

---

## Key Finding
**Biomarkers (11 features) outperform full feature set (112 features) cross-lingually:**
- Phonatory AUC: 0.729
- All Features AUC: 0.630

---

## Files Location
```
/home/z/my-project/download/pd_analysis/
├── scripts/         # Analysis scripts
├── opensmile/       # OpenSMILE extraction
├── data/            # Balanced datasets
├── results/         # Analysis results
├── images/          # Plots
├── git_push.sh      # Bash push helper
├── git_helper.py    # Python push helper
└── README.md        # Project documentation
```

---

## For New Chat Sessions

Just tell the AI:
> "Continue working on the PD cross-lingual detection project. 
> The repo is at /home/z/my-project/download/pd_analysis/
> Use git_helper.py to push updates to GitHub."

The AI can then:
```python
exec(open('/home/z/my-project/download/pd_analysis/git_helper.py').read())
push_to_github("New updates")
```
