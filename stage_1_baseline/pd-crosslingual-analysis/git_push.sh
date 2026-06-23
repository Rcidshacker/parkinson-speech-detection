#!/bin/bash
# Git Push Helper Script
# ======================
# Run this script to push updates to GitHub
#
# Usage:
#   ./git_push.sh "Your commit message here"
#
# If no message provided, uses timestamp

cd /home/z/my-project/download/pd_analysis

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit."
    exit 0
fi

# Get commit message from argument or use timestamp
if [ -z "$1" ]; then
    MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
else
    MSG="$1"
fi

echo "=== Git Status ==="
git status -s

echo ""
echo "=== Committing ==="
git add .
git commit -m "$MSG"

echo ""
echo "=== Pushing to GitHub ==="
git push origin main

echo ""
echo "=== Done! ==="
echo "Repository: https://github.com/Rcidshacker/pd-crosslingual-analysis"
