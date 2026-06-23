"""
Git Push Helper
===============
Quick way to push updates to GitHub from Python.

Usage:
    from git_helper import push_to_github
    push_to_github("Your commit message")
    
Or run directly:
    python git_helper.py "Your commit message"
"""

import subprocess
import os
from datetime import datetime

REPO_PATH = "/home/z/my-project/download/pd_analysis"
GITHUB_URL = "https://github.com/Rcidshacker/pd-crosslingual-analysis"


def run_git_command(cmd, cwd=REPO_PATH):
    """Run a git command and return output."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def has_changes():
    """Check if there are uncommitted changes."""
    code, out, _ = run_git_command("git status --porcelain")
    return len(out.strip()) > 0


def push_to_github(commit_message=None):
    """Add, commit, and push changes to GitHub."""
    
    os.chdir(REPO_PATH)
    
    if not has_changes():
        print("✓ No changes to commit.")
        return True
    
    if commit_message is None:
        commit_message = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print("=== Git Status ===")
    run_git_command("git status -s")
    code, out, _ = run_git_command("git status -s")
    print(out)
    
    print("=== Adding files ===")
    run_git_command("git add .")
    
    print("=== Committing ===")
    code, out, err = run_git_command(f'git commit -m "{commit_message}"')
    print(out)
    if code != 0 and err:
        print(err)
    
    print("=== Pushing to GitHub ===")
    code, out, err = run_git_command("git push origin main")
    print(out)
    if code != 0 and err:
        print(err)
        return False
    
    print(f"\n✓ Pushed successfully!")
    print(f"  Repository: {GITHUB_URL}")
    return True


def get_repo_info():
    """Get repository information."""
    print("=== Repository Info ===")
    print(f"Local path: {REPO_PATH}")
    print(f"GitHub URL: {GITHUB_URL}")
    print("")
    
    code, out, _ = run_git_command("git log --oneline -3")
    print("Recent commits:")
    print(out)
    
    code, out, _ = run_git_command("git remote -v")
    print("Remote:")
    print(out)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        msg = None
    
    push_to_github(msg)
