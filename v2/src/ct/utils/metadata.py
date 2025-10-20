import subprocess

"""
src/utils/metadata.py
--------------------------------
This module provides utility functions to retrieve Git metadata.
* get_git_commit_hash: Returns the current Git commit hash.
* is_git_dirty: Checks if there are uncommitted changes in the Git repository.
"""

def get_git_commit_hash():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def is_git_dirty():
    try:
        result = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"])
        return result.returncode != 0
    except Exception:
        return False