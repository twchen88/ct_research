import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.metadata import get_git_commit_hash


if __name__ == "__main__":
    # Get the current git commit hash
    commit_hash = get_git_commit_hash()
    
    # Print the commit hash
    print(f"Current Git Commit Hash: {commit_hash}")
    
    # Save the commit hash to a file
    with open("sandbox/git_commit_hash.txt", "w") as f:
        f.write(commit_hash)
    
    print("Git commit hash saved to sandbox/git_commit_hash.txt")