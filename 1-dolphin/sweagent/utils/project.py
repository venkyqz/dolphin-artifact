from git import Repo

from sweagent.utils.telemetry import logger

def prepare_git_project(workspace, commit):
    # Check if workspace is a git repo and checkout to specified commit
    try:
        repo = Repo(workspace)
        if repo.git_dir and commit:  # Check if valid git repo and commit exists
            current = repo.head.commit.hexsha

            # Check if commit exists in local repo
            try:
                repo.commit(commit)
            except:
                # Fetch commit if not found locally
                # Fetch with depth 1 first to get the target commit
                repo.git.fetch("--depth", "1", "origin", commit)
            repo.git.checkout(commit)
            logger.info("Checked out from {} to commit {}", current, commit)
    except Exception as e:
        logger.warning("Failed to checkout git repo: {}", str(e))
        raise e
