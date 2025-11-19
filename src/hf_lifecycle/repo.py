"""
Repository management module for HuggingFace Lifecycle Manager.
"""
import logging
from typing import Optional, List, Union

from huggingface_hub import HfApi, create_repo, delete_repo, update_repo_visibility
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from hf_lifecycle.auth import AuthManager
from hf_lifecycle.exceptions import RepositoryError

# Configure logging
logger = logging.getLogger(__name__)

class RepoManager:
    """
    Manages interactions with HuggingFace Hub repositories.
    """

    def __init__(self, auth_manager: AuthManager):
        """
        Initialize the RepoManager.

        Args:
            auth_manager: An authenticated AuthManager instance.
        """
        self._auth = auth_manager
        self._api = HfApi(token=self._auth.get_token())

    def create_repo(
        self, 
        repo_id: str, 
        repo_type: str = "model", 
        private: bool = True, 
        exist_ok: bool = False
    ) -> str:
        """
        Create a new repository on the HuggingFace Hub.

        Args:
            repo_id: The ID of the repository (e.g., "username/repo-name").
            repo_type: The type of repository ("model", "dataset", "space").
            private: Whether the repository should be private.
            exist_ok: If True, do not raise error if repo already exists.

        Returns:
            The URL of the created repository.

        Raises:
            RepositoryError: If creation fails.
        """
        token = self._auth.get_token()
        if not token:
            raise RepositoryError("Authentication required to create repository.")

        try:
            url = create_repo(
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                private=private,
                exist_ok=exist_ok
            )
            logger.info(f"Created {repo_type} repository: {url}")
            return url
        except Exception as e:
            raise RepositoryError(f"Failed to create repository {repo_id}: {e}")

    def delete_repo(self, repo_id: str, repo_type: str = "model") -> None:
        """
        Delete a repository from the HuggingFace Hub.

        Args:
            repo_id: The ID of the repository.
            repo_type: The type of repository.

        Raises:
            RepositoryError: If deletion fails.
        """
        token = self._auth.get_token()
        if not token:
            raise RepositoryError("Authentication required to delete repository.")

        try:
            delete_repo(repo_id=repo_id, token=token, repo_type=repo_type)
            logger.info(f"Deleted {repo_type} repository: {repo_id}")
        except Exception as e:
            raise RepositoryError(f"Failed to delete repository {repo_id}: {e}")

    def list_repos(self, username: Optional[str] = None) -> List[str]:
        """
        List repositories (models and datasets) for a user or organization.

        Args:
            username: The username or organization to list repos for. 
                      If None, lists for the authenticated user.

        Returns:
            A list of repository IDs.
        """
        try:
            models = self._api.list_models(author=username)
            datasets = self._api.list_datasets(author=username)
            
            repo_ids = [model.modelId for model in models]
            repo_ids.extend([dataset.id for dataset in datasets])
            
            return repo_ids
        except Exception as e:
            raise RepositoryError(f"Failed to list repositories: {e}")

    def update_card(self, repo_id: str, content: str, repo_type: str = "model") -> None:
        """
        Update the README.md (Model Card) of a repository.

        Args:
            repo_id: The ID of the repository.
            content: The content to write to README.md.
            repo_type: The type of repository.
        """
        token = self._auth.get_token()
        if not token:
            raise RepositoryError("Authentication required to update repository card.")

        try:
            self._api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message="Update README.md via hf_lifecycle"
            )
            logger.info(f"Updated README.md for {repo_id}")
        except Exception as e:
            raise RepositoryError(f"Failed to update card for {repo_id}: {e}")

    def create_branch(self, repo_id: str, branch: str, repo_type: str = "model") -> None:
        """
        Create a git branch in the repository.

        Args:
            repo_id: The ID of the repository.
            branch: The name of the branch to create.
            repo_type: The type of repository.
        """
        token = self._auth.get_token()
        if not token:
            raise RepositoryError("Authentication required to create branch.")

        try:
            self._api.create_branch(
                repo_id=repo_id,
                branch=branch,
                repo_type=repo_type
            )
            logger.info(f"Created branch {branch} in {repo_id}")
        except Exception as e:
            raise RepositoryError(f"Failed to create branch {branch} in {repo_id}: {e}")

    def file_exists(
        self, 
        repo_id: str, 
        filename: str, 
        repo_type: str = "model", 
        revision: Optional[str] = None
    ) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            repo_id: The ID of the repository.
            filename: The path to the file in the repo.
            repo_type: The type of repository.
            revision: The git revision (branch, tag, commit).

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            self._api.get_file_metadata(
                repo_id=repo_id,
                path=filename,
                repo_type=repo_type,
                revision=revision
            )
            return True
        except (RepositoryNotFoundError, HfHubHTTPError):
            return False
        except Exception as e:
            logger.warning(f"Error checking file existence: {e}")
            return False
