"""
Dataset management for HuggingFace Lifecycle Manager.
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Any
import logging
import tempfile
import shutil

from hf_lifecycle.repo import RepoManager
from hf_lifecycle.exceptions import DatasetError

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages interactions with HuggingFace Hub datasets.
    """

    def __init__(self, repo_manager: RepoManager):
        """
        Initialize the DatasetManager.

        Args:
            repo_manager: Repository manager instance.
        """
        self.repo_manager = repo_manager

    def create_dataset(
        self, repo_id: str, private: bool = True, exist_ok: bool = False
    ) -> str:
        """
        Create a new dataset repository.

        Args:
            repo_id: Repository ID (username/dataset-name).
            private: Whether the dataset should be private.
            exist_ok: If True, do not raise error if repo already exists.

        Returns:
            URL of the created dataset.

        Raises:
            DatasetError: If creation fails.
        """
        try:
            url = self.repo_manager.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=exist_ok,
            )
            logger.info(f"Created dataset repository: {url}")
            return url
        except Exception as e:
            raise DatasetError(f"Failed to create dataset {repo_id}: {e}")

    def upload_file(
        self,
        repo_id: str,
        file_path: str,
        path_in_repo: str,
        commit_message: Optional[str] = None,
    ) -> None:
        """
        Upload a single file to a dataset repository.

        Args:
            repo_id: Repository ID.
            file_path: Local path to the file.
            path_in_repo: Path in the repository.
            commit_message: Git commit message.

        Raises:
            DatasetError: If upload fails.
        """
        try:
            self.repo_manager._api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or f"Upload {path_in_repo}",
            )
            logger.info(f"Uploaded {file_path} to {repo_id}")
        except Exception as e:
            raise DatasetError(f"Failed to upload file to {repo_id}: {e}")

    def upload_folder(
        self,
        repo_id: str,
        folder_path: str,
        path_in_repo: str = ".",
        commit_message: Optional[str] = None,
    ) -> None:
        """
        Upload a folder to a dataset repository.

        Args:
            repo_id: Repository ID.
            folder_path: Local path to the folder.
            path_in_repo: Path in the repository.
            commit_message: Git commit message.

        Raises:
            DatasetError: If upload fails.
        """
        try:
            self.repo_manager._api.upload_folder(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or f"Upload folder {folder_path}",
            )
            logger.info(f"Uploaded folder {folder_path} to {repo_id}")
        except Exception as e:
            raise DatasetError(f"Failed to upload folder to {repo_id}: {e}")

    def upload_dataframe(
        self,
        repo_id: str,
        df: pd.DataFrame,
        path_in_repo: str,
        format: str = "parquet",
        commit_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Upload a Pandas DataFrame to a dataset repository.

        Args:
            repo_id: Repository ID.
            df: Pandas DataFrame to upload.
            path_in_repo: Path in the repository (including extension).
            format: File format ('parquet', 'csv', 'json').
            commit_message: Git commit message.
            **kwargs: Additional arguments for pandas save methods.

        Raises:
            DatasetError: If upload fails.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = Path(tmp_dir) / Path(path_in_repo).name
                
                if format == "parquet":
                    df.to_parquet(file_path, **kwargs)
                elif format == "csv":
                    df.to_csv(file_path, index=False, **kwargs)
                elif format == "json":
                    df.to_json(file_path, orient="records", **kwargs)
                else:
                    raise ValueError(f"Unsupported format: {format}")

                self.upload_file(
                    repo_id=repo_id,
                    file_path=str(file_path),
                    path_in_repo=path_in_repo,
                    commit_message=commit_message or f"Upload dataframe as {format}",
                )
                
            logger.info(f"Uploaded DataFrame to {repo_id} as {format}")
        except Exception as e:
            raise DatasetError(f"Failed to upload DataFrame to {repo_id}: {e}")

    def download_dataset(
        self,
        repo_id: str,
        local_dir: str,
        allow_patterns: Optional[Union[List[str], str]] = None,
        revision: Optional[str] = None,
    ) -> str:
        """
        Download a dataset from HuggingFace Hub.

        Args:
            repo_id: Repository ID.
            local_dir: Local directory to download to.
            allow_patterns: Patterns of files to download.
            revision: Git revision.

        Returns:
            Path to the downloaded dataset.

        Raises:
            DatasetError: If download fails.
        """
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                revision=revision,
            )
            logger.info(f"Downloaded dataset {repo_id} to {local_dir}")
            return cache_dir
        except Exception as e:
            raise DatasetError(f"Failed to download dataset {repo_id}: {e}")

    def delete_dataset(self, repo_id: str) -> None:
        """
        Delete a dataset repository.

        Args:
            repo_id: Repository ID.

        Raises:
            DatasetError: If deletion fails.
        """
        try:
            self.repo_manager.delete_repo(repo_id=repo_id, repo_type="dataset")
            logger.info(f"Deleted dataset repository: {repo_id}")
        except Exception as e:
            raise DatasetError(f"Failed to delete dataset {repo_id}: {e}")
