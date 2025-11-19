import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from hf_lifecycle.dataset import DatasetManager
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.exceptions import DatasetError

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class TestDatasetManager:
    @pytest.fixture
    def mock_repo_manager(self):
        repo_mgr = MagicMock(spec=RepoManager)
        repo_mgr._api = MagicMock()
        repo_mgr.create_repo.return_value = "https://huggingface.co/datasets/user/dataset"
        return repo_mgr

    @pytest.fixture
    def dataset_manager(self, mock_repo_manager):
        return DatasetManager(mock_repo_manager)

    def test_create_dataset(self, dataset_manager, mock_repo_manager):
        url = dataset_manager.create_dataset("user/dataset")
        assert url == "https://huggingface.co/datasets/user/dataset"
        mock_repo_manager.create_repo.assert_called_with(
            repo_id="user/dataset",
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )

    def test_upload_file(self, dataset_manager, mock_repo_manager):
        dataset_manager.upload_file("user/dataset", "local.txt", "remote.txt")
        mock_repo_manager._api.upload_file.assert_called_once()

    def test_upload_folder(self, dataset_manager, mock_repo_manager):
        dataset_manager.upload_folder("user/dataset", "local_folder")
        mock_repo_manager._api.upload_folder.assert_called_once()

    def test_upload_dataframe_parquet(self, dataset_manager, mock_repo_manager, tmp_path):
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        
        with patch("hf_lifecycle.dataset.tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(tmp_path)
            
            dataset_manager.upload_dataframe(
                "user/dataset", df, "data.parquet", format="parquet"
            )
            
            mock_repo_manager._api.upload_file.assert_called_once()
            # Verify it was uploaded as parquet
            args = mock_repo_manager._api.upload_file.call_args
            assert "data.parquet" in args.kwargs["path_in_repo"]

    def test_upload_dataframe_csv(self, dataset_manager, mock_repo_manager, tmp_path):
        df = pd.DataFrame({"col1": [1, 2]})
        
        with patch("hf_lifecycle.dataset.tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(tmp_path)
            
            dataset_manager.upload_dataframe(
                "user/dataset", df, "data.csv", format="csv"
            )
            
            mock_repo_manager._api.upload_file.assert_called_once()

    def test_upload_dataframe_invalid_format(self, dataset_manager):
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(DatasetError):
            dataset_manager.upload_dataframe(
                "user/dataset", df, "data.xyz", format="xyz"
            )

    @patch("huggingface_hub.snapshot_download")
    def test_download_dataset(self, mock_download, dataset_manager):
        mock_download.return_value = "/cache/dataset"
        
        path = dataset_manager.download_dataset("user/dataset", "local_dir")
        
        assert path == "/cache/dataset"
        mock_download.assert_called_with(
            repo_id="user/dataset",
            repo_type="dataset",
            local_dir="local_dir",
            allow_patterns=None,
            revision=None,
        )

    def test_delete_dataset(self, dataset_manager, mock_repo_manager):
        dataset_manager.delete_dataset("user/dataset")
        mock_repo_manager.delete_repo.assert_called_with(
            repo_id="user/dataset", repo_type="dataset"
        )
