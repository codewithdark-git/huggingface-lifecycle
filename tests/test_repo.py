import pytest
from unittest.mock import MagicMock, patch
from hf_lifecycle.repo import RepoManager, RepositoryError
from hf_lifecycle.auth import AuthManager

class TestRepoManager:
    @pytest.fixture
    def mock_auth(self):
        auth = MagicMock(spec=AuthManager)
        auth.get_token.return_value = "mock_token"
        return auth

    @pytest.fixture
    def repo_manager(self, mock_auth):
        with patch("hf_lifecycle.repo.HfApi") as mock_api_cls:
            manager = RepoManager(mock_auth)
            manager._api = mock_api_cls.return_value
            return manager

    @patch("hf_lifecycle.repo.create_repo")
    def test_create_repo_success(self, mock_create, repo_manager):
        mock_create.return_value = "https://huggingface.co/user/repo"
        url = repo_manager.create_repo("user/repo")
        assert url == "https://huggingface.co/user/repo"
        mock_create.assert_called_once_with(
            repo_id="user/repo",
            token="mock_token",
            repo_type="model",
            private=True,
            exist_ok=False
        )

    @patch("hf_lifecycle.repo.create_repo")
    def test_create_repo_failure(self, mock_create, repo_manager):
        mock_create.side_effect = Exception("API Error")
        with pytest.raises(RepositoryError):
            repo_manager.create_repo("user/repo")

    @patch("hf_lifecycle.repo.delete_repo")
    def test_delete_repo_success(self, mock_delete, repo_manager):
        repo_manager.delete_repo("user/repo")
        mock_delete.assert_called_once_with(
            repo_id="user/repo",
            token="mock_token",
            repo_type="model"
        )

    def test_list_repos(self, repo_manager):
        mock_model = MagicMock()
        mock_model.modelId = "user/model1"
        
        mock_dataset = MagicMock()
        mock_dataset.id = "user/dataset1"
        
        repo_manager._api.list_models.return_value = [mock_model]
        repo_manager._api.list_datasets.return_value = [mock_dataset]
        
        repos = repo_manager.list_repos("user")
        assert "user/model1" in repos
        assert "user/dataset1" in repos

    def test_update_card(self, repo_manager):
        repo_manager.update_card("user/repo", "# README")
        repo_manager._api.upload_file.assert_called_once()

    def test_create_branch(self, repo_manager):
        repo_manager.create_branch("user/repo", "dev")
        repo_manager._api.create_branch.assert_called_once_with(
            repo_id="user/repo",
            branch="dev",
            repo_type="model"
        )

    def test_file_exists_true(self, repo_manager):
        repo_manager._api.get_file_metadata.return_value = {}
        assert repo_manager.file_exists("user/repo", "config.json") is True

    def test_file_exists_false(self, repo_manager):
        from huggingface_hub.utils import RepositoryNotFoundError
        repo_manager._api.get_file_metadata.side_effect = RepositoryNotFoundError("Not found")
        assert repo_manager.file_exists("user/repo", "missing.json") is False
