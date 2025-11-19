import pytest
from unittest.mock import MagicMock, patch
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.exceptions import CheckpointError, CheckpointNotFoundError


class TestCheckpointHubSync:
    @pytest.fixture
    def mock_repo_manager(self):
        repo_mgr = MagicMock()
        repo_mgr._api = MagicMock()
        return repo_mgr

    @pytest.fixture
    def checkpoint_manager(self, mock_repo_manager, tmp_path):
        return CheckpointManager(
            repo_manager=mock_repo_manager,
            local_dir=str(tmp_path / "checkpoints"),
        )

    def test_upload_specific_checkpoint(self, checkpoint_manager, mock_repo_manager):
        # Create a fake checkpoint
        checkpoint_dir = checkpoint_manager.local_dir / "test-ckpt"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "checkpoint.pt").write_text("fake checkpoint")
        (checkpoint_dir / "metadata.json").write_text('{"epoch": 1}')
        
        # Upload
        checkpoint_manager.upload_to_hub("user/repo", "test-ckpt")
        
        # Verify API was called
        assert mock_repo_manager._api.upload_file.called

    def test_upload_nonexistent_checkpoint(self, checkpoint_manager):
        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.upload_to_hub("user/repo", "nonexistent")

    @patch("huggingface_hub.snapshot_download")
    def test_download_from_hub(self, mock_snapshot, checkpoint_manager, tmp_path):
        # Setup mock
        fake_cache = tmp_path / "cache"
        fake_cache.mkdir()
        fake_ckpt_dir = fake_cache / "checkpoints" / "test-ckpt"
        fake_ckpt_dir.mkdir(parents=True)
        (fake_ckpt_dir / "checkpoint.pt").write_text("downloaded")
        
        mock_snapshot.return_value = str(fake_cache)
        
        # Download
        result = checkpoint_manager.download_from_hub("user/repo", "test-ckpt")
        
        # Verify
        assert "test-ckpt" in result
        mock_snapshot.assert_called_once()
