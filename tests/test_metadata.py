import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from hf_lifecycle.metadata import MetadataTracker


class TestMetadataTracker:
    @pytest.fixture
    def tracker(self):
        return MetadataTracker()

    def test_init(self, tracker):
        assert "created_at" in tracker.metadata
        assert "system" in tracker.metadata
        assert "environment" in tracker.metadata
        assert "hyperparameters" in tracker.metadata
        assert "metrics" in tracker.metadata

    def test_capture_system_info(self, tracker):
        system_info = tracker.capture_system_info()
        
        assert "platform" in system_info
        assert "python_version" in system_info
        assert "os" in system_info
        assert "cpu_count" in system_info
        assert "cuda_available" in system_info

    @patch("subprocess.check_output")
    def test_capture_environment(self, mock_subprocess, tracker):
        # Mock git commands
        mock_subprocess.side_effect = [
            b"abc123def456\n",  # git commit
            b"main\n",  # git branch
            b"",  # git status (clean)
        ]
        
        env_info = tracker.capture_environment()
        
        assert "packages" in env_info
        assert "git" in env_info
        assert env_info["git"]["commit"] == "abc123def456"
        assert env_info["git"]["branch"] == "main"
        assert env_info["git"]["dirty"] is False

    def test_track_hyperparameters(self, tracker):
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        
        tracker.track_hyperparameters(params)
        
        assert tracker.metadata["hyperparameters"]["learning_rate"] == 0.001
        assert tracker.metadata["hyperparameters"]["batch_size"] == 32

    def test_track_metrics_with_step(self, tracker):
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        tracker.track_metrics(metrics, step=100)
        
        assert "history" in tracker.metadata["metrics"]
        assert len(tracker.metadata["metrics"]["history"]) == 1
        assert tracker.metadata["metrics"]["history"][0]["step"] == 100
        assert tracker.metadata["metrics"]["history"][0]["metrics"]["loss"] == 0.5

    def test_track_metrics_final(self, tracker):
        metrics = {"loss": 0.3, "accuracy": 0.95}
        
        tracker.track_metrics(metrics)
        
        assert "final" in tracker.metadata["metrics"]
        assert tracker.metadata["metrics"]["final"]["loss"] == 0.3

    def test_add_custom(self, tracker):
        tracker.add_custom("model_name", "bert-base")
        tracker.add_custom("dataset", "imdb")
        
        assert tracker.metadata["custom"]["model_name"] == "bert-base"
        assert tracker.metadata["custom"]["dataset"] == "imdb"

    def test_save_and_load_metadata(self, tracker, tmp_path):
        # Add some data
        tracker.track_hyperparameters({"lr": 0.001})
        tracker.track_metrics({"loss": 0.5})
        
        # Save
        metadata_path = tmp_path / "metadata.json"
        tracker.save_metadata(str(metadata_path))
        
        assert metadata_path.exists()
        
        # Load
        new_tracker = MetadataTracker()
        loaded = new_tracker.load_metadata(str(metadata_path))
        
        assert loaded["hyperparameters"]["lr"] == 0.001
        assert loaded["metrics"]["final"]["loss"] == 0.5

    def test_get_summary(self, tracker):
        tracker.capture_system_info()
        tracker.track_hyperparameters({"lr": 0.001, "epochs": 10})
        tracker.track_metrics({"loss": 0.3, "accuracy": 0.95})
        
        summary = tracker.get_summary()
        
        assert "EXPERIMENT METADATA SUMMARY" in summary
        assert "SYSTEM:" in summary
        assert "HYPERPARAMETERS:" in summary
        assert "FINAL METRICS:" in summary

    def test_to_model_card_section(self, tracker):
        tracker.capture_system_info()
        tracker.track_hyperparameters({"learning_rate": 0.001, "batch_size": 32})
        tracker.track_metrics({"accuracy": 0.95, "f1": 0.93})
        
        card = tracker.to_model_card_section()
        
        assert "## Training Details" in card
        assert "### System Information" in card
        assert "### Hyperparameters" in card
        assert "### Results" in card
        assert "learning_rate" in card
        assert "0.95" in card
