import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from hf_lifecycle.model_registry import ModelRegistry
from hf_lifecycle.repo import RepoManager


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestModelRegistry:
    @pytest.fixture
    def mock_repo_manager(self):
        repo_mgr = MagicMock(spec=RepoManager)
        repo_mgr._api = MagicMock()
        repo_mgr.create_repo.return_value = "https://huggingface.co/user/model"
        return repo_mgr

    @pytest.fixture
    def model_registry(self, mock_repo_manager):
        return ModelRegistry(mock_repo_manager)

    @pytest.fixture
    def mock_model(self):
        return SimpleTestModel()

    def test_generate_model_card_basic(self, model_registry):
        card = model_registry.generate_model_card(
            model_name="test-model",
            description="A test model",
        )

        assert "test-model" in card
        assert "A test model" in card
        assert "---" in card  # YAML frontmatter
        assert "Usage" in card

    def test_generate_model_card_with_metrics(self, model_registry):
        metrics = {"accuracy": 0.95, "loss": 0.05}
        card = model_registry.generate_model_card(
            model_name="test-model",
            metrics=metrics,
        )

        assert "accuracy" in card
        assert "0.9500" in card
        assert "loss" in card

    def test_generate_model_card_with_datasets(self, model_registry):
        datasets = ["squad", "glue"]
        card = model_registry.generate_model_card(
            model_name="test-model",
            datasets=datasets,
        )

        assert "squad" in card
        assert "glue" in card

    def test_generate_model_card_with_architecture(self, model_registry):
        card = model_registry.generate_model_card(
            model_name="test-model",
            architecture="transformer",
        )

        assert "transformer" in card
        assert "Architecture" in card

    @patch("shutil.rmtree")
    @patch("hf_lifecycle.model_registry.Path.rglob")
    @patch("hf_lifecycle.model_registry.Path.mkdir")
    @patch("hf_lifecycle.model_registry.torch.save")
    def test_register_model(
        self, mock_save, mock_mkdir, mock_rglob, mock_rmtree, model_registry, mock_repo_manager, mock_model
    ):
        # Mock file iteration
        mock_rglob.return_value = []
        
        repo_url = model_registry.register_model(
            model=mock_model,
            repo_id="user/test-model",
            description="Test model",
        )

        assert repo_url == "https://huggingface.co/user/model"
        mock_repo_manager.create_repo.assert_called_once()
        mock_repo_manager.update_card.assert_called_once()

    @patch("shutil.rmtree")
    @patch("hf_lifecycle.model_registry.Path.rglob")
    @patch("hf_lifecycle.model_registry.Path.mkdir")
    @patch("hf_lifecycle.model_registry.torch.save")
    def test_register_model_with_metrics(self, mock_save, mock_mkdir, mock_rglob, mock_rmtree, model_registry, mock_repo_manager, mock_model):
        mock_rglob.return_value = []

        model_registry.register_model(
            model=mock_model,
            repo_id="user/test-model",
            metrics={"accuracy": 0.95},
        )

        # Verify model card was generated with metrics
        call_args = mock_repo_manager.update_card.call_args
        model_card = call_args[0][1]
        assert "accuracy" in model_card

    @patch("transformers.AutoModel")
    def test_load_model(self, mock_automodel, model_registry):
        mock_model = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model

        result = model_registry.load_model("user/test-model")

        assert result == mock_model
        mock_automodel.from_pretrained.assert_called_once_with(
            "user/test-model", revision=None
        )
