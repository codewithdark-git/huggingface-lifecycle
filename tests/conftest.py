import pytest
from unittest.mock import MagicMock

@pytest.fixture
def sample_fixture():
    return "sample"

@pytest.fixture
def mock_hf_api(monkeypatch):
    """Mock the HuggingFace Hub API."""
    mock_api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: mock_api)
    return mock_api
