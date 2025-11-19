import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.retention import KeepLastN, KeepBestM
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.exceptions import CheckpointNotFoundError, CheckpointCorruptedError
from unittest.mock import MagicMock


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def test_dir(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    yield checkpoint_dir
    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


@pytest.fixture
def mock_repo_manager():
    return MagicMock(spec=RepoManager)


@pytest.fixture
def checkpoint_manager(mock_repo_manager, test_dir):
    return CheckpointManager(
        repo_manager=mock_repo_manager,
        local_dir=str(test_dir),
        retention_policy=KeepLastN(3),
    )


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)


class TestCheckpointManager:
    def test_init(self, checkpoint_manager, test_dir):
        assert checkpoint_manager.local_dir == Path(test_dir)
        assert test_dir.exists()

    def test_save_basic(self, checkpoint_manager, model):
        path = checkpoint_manager.save(model, epoch=1, step=100)
        assert Path(path).exists()
        assert (Path(path) / "checkpoint.pt").exists()
        assert (Path(path) / "metadata.json").exists()

    def test_save_with_optimizer(self, checkpoint_manager, model, optimizer):
        path = checkpoint_manager.save(
            model, optimizer=optimizer, epoch=1, step=100
        )
        checkpoint = torch.load(Path(path) / "checkpoint.pt")
        assert "optimizer_state_dict" in checkpoint

    def test_save_with_metrics(self, checkpoint_manager, model):
        metrics = {"loss": 0.5, "accuracy": 0.95}
        path = checkpoint_manager.save(model, step=100, metrics=metrics)
        checkpoint = torch.load(Path(path) / "checkpoint.pt")
        assert checkpoint["metrics"] == metrics

    def test_save_custom_name(self, checkpoint_manager, model):
        path = checkpoint_manager.save(model, name="my-checkpoint")
        assert "my-checkpoint" in path

    def test_load(self, checkpoint_manager, model):
        # Save first
        checkpoint_manager.save(model, epoch=1, step=100)

        # Load
        new_model = SimpleModel()
        checkpoint = checkpoint_manager.load("checkpoint-step-100", model=new_model)

        assert checkpoint["epoch"] == 1
        assert checkpoint["step"] == 100

    def test_load_not_found(self, checkpoint_manager, model):
        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.load("nonexistent", model=model)

    def test_load_latest(self, checkpoint_manager, model):
        # Save multiple checkpoints
        checkpoint_manager.save(model, step=100)
        checkpoint_manager.save(model, step=200)
        checkpoint_manager.save(model, step=300)

        # Load latest
        checkpoint = checkpoint_manager.load_latest()
        assert checkpoint["step"] == 300

    def test_load_latest_no_checkpoints(self, checkpoint_manager):
        result = checkpoint_manager.load_latest()
        assert result is None

    def test_load_best(self, checkpoint_manager, model):
        # Save checkpoints with different metrics
        checkpoint_manager.save(model, step=100, metrics={"loss": 0.5})
        checkpoint_manager.save(model, step=200, metrics={"loss": 0.3})
        checkpoint_manager.save(model, step=300, metrics={"loss": 0.4})

        # Load best (lowest loss)
        checkpoint = checkpoint_manager.load_best("loss", mode="min")
        assert checkpoint["step"] == 200

    def test_load_best_max_mode(self, checkpoint_manager, model):
        # Save checkpoints with different metrics
        checkpoint_manager.save(model, step=100, metrics={"accuracy": 0.8})
        checkpoint_manager.save(model, step=200, metrics={"accuracy": 0.95})
        checkpoint_manager.save(model, step=300, metrics={"accuracy": 0.9})

        # Load best (highest accuracy)
        checkpoint = checkpoint_manager.load_best("accuracy", mode="max")
        assert checkpoint["step"] == 200

    def test_list_checkpoints(self, checkpoint_manager, model):
        # Save multiple checkpoints
        checkpoint_manager.save(model, step=100, metrics={"loss": 0.5})
        checkpoint_manager.save(model, step=200, metrics={"loss": 0.3})

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert all("name" in ckpt for ckpt in checkpoints)
        assert all("step" in ckpt for ckpt in checkpoints)

    def test_cleanup_dry_run(self, checkpoint_manager, model):
        # Save more checkpoints than retention policy allows
        for i in range(5):
            checkpoint_manager.save(model, step=100 * (i + 1))

        # Dry run cleanup
        deleted = checkpoint_manager.cleanup(dry_run=True)
        
        # Should identify 2 checkpoints to delete (keep last 3)
        assert len(deleted) == 2
        
        # Verify nothing was actually deleted
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 5

    def test_cleanup_actual(self, checkpoint_manager, model):
        # Save more checkpoints than retention policy allows
        for i in range(5):
            checkpoint_manager.save(model, step=100 * (i + 1))

        # Actual cleanup
        deleted = checkpoint_manager.cleanup(dry_run=False)
        
        # Should delete 2 checkpoints
        assert len(deleted) == 2
        
        # Verify only 3 remain
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_save_and_load_with_scheduler(self, checkpoint_manager, model, optimizer):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Save
        checkpoint_manager.save(
            model, optimizer=optimizer, scheduler=scheduler, step=100
        )
        
        # Load
        new_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
        checkpoint = checkpoint_manager.load(
            "checkpoint-step-100", scheduler=new_scheduler
        )
        
        assert "scheduler_state_dict" in checkpoint
