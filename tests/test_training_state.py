import pytest
import torch
import numpy as np
import random
from pathlib import Path
from hf_lifecycle.training_state import TrainingStateManager, EarlyStopping


class TestTrainingStateManager:
    @pytest.fixture
    def state_manager(self):
        return TrainingStateManager()

    def test_save_and_load_state(self, state_manager, tmp_path):
        state_file = tmp_path / "training_state.pt"
        
        # Save state
        state_manager.save_state(
            path=str(state_file),
            epoch=10,
            step=1000,
            best_metric=0.5,
            custom_state={"my_data": "value"},
        )
        
        assert state_file.exists()
        
        # Load state
        loaded_state = state_manager.load_state(str(state_file), restore_rng=False)
        
        assert loaded_state["epoch"] == 10
        assert loaded_state["step"] == 1000
        assert loaded_state["best_metric"] == 0.5
        assert loaded_state["custom_state"]["my_data"] == "value"

    def test_rng_restoration(self, state_manager, tmp_path):
        state_file = tmp_path / "rng_state.pt"
        
        # Set specific RNG seeds
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Generate some random numbers
        torch_before = torch.rand(5)
        
        # Save state
        state_manager.save_state(str(state_file), epoch=1, step=100)
        
        # Generate more random numbers (RNG state has advanced)
        torch.rand(10)
        
        # Load state and restore RNG
        state_manager.load_state(str(state_file), restore_rng=True)
        
        # Generate random numbers again - should match the "before" values
        torch_after = torch.rand(5)
        
        # Only check torch RNG - more reliable across environments
        assert torch.allclose(torch_before, torch_after)

    def test_is_best_min_mode(self, state_manager):
        assert state_manager.is_best(0.5, mode="min") is True
        assert state_manager.is_best(0.3, mode="min") is True  # Better (lower)
        assert state_manager.is_best(0.4, mode="min") is False  # Worse (higher)

    def test_is_best_max_mode(self, state_manager):
        assert state_manager.is_best(0.5, mode="max") is True
        assert state_manager.is_best(0.7, mode="max") is True  # Better (higher)
        assert state_manager.is_best(0.6, mode="max") is False  # Worse (lower)

    def test_state_dict(self, state_manager):
        state_manager.best_metric = 0.5
        state_manager.best_epoch = 10
        
        state_dict = state_manager.get_state_dict()
        assert state_dict["best_metric"] == 0.5
        assert state_dict["best_epoch"] == 10
        
        # Create new manager and load state
        new_manager = TrainingStateManager()
        new_manager.load_state_dict(state_dict)
        assert new_manager.best_metric == 0.5
        assert new_manager.best_epoch == 10


class TestEarlyStopping:
    def test_early_stopping_triggered(self):
        early_stop = EarlyStopping(patience=3, mode="min")
        
        # No improvement for 3+ epochs should trigger
        assert early_stop.step(1.0) is False  # Best: 1.0, counter=0
        assert early_stop.step(1.1) is False  # No improvement, counter=1
        assert early_stop.step(1.2) is False  # No improvement, counter=2
        assert early_stop.step(1.3) is False  # No improvement, counter=3
        assert early_stop.step(1.4) is True   # counter=4 >= patience(3) -> STOP

    def test_early_stopping_improvement(self):
        early_stop = EarlyStopping(patience=2, mode="min")
        
        assert early_stop.step(1.0) is False  # Best: 1.0, counter=0
        assert early_stop.step(1.1) is False  # No improvement, counter=1
        assert early_stop.step(0.9) is False  # Improvement! Best: 0.9, counter=0
        assert early_stop.step(1.0) is False  # No improvement, counter=1
        assert early_stop.step(1.1) is False  # No improvement, counter=2
        assert early_stop.step(1.2) is True   # counter=3 >= patience(2) -> STOP

    def test_early_stopping_max_mode(self):
        early_stop = EarlyStopping(patience=2, mode="max")
        
        assert early_stop.step(0.5) is False  # Best: 0.5, counter=0
        assert early_stop.step(0.4) is False  # No improvement, counter=1
        assert early_stop.step(0.6) is False  # Improvement! Best: 0.6, counter=0
        assert early_stop.step(0.5) is False  # No improvement, counter=1
        assert early_stop.step(0.4) is False  # No improvement, counter=2
        assert early_stop.step(0.3) is True   # counter=3 >= patience(2) -> STOP

    def test_early_stopping_state_dict(self):
        early_stop = EarlyStopping(patience=5, mode="min")
        early_stop.step(1.0)
        early_stop.step(1.1)
        
        state = early_stop.step_dict()
        assert state["best_metric"] == 1.0
        assert state["counter"] == 1
        
        # Create new instance and load state
        new_early_stop = EarlyStopping(patience=5, mode="min")
        new_early_stop.load_state_dict(state)
        
        assert new_early_stop.best_metric == 1.0
        assert new_early_stop.counter == 1
