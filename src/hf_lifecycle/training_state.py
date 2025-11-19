"""
Training state management for reproducible training workflows.
"""
import torch
import numpy as np
import random
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TrainingStateManager:
    """
    Manages complete training state for exact resumption.
    """

    def __init__(self):
        """Initialize the TrainingStateManager."""
        self.best_metric = None
        self.best_epoch = None

    def save_state(
        self,
        path: str,
        epoch: int,
        step: int,
        best_metric: Optional[float] = None,
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save complete training state including RNG states.

        Args:
            path: Path to save state file.
            epoch: Current epoch.
            step: Current step.
            best_metric: Best metric value so far.
            custom_state: Additional custom state to save.
        """
        state = {
            "epoch": epoch,
            "step": step,
            "best_metric": best_metric,
            "custom_state": custom_state or {},
            # RNG states
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # CUDA RNG state if available
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Save state
        torch.save(state, path)
        logger.info(f"Saved training state to {path}")

    def load_state(
        self, path: str, restore_rng: bool = True
    ) -> Dict[str, Any]:
        """
        Load training state.

        Args:
            path: Path to state file.
            restore_rng: Whether to restore RNG states.

        Returns:
            Dictionary containing training state.
        """
        # Load with weights_only=False for compatibility with saved RNG states
        # PyTorch 2.6+ requires explicit allowlist for numpy types
        try:
            import torch.serialization
            # Add numpy types to safe globals for PyTorch 2.6+
            with torch.serialization.safe_globals([
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                np.core.multiarray.scalar,
            ]):
                state = torch.load(path, weights_only=False)
        except (AttributeError, TypeError):
            # Fallback for older PyTorch versions
            state = torch.load(path, weights_only=False)

        if restore_rng:
            # Restore RNG states
            torch.set_rng_state(state["torch_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])

            # Restore CUDA RNG state if available
            if torch.cuda.is_available() and "cuda_rng_state" in state:
                torch.cuda.set_rng_state(state["cuda_rng_state"])
                if "cuda_rng_state_all" in state:
                    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

            logger.info("Restored RNG states for reproducible training")

        logger.info(f"Loaded training state from {path}")
        return state

    def is_best(
        self, current_metric: float, mode: str = "min"
    ) -> bool:
        """
        Check if current metric is the best so far.

        Args:
            current_metric: Current metric value.
            mode: 'min' for lower is better, 'max' for higher is better.

        Returns:
            True if current metric is best.
        """
        if self.best_metric is None:
            self.best_metric = current_metric
            return True

        if mode == "min":
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric

        return is_best

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current state as dictionary.

        Returns:
            State dictionary.
        """
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from dictionary.

        Args:
            state_dict: State dictionary.
        """
        self.best_metric = state_dict.get("best_metric")
        self.best_epoch = state_dict.get("best_epoch")


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for lower is better, 'max' for higher is better.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self.mode == "min":
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement"
                )
                return True

        return False

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        return {
            "best_metric": self.best_metric,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.best_metric = state_dict.get("best_metric")
        self.counter = state_dict.get("counter", 0)
        self.should_stop = state_dict.get("should_stop", False)
