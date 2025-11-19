"""
Retention policies for checkpoint management.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class RetentionPolicy(ABC):
    """Base class for checkpoint retention policies."""

    @abstractmethod
    def select_checkpoints_to_keep(
        self, checkpoints: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Select which checkpoints to keep.

        Args:
            checkpoints: List of checkpoint metadata dictionaries.
                Each dict should have 'name', 'epoch', 'step', 'metrics', etc.

        Returns:
            List of checkpoint names to keep.
        """
        pass


class KeepLastN(RetentionPolicy):
    """Keep the N most recent checkpoints."""

    def __init__(self, n: int = 3):
        """
        Initialize KeepLastN policy.

        Args:
            n: Number of most recent checkpoints to keep.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n

    def select_checkpoints_to_keep(
        self, checkpoints: List[Dict[str, Any]]
    ) -> List[str]:
        """Keep N most recent checkpoints based on step number."""
        if not checkpoints:
            return []

        # Sort by step in descending order
        sorted_ckpts = sorted(
            checkpoints, key=lambda x: x.get("step", 0), reverse=True
        )

        # Keep the first N
        to_keep = [ckpt["name"] for ckpt in sorted_ckpts[: self.n]]
        logger.info(f"KeepLastN({self.n}): Keeping {len(to_keep)} checkpoints")
        return to_keep


class KeepBestM(RetentionPolicy):
    """Keep the M best checkpoints based on a metric."""

    def __init__(
        self, m: int = 2, metric: str = "loss", mode: str = "min"
    ):
        """
        Initialize KeepBestM policy.

        Args:
            m: Number of best checkpoints to keep.
            metric: Metric name to compare.
            mode: 'min' for lower is better, 'max' for higher is better.
        """
        if m < 1:
            raise ValueError("m must be at least 1")
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.m = m
        self.metric = metric
        self.mode = mode

    def select_checkpoints_to_keep(
        self, checkpoints: List[Dict[str, Any]]
    ) -> List[str]:
        """Keep M best checkpoints based on the specified metric."""
        if not checkpoints:
            return []

        # Filter checkpoints that have the metric
        valid_ckpts = [
            ckpt
            for ckpt in checkpoints
            if ckpt.get("metrics") and self.metric in ckpt["metrics"]
        ]

        if not valid_ckpts:
            logger.warning(
                f"No checkpoints found with metric '{self.metric}', keeping none"
            )
            return []

        # Sort by metric
        reverse = self.mode == "max"
        sorted_ckpts = sorted(
            valid_ckpts,
            key=lambda x: x["metrics"][self.metric],
            reverse=reverse,
        )

        # Keep the first M
        to_keep = [ckpt["name"] for ckpt in sorted_ckpts[: self.m]]
        logger.info(
            f"KeepBestM({self.m}, {self.metric}, {self.mode}): Keeping {len(to_keep)} checkpoints"
        )
        return to_keep


class CombinedRetentionPolicy(RetentionPolicy):
    """Combine multiple retention policies (union of checkpoints to keep)."""

    def __init__(self, policies: List[RetentionPolicy]):
        """
        Initialize combined policy.

        Args:
            policies: List of retention policies to combine.
        """
        self.policies = policies

    def select_checkpoints_to_keep(
        self, checkpoints: List[Dict[str, Any]]
    ) -> List[str]:
        """Keep union of checkpoints from all policies."""
        to_keep = set()
        for policy in self.policies:
            to_keep.update(policy.select_checkpoints_to_keep(checkpoints))
        logger.info(f"CombinedPolicy: Keeping {len(to_keep)} checkpoints total")
        return list(to_keep)


class CustomRetentionPolicy(RetentionPolicy):
    """Custom retention policy using a callback function."""

    def __init__(self, callback: Callable[[List[Dict[str, Any]]], List[str]]):
        """
        Initialize custom policy.

        Args:
            callback: Function that takes list of checkpoint metadata
                and returns list of checkpoint names to keep.
        """
        self.callback = callback

    def select_checkpoints_to_keep(
        self, checkpoints: List[Dict[str, Any]]
    ) -> List[str]:
        """Use custom callback to select checkpoints."""
        return self.callback(checkpoints)
