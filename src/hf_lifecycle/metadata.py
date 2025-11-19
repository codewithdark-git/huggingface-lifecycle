"""
Metadata tracking for HuggingFace Lifecycle Manager.
"""
import os
import sys
import platform
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class MetadataTracker:
    """
    Tracks and manages metadata for training experiments.
    """

    def __init__(self):
        """Initialize the MetadataTracker."""
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "system": {},
            "environment": {},
            "hyperparameters": {},
            "metrics": {},
            "custom": {},
        }

    def capture_system_info(self) -> Dict[str, Any]:
        """
        Capture system information.

        Returns:
            Dictionary containing system information.
        """
        import torch
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
        }

        # GPU information
        if torch.cuda.is_available():
            system_info["cuda_available"] = True
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            system_info["gpu_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            system_info["cuda_available"] = False

        self.metadata["system"] = system_info
        logger.info("Captured system information")
        return system_info

    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture environment information including package versions and git info.

        Returns:
            Dictionary containing environment information.
        """
        import torch
        import pkg_resources

        env_info = {
            "packages": {},
            "git": {},
        }

        # Key packages
        key_packages = [
            "torch",
            "transformers",
            "huggingface-hub",
            "numpy",
            "pandas",
        ]

        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                env_info["packages"][package] = version
            except Exception:
                env_info["packages"][package] = "not installed"

        # Git information
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info["git"]["commit"] = git_commit

            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env_info["git"]["branch"] = git_branch

            # Check if there are uncommitted changes
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info["git"]["dirty"] = len(git_status) > 0

        except Exception as e:
            env_info["git"]["error"] = str(e)
            logger.warning(f"Could not capture git information: {e}")

        self.metadata["environment"] = env_info
        logger.info("Captured environment information")
        return env_info

    def track_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Track training hyperparameters.

        Args:
            params: Dictionary of hyperparameters.
        """
        self.metadata["hyperparameters"].update(params)
        logger.info(f"Tracked {len(params)} hyperparameters")

    def track_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Track training/validation metrics.

        Args:
            metrics: Dictionary of metrics.
            step: Optional step number.
        """
        if step is not None:
            if "history" not in self.metadata["metrics"]:
                self.metadata["metrics"]["history"] = []
            
            self.metadata["metrics"]["history"].append({
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            })
        else:
            self.metadata["metrics"]["final"] = metrics

        logger.info(f"Tracked metrics: {list(metrics.keys())}")

    def add_custom(self, key: str, value: Any) -> None:
        """
        Add custom metadata.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata["custom"][key] = value

    def save_metadata(self, path: str) -> None:
        """
        Save metadata to JSON file.

        Args:
            path: Path to save metadata.
        """
        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {path}")

    def load_metadata(self, path: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file.

        Args:
            path: Path to load metadata from.

        Returns:
            Loaded metadata dictionary.
        """
        with open(path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata from {path}")
        return self.metadata

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the metadata.

        Returns:
            Formatted summary string.
        """
        summary = []
        summary.append("=" * 50)
        summary.append("EXPERIMENT METADATA SUMMARY")
        summary.append("=" * 50)
        
        # System
        if self.metadata.get("system"):
            summary.append("\nSYSTEM:")
            summary.append(f"  Platform: {self.metadata['system'].get('platform', 'N/A')}")
            summary.append(f"  Python: {self.metadata['system'].get('python_version', 'N/A').split()[0]}")
            if self.metadata['system'].get('cuda_available'):
                summary.append(f"  GPUs: {self.metadata['system'].get('gpu_count', 0)}")

        # Environment
        if self.metadata.get("environment", {}).get("packages"):
            summary.append("\nKEY PACKAGES:")
            for pkg, ver in self.metadata["environment"]["packages"].items():
                summary.append(f"  {pkg}: {ver}")

        # Git
        if self.metadata.get("environment", {}).get("git", {}).get("commit"):
            summary.append("\nGIT:")
            summary.append(f"  Commit: {self.metadata['environment']['git']['commit'][:8]}")
            summary.append(f"  Branch: {self.metadata['environment']['git'].get('branch', 'N/A')}")
            if self.metadata['environment']['git'].get('dirty'):
                summary.append("  Status: DIRTY (uncommitted changes)")

        # Hyperparameters
        if self.metadata.get("hyperparameters"):
            summary.append("\nHYPERPARAMETERS:")
            for key, value in self.metadata["hyperparameters"].items():
                summary.append(f"  {key}: {value}")

        # Final metrics
        if self.metadata.get("metrics", {}).get("final"):
            summary.append("\nFINAL METRICS:")
            for key, value in self.metadata["metrics"]["final"].items():
                summary.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        summary.append("=" * 50)
        return "\n".join(summary)

    def to_model_card_section(self) -> str:
        """
        Generate a model card section from metadata.

        Returns:
            Markdown formatted model card section.
        """
        card = "## Training Details\n\n"
        
        # System
        if self.metadata.get("system"):
            card += "### System Information\n\n"
            card += f"- Platform: {self.metadata['system'].get('platform', 'N/A')}\n"
            card += f"- Python: {self.metadata['system'].get('python_version', 'N/A').split()[0]}\n"
            if self.metadata['system'].get('cuda_available'):
                card += f"- GPUs: {self.metadata['system'].get('gpu_count', 0)}\n"
            card += "\n"

        # Hyperparameters
        if self.metadata.get("hyperparameters"):
            card += "### Hyperparameters\n\n"
            card += "| Parameter | Value |\n"
            card += "|-----------|-------|\n"
            for key, value in self.metadata["hyperparameters"].items():
                card += f"| {key} | {value} |\n"
            card += "\n"

        # Metrics
        if self.metadata.get("metrics", {}).get("final"):
            card += "### Results\n\n"
            card += "| Metric | Value |\n"
            card += "|--------|-------|\n"
            for key, value in self.metadata["metrics"]["final"].items():
                if isinstance(value, float):
                    card += f"| {key} | {value:.4f} |\n"
                else:
                    card += f"| {key} | {value} |\n"
            card += "\n"

        return card
