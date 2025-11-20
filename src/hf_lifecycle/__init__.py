"""
HuggingFace Lifecycle Manager
"""

__version__ = "0.1.0"

# High-level API
from hf_lifecycle.manager import HFManager

# Low-level components (for advanced users)
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.metadata import MetadataTracker
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.model_registry import ModelRegistry
from hf_lifecycle.retention import RetentionPolicy, KeepLastN, KeepBestM

__all__ = [
    "HFManager",
    "RepoManager",
    "MetadataTracker",
    "CheckpointManager",
    "ModelRegistry",
    "RetentionPolicy",
    "KeepLastN",
    "KeepBestM",
]
