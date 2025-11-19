"""
Quick Start Example - HuggingFace Lifecycle Manager

This example shows the basic workflow in just a few lines.
"""

from hf_lifecycle.auth import AuthManager
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.metadata import MetadataTracker

# 1. Authenticate
auth = AuthManager()
# auth.login(token="your_token_here")

# 2. Setup managers
repo_mgr = RepoManager(auth)
ckpt_mgr = CheckpointManager(repo_mgr)

# 3. Track metadata
tracker = MetadataTracker()
tracker.capture_system_info()
tracker.track_hyperparameters({"lr": 0.001, "batch_size": 32})

# 4. During training...
# ckpt_mgr.save(model, optimizer, epoch=10, metrics={"loss": 0.5})

# 5. Load best checkpoint
# checkpoint = ckpt_mgr.load_best("loss", mode="min", model=model)

print("✓ Quick start example ready!")
print("See complete_training_example.py for a full workflow.")
