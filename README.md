# 🚀 HuggingFace Lifecycle Manager

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://huggingface-lifecycle.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Python package for comprehensive lifecycle management of HuggingFace training workflows. Streamline your ML experiments with automated checkpoint management, model registration, dataset handling, and complete reproducibility tracking.

📖 **[Read the Full Documentation →](https://huggingface-lifecycle.readthedocs.io)**

## ✨ Features

- **Unified Authentication** - Manage tokens securely across environments
- **Repository Management** - Create, update, and manage HuggingFace Hub repositories
- **Checkpoint Operations** - Intelligent saving, loading, and retention policies
- **Model Registry** - Register custom models and configurations
- **Dataset Management** - Upload, version, and manage datasets
- **Training State Persistence** - Save and restore complete training states
- **Metadata Tracking** - Auto-capture system info, hyperparameters, and metrics
- **Command-Line Interface** - Full-featured CLI for all operations
- **Utilities** - Progress tracking, logging, and error handling

## 📥 Installation

### From PyPI

```bash
pip install huggingface-lifecycle
```

### From Source

```bash
git clone https://github.com/codewithdark-git/huggingface-lifecycle.git
cd huggingface-lifecycle
pip install -e .
```

## 🚀 Quick Start

### Authentication

The `AuthManager` handles authentication with the HuggingFace Hub:

```python
from hf_lifecycle.auth import AuthManager

# Initialize (checks HF_TOKEN env var and CLI cache)
auth = AuthManager()

# Explicit login
auth.login(token="hf_...", write_to_disk=True)

# Get active token
token = auth.get_token()
```

### Repository Management

Create and manage repositories on HuggingFace Hub:

```python
from hf_lifecycle.repo import RepoManager

repo_mgr = RepoManager(auth)

# Create a private model repository
url = repo_mgr.create_repo("username/my-model", private=True)

# Update Model Card
repo_mgr.update_card("username/my-model", "# My Model\n\nDescription...")

# List your repositories
repos = repo_mgr.list_repos()
```

### Checkpoint Management

Smart checkpoint saving with retention policies:

```python
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.retention import KeepLastN, KeepBestM, CombinedRetentionPolicy

# Configure retention policy
retention = CombinedRetentionPolicy([
    KeepLastN(3),                    # Keep last 3 checkpoints
    KeepBestM(2, "val_loss", "min")  # Keep best 2 by validation loss
])

ckpt_mgr = CheckpointManager(repo_mgr, retention_policy=retention)

# Save checkpoint
ckpt_mgr.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"val_loss": 0.25, "accuracy": 0.95}
)

# Load best checkpoint
best = ckpt_mgr.load_best("val_loss", mode="min", model=model)

# Cleanup old checkpoints
ckpt_mgr.cleanup()
```

### Model Registry

Register models to HuggingFace Hub with auto-generated cards:

```python
from hf_lifecycle.model_registry import ModelRegistry

registry = ModelRegistry(repo_mgr)

# Register standard model
registry.register_model(
    model=model,
    repo_id="username/my-awesome-model",
    description="Fine-tuned BERT for sentiment analysis",
    metrics={"accuracy": 0.95, "f1": 0.94},
    datasets=["imdb", "sst2"],
    tags=["sentiment-analysis", "pytorch"]
)

# Register custom model architecture
# Ensure your model inherits from PreTrainedModel and config from PretrainedConfig
registry.register_custom_model(
    model=custom_model,
    config=custom_config,
    repo_id="username/my-custom-model",
    model_type="my-custom-model-type",
    push_to_hub=True
)
```

### Dataset Management

Upload datasets and DataFrames directly:

```python
from hf_lifecycle.dataset import DatasetManager
import pandas as pd

dataset_mgr = DatasetManager(repo_mgr)

# Upload DataFrame as Parquet
df = pd.DataFrame({"text": [...], "label": [...]})
dataset_mgr.upload_dataframe(
    "username/my-dataset",
    df,
    "train.parquet",
    format="parquet"
)

# Upload files
dataset_mgr.upload_file("username/my-dataset", "data.csv", "data.csv")
```

### Training State Persistence

Complete training state management with RNG reproducibility:

```python
from hf_lifecycle.training_state import TrainingStateManager, EarlyStopping

# Save complete training state
state_mgr = TrainingStateManager()
state_mgr.save_state(
    "training_state.pt",
    epoch=10,
    step=1000,
    best_metric=0.95
)

# Load and restore state (including RNG)
state = state_mgr.load_state("training_state.pt", restore_rng=True)

# Early stopping
early_stop = EarlyStopping(patience=5, mode="min")
if early_stop.step(val_loss):
    print("Early stopping triggered!")
```

### Metadata Tracking

Capture comprehensive experiment metadata:

```python
from hf_lifecycle.metadata import MetadataTracker

tracker = MetadataTracker()

# Capture system and environment info
tracker.capture_system_info()
tracker.capture_environment()

# Track hyperparameters and metrics
tracker.track_hyperparameters({"lr": 0.001, "batch_size": 32})
tracker.track_metrics({"loss": 0.25, "accuracy": 0.95}, step=100)

# Save metadata
tracker.save_metadata("experiment_metadata.json")

# Generate model card section
card_section = tracker.to_model_card_section()
```

### Command-Line Interface

Full CLI for all operations:

```bash
# Authentication
hf-lifecycle auth login
hf-lifecycle auth logout

# Repository operations
hf-lifecycle repo create username/my-model --type model --public
hf-lifecycle repo list
hf-lifecycle repo delete username/my-model

# Checkpoint operations
hf-lifecycle checkpoint list --local-dir ./checkpoints
hf-lifecycle checkpoint cleanup --dry-run

# Dataset operations
hf-lifecycle dataset create-dataset username/my-dataset
hf-lifecycle dataset upload username/my-dataset data.csv data.csv

# Metadata operations
hf-lifecycle metadata capture -o metadata.json
hf-lifecycle metadata show metadata.json
```

## 📚 Documentation

Comprehensive documentation is available at **[ReadTheDocs](https://huggingface-lifecycle.readthedocs.io)**:

### User Guides
- **[Quick Start Guide](https://huggingface-lifecycle.readthedocs.io/en/latest/quickstart.html)** - Get started in 5 minutes
- **[Authentication](https://huggingface-lifecycle.readthedocs.io/en/latest/authentication.html)** - Token management and profiles
- **[Repository Management](https://huggingface-lifecycle.readthedocs.io/en/latest/repository.html)** - Create and manage repos
- **[Checkpoint Management](https://huggingface-lifecycle.readthedocs.io/en/latest/checkpoint.html)** - Smart checkpointing and retention
- **[Model Registry](https://huggingface-lifecycle.readthedocs.io/en/latest/model_registry.html)** - Register and publish models
- **[Dataset Management](https://huggingface-lifecycle.readthedocs.io/en/latest/dataset.html)** - Upload and manage datasets
- **[Training State](https://huggingface-lifecycle.readthedocs.io/en/latest/training_state.html)** - State persistence and early stopping
- **[Utilities](https://huggingface-lifecycle.readthedocs.io/en/latest/utilities.html)** - Helper functions and utilities

### API Reference
- **[Complete API Documentation](https://huggingface-lifecycle.readthedocs.io/en/latest/api/index.html)**


## 🤝 Contributing

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see the **[LICENSE](LICENSE)** file for details.

## 🙏 Acknowledgments

- Built on top of [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)
- Inspired by best practices from the ML community
- Thanks to all contributors!

## 📮 Support

- **Documentation:** [https://huggingface-lifecycle.readthedocs.io](https://huggingface-lifecycle.readthedocs.io)
- **Issues:** [GitHub Issues](https://github.com/codewithdark-git/huggingface-lifecycle/issues)
- **Discussions:** [GitHub Discussions](https://github.com/codewithdark-git/huggingface-lifecycle/discussions)

---

<p align="center">Made with ❤️ for the ML community</p>
