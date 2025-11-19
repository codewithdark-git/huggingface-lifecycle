# 🚀 HuggingFace Lifecycle Manager

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Python package for comprehensive lifecycle management of HuggingFace training workflows. Streamline your ML experiments with automated checkpoint management, model registration, dataset handling, and complete reproducibility tracking.

## ✨ Key Features

- **🔐 Authentication** - Unified HuggingFace Hub authentication with token management
- **📦 Repository Management** - Create, manage repos with automated model cards
- **💾 Smart Checkpointing** - Intelligent retention policies (keep last N, best M)
- **🤖 Model Registry** - Register PyTorch models with auto-generated cards
- **📊 Dataset Management** - Upload Pandas DataFrames directly (Parquet/CSV/JSON)
- **🔄 Training State** - Complete state persistence with RNG reproducibility
- **📈 Metadata Tracking** - Auto-capture system info, hyperparameters, metrics
- **💻 CLI** - Full-featured command-line interface
- **🛠️ Utilities** - Checksums, disk checks, retry logic, timers, logging

##  Installation

```bash
# From source
git clone https://github.com/codewithdark-git/huggingface-lifecycle.git
cd huggingface-lifecycle
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
from hf_lifecycle.auth import AuthManager
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.retention import KeepLastN, KeepBestM, CombinedRetentionPolicy
from hf_lifecycle.repo import RepoManager

# Setup
auth = AuthManager()
repo_mgr = RepoManager(auth)

# Smart checkpoint retention
retention = CombinedRetentionPolicy([
    KeepLastN(3),                    # Keep last 3 checkpoints
    KeepBestM(2, "val_loss", "min")  # Keep best 2 by validation loss
])
ckpt_mgr = CheckpointManager(repo_mgr, retention_policy=retention)

# Training loop
for epoch in range(100):
    # Your training code...
    ckpt_mgr.save(model, optimizer, epoch=epoch, metrics={"val_loss": 0.25})
```

### CLI Usage

```bash
# Authenticate
hf-lifecycle auth login

# Manage repositories
hf-lifecycle repo create username/my-model --type model

# List and cleanup checkpoints
hf-lifecycle checkpoint list
hf-lifecycle checkpoint cleanup --dry-run

# Dataset operations
hf-lifecycle dataset create-dataset username/my-dataset
hf-lifecycle dataset upload username/my-dataset data.csv data.csv

# Metadata tracking
hf-lifecycle metadata capture -o metadata.json
```

## 📚 Documentation

Full documentation available at [ReadTheDocs](https://huggingface-lifecycle.readthedocs.io):

- **[QuickStart Guide](docs/quickstart.rst)** - Get started in 5 minutes
- **[Authentication](docs/authentication.rst)** - Token management
- **[Checkpoint Management](docs/checkpoint.rst)** - Smart checkpointing
- **[Model Registry](docs/model_registry.rst)** - Register models
- **[Dataset Management](docs/dataset.rst)** - Handle datasets
- **[Training State](docs/training_state.rst)** - Reproducible training
- **[API Reference](docs/api/)** - Complete API documentation

Build locally:
```bash
cd docs && sphinx-build -b html . build/html
```

## 🎯 Core Capabilities

### Intelligent Checkpoint Management
```python
# Combined retention: keep recent + best performers
policy = CombinedRetentionPolicy([
    KeepLastN(5),
    KeepBestM(3, "f1_score", "max")
])
```

### Complete Reproducibility
```python
from hf_lifecycle.training_state import TrainingStateManager

state_mgr = TrainingStateManager()
state_mgr.save_state("state.pt", epoch=10, step=1000)
# Saves model, optimizer, scheduler + all RNG states
```

### One-Line Model Publishing
```python
from hf_lifecycle.model_registry import ModelRegistry

registry = ModelRegistry(repo_mgr)
registry.register_model(
    model=model,
    repo_id="username/my-model",
    metrics={"accuracy": 0.95}
)
```

### DataFrame to Dataset
```python
from hf_lifecycle.dataset import DatasetManager

dataset_mgr = DatasetManager(repo_mgr)
dataset_mgr.upload_dataframe(
    "username/dataset",
    df,
    "train.parquet",
    format="parquet"
)
```

## 🏗️ Architecture

```
hf_lifecycle/
├── auth.py              # Authentication
├── repo.py              # Repository ops  
├── checkpoint.py        # Checkpoint management
├── retention.py         # Retention policies
├── model_registry.py    # Model registration
├── dataset.py           # Dataset management
├── training_state.py    # State & early stopping
├── metadata.py          # Metadata tracking
├── cli.py               # CLI interface
├── utils.py             # Utilities
├── logger.py            # Logging
└── exceptions.py        # Custom exceptions
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=hf_lifecycle --cov-report=html
```

**Coverage:** 96% with 96+ passing tests

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick steps:
1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run `pytest`, `black`, `isort`
5. Submit Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built on [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)
- Inspired by ML community best practices

## 📮 Support

- **Issues:** [GitHub Issues](https://github.com/codewithdark-git/huggingface-lifecycle/issues)
- **Docs:** [ReadTheDocs](https://huggingface-lifecycle.readthedocs.io)

---

<p align="center">Made with ❤️ for the ML community</p>
