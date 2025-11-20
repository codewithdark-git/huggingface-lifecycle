# 🚀 HuggingFace Lifecycle Manager

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://huggingface-lifecycle.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Streamline your HuggingFace training workflows with automated lifecycle management and complete reproducibility.**

A production-ready Python package for comprehensive lifecycle management of HuggingFace training workflows. Streamline your ML experiments with automated checkpoint management, model registration, and complete reproducibility tracking.

📖 **[Read the Full Documentation →](https://huggingface-lifecycle.readthedocs.io)**

## ✨ Features

- **Unified Manager** - Single `HFManager` class for all operations
- **Checkpoint Operations** - Intelligent saving, loading, and retention policies
- **Model Registry** - Register custom models and configurations
- **Metadata Tracking** - Auto-capture system info, hyperparameters, and metrics
- **Hub Integration** - Push checkpoints, metadata, and models to HuggingFace Hub
- **Flexible Pushing** - Auto-push after each save or batch push at the end

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

The `HFManager` class provides a simple, unified interface for all lifecycle operations:

```python
from hf_lifecycle import HFManager
import torch

# Initialize manager
manager = HFManager(
    repo_id="username/my-model",
    local_dir="./outputs",
    checkpoint_dir="./checkpoints",
    hf_token="your_token",
    auto_push=False  # Set to True to auto-push checkpoints
)

# Track hyperparameters
manager.track_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})

# During training loop
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = evaluate(...)
    
    # Log metrics
    manager.log_metrics({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }, step=epoch)
    
    # Save checkpoint (auto-pushed if auto_push=True)
    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"val_loss": val_loss},
        config=config,
        push=False  # Override auto_push for this checkpoint
    )

# Save final model
manager.save_final_model(
    model=model,
    format="safetensors",  # or "pt"
    config=config
)

# Push everything to Hub at once
manager.push(
    push_checkpoints=True,
    push_metadata=True,
    push_final_model=True
)

# Cleanup old checkpoints
manager.cleanup_checkpoints()
```

## 📚 Documentation

For detailed documentation on all features and advanced usage, visit our [full documentation](https://huggingface-lifecycle.readthedocs.io).

### Key Topics

- **[Quick Start Guide](docs/quickstart.rst)** - Get started with HFManager
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Examples](examples/)** - Working code examples

## 🎯 Features Overview

### Metadata Tracking
Automatically capture and track:
- System information (OS, Python version, GPU info)
- Environment details (package versions, Git info)
- Hyperparameters
- Training metrics

### Checkpoint Management
- Smart checkpoint saving with retention policies
- Load latest or best checkpoints
- Push to HuggingFace Hub
- Automatic cleanup of old checkpoints

### Model Registry
- Register custom model architectures
- Auto-generate model cards
- Push models to HuggingFace Hub

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace team for the amazing Hub and Transformers library
- PyTorch team for the deep learning framework

## 📧 Contact

For questions and support:
- GitHub Issues: [github.com/codewithdark-git/huggingface-lifecycle/issues](https://github.com/codewithdark-git/huggingface-lifecycle/issues)
- Documentation: [huggingface-lifecycle.readthedocs.io](https://huggingface-lifecycle.readthedocs.io)

---

<p align="center">Made with ❤️ for the ML community</p>
