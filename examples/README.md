# Examples

This directory contains example scripts demonstrating how to use HuggingFace Lifecycle Manager.

## Quick Start

**[quickstart.py](quickstart.py)** - Minimal example showing basic usage in a few lines.

```bash
python examples/quickstart.py
```

## Complete Training Example

**[complete_training_example.py](complete_training_example.py)** - Comprehensive example demonstrating:

- Authentication
- Checkpoint management with retention policies
- Training state management with early stopping
- Metadata tracking
- Model registration

```bash
python examples/complete_training_example.py
```

This example includes:
- ✅ Full training loop with validation
- ✅ Automatic checkpoint saving
- ✅ Retention policy (keep last 3 + best 2)
- ✅ Early stopping
- ✅ Metadata capture (system, environment, hyperparameters, metrics)
- ✅ Training state persistence for resumption

## Running the Examples

1. Install the package:
```bash
pip install -e .
```

2. (Optional) Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

3. Run any example:
```bash
python examples/complete_training_example.py
```

## What You'll Learn

- How to set up authentication
- How to manage checkpoints with intelligent retention
- How to track experiments with metadata
- How to implement early stopping
- How to save and resume training state
- How to register models to HuggingFace Hub
