# HuggingFace Lifecycle Manager

**huggingface-lifecycle** (import as `hf_lifecycle`) is a production-ready Python package that provides comprehensive lifecycle management for HuggingFace training workflows. This package eliminates repetitive checkpoint management code and provides a unified interface for authentication, repository management, checkpoint operations, model registration, dataset management, and training state persistence.

## Features

- **Unified Authentication**: Manage tokens securely across environments.
- **Repository Management**: Create, update, and manage HuggingFace Hub repositories.
- **Checkpoint Operations**: Intelligent saving, loading, and retention policies.
- **Model Registry**: Register custom models and configurations.
- **Dataset Management**: Upload, version, and manage datasets.
- **Training State Persistence**: Save and restore complete training states.
- **Utilities**: Progress tracking, logging, and error handling.

## Installation

### From PyPI

```bash
pip install huggingface-lifecycle
```

### From Source

```bash
git clone https://github.com/user/huggingface-lifecycle.git
cd huggingface-lifecycle
pip install -e .
```

## Usage

### Authentication

The `AuthManager` handles authentication with the HuggingFace Hub. It supports tokens from environment variables (`HF_TOKEN`), configuration files, or the CLI cache.

```python
from hf_lifecycle.auth import AuthManager

# Initialize (automatically checks HF_TOKEN env var and CLI cache)
auth = AuthManager()

# Explicit login (optional, useful for scripts)
try:
    auth.login(token="hf_...", write_to_disk=True)
    print("Successfully logged in!")
except Exception as e:
    print(f"Login failed: {e}")

# Get the active token
token = auth.get_token()
```

### Repository Management

The `RepoManager` simplifies creating, deleting, and managing repositories.

```python
from hf_lifecycle.repo import RepoManager

repo_mgr = RepoManager(auth)

# Create a new private model repository
url = repo_mgr.create_repo("username/my-new-model", private=True)
print(f"Created repo: {url}")

# Update the Model Card (README.md)
repo_mgr.update_card("username/my-new-model", "# My New Model\n\nDescription here.")

# List your repositories
repos = repo_mgr.list_repos()
print(f"My repos: {repos}")
```

```python
import hf_lifecycle

# Example usage will be added here
print(hf_lifecycle.__version__)
```

## Development

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv .venv`
3.  Activate the environment: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
4.  Install development dependencies: `pip install -r requirements-dev.txt`
5.  Install the package in editable mode: `pip install -e .`
6.  Run tests: `pytest`

## License

MIT License
