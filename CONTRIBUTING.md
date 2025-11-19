# Contributing to HuggingFace Lifecycle Manager

Thank you for your interest in contributing to HuggingFace Lifecycle Manager! 🎉

This document provides guidelines and instructions for contributing to the project. We appreciate your help in making this tool better for the entire ML community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards others

**Unacceptable behavior includes:**
- Harassment, trolling, or derogatory comments
- Publishing others' private information
- Any conduct which could reasonably be considered inappropriate

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A HuggingFace account (for testing Hub features)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/codewithdark-git/huggingface-lifecycle.git
   cd huggingface-lifecycle
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/codewithdark-git/huggingface-lifecycle.git
   ```

## 💻 Development Setup

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will run automated checks (linting, formatting) before each commit.

### 4. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check src/ tests/
isort --check-only src/ tests/
```

## 🎯 How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Minimal code example if possible

### Suggesting Features

1. **Check existing feature requests**
2. **Create a new issue** with:
   - Clear use case description
   - Why this feature would be useful
   - Proposed API/interface (if applicable)
   - Alternative solutions you've considered

### Contributing Code

1. **Find or create an issue** for what you're working on
2. **Comment on the issue** to let others know you're working on it
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Write your code** following our coding standards
5. **Add tests** for your changes
6. **Update documentation** if needed
7. **Run tests and linting**
8. **Commit your changes** with clear messages
9. **Push to your fork** and create a Pull Request

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Formatting**: Use Black for automatic formatting
- **Import sorting**: Use isort
- **Type hints**: Encouraged for all public APIs

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check with flake8
flake8 src/ tests/
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Dict, Any

def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save model checkpoint with metadata."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    Longer description with more details about what the
    function does and how to use it.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When validation fails.

    Example:
        >>> result = example_function("test", 42)
        >>> assert result is True
    """
    ...
```

### Error Handling

- Use custom exceptions from `hf_lifecycle.exceptions`
- Provide clear error messages
- Include context in exception messages

```python
from hf_lifecycle.exceptions import CheckpointError

def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise CheckpointNotFoundError(
            f"Checkpoint not found: {path}. "
            f"Available checkpoints: {list_available()}"
        )
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_<module>.py`
- Use pytest fixtures for common setup
- Aim for >90% code coverage

### Test Structure

```python
import pytest
from hf_lifecycle.checkpoint import CheckpointManager

class TestCheckpointManager:
    @pytest.fixture
    def checkpoint_manager(self, tmp_path):
        """Fixture for checkpoint manager."""
        return CheckpointManager(local_dir=str(tmp_path))
    
    def test_save_checkpoint(self, checkpoint_manager):
        """Test checkpoint saving."""
        # Arrange
        model = SimpleModel()
        
        # Act
        path = checkpoint_manager.save(model, epoch=1)
        
        # Assert
        assert os.path.exists(path)
        assert "checkpoint" in path
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_checkpoint.py -v

# Run with coverage
pytest tests/ --cov=hf_lifecycle --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "checkpoint" -v

# Run in parallel
pytest tests/ -n auto
```

### Mocking External Services

Use `unittest.mock` for Hub API calls:

```python
from unittest.mock import MagicMock, patch

def test_upload_to_hub(checkpoint_manager):
    with patch("hf_lifecycle.checkpoint.HfApi") as mock_api:
        mock_api.return_value.upload_file = MagicMock()
        checkpoint_manager.upload_to_hub("user/repo")
        assert mock_api.return_value.upload_file.called
```

## 📚 Documentation

### Code Documentation

- All public APIs must have docstrings
- Include usage examples in docstrings
- Document edge cases and limitations

### Sphinx Documentation

Documentation is in `docs/` using Sphinx with RST format:

```bash
# Build documentation
cd docs
sphinx-build -b html . build/html

# View documentation
open build/html/index.html
```

### Adding New Documentation

1. Create `.rst` file in `docs/`
2. Add to `docs/index.rst` table of contents
3. Build and verify locally
4. Include in your PR

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] All commits have clear messages

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Screenshots (if applicable)

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Follows code style guidelines
```

### Review Process

1. **Automated checks** must pass (tests, linting)
2. **At least one maintainer review** required
3. **All comments addressed** before merge
4. **No merge conflicts** with main branch

### After Approval

- Maintainer will merge using "Squash and Merge"
- Your changes will be in the next release
- Thank you for your contribution! 🎉

## 🚢 Release Process

(For maintainers)

### Version Numbering

We use Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Steps

1. Update version in `src/hf_lifecycle/__init__.py`
2. Update `CHANGELOG.md`
3. Create release commit:
   ```bash
   git commit -m "Release v1.2.3"
   ```
4. Tag the release:
   ```bash
   git tag -a v1.2.3 -m "Version 1.2.3"
   git push origin v1.2.3
   ```
5. GitHub Actions will automatically:
   - Run tests
   - Build package
   - Publish to PyPI
   - Create GitHub release

## 💡 Tips for Contributors

### Good First Issues

Look for issues labeled `good-first-issue` - these are great starting points!

### Communication

- Comment on issues before starting work
- Ask questions if anything is unclear
- Be patient - maintainers are volunteers
- Be open to feedback

### Stay Updated

```bash
# Sync your fork with upstream
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 🏆 Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Given credit in documentation

## 📞 Getting Help

- **Questions?** Open a [Discussion](https://github.com/OWNER/REPO/discussions)
- **Bugs?** Open an [Issue](https://github.com/OWNER/REPO/issues)
- **Chat?** Join our community (link TBD)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HuggingFace Lifecycle Manager! Your efforts help make ML development better for everyone. 🚀

**Happy Coding!** 💻✨
