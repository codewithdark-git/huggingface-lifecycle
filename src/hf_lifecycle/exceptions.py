"""
Custom exceptions for HuggingFace Lifecycle Manager.
"""

class HfLifecycleError(Exception):
    """Base exception for all hf_lifecycle errors."""
    pass

class AuthenticationError(HfLifecycleError):
    """Base exception for authentication failures."""
    pass

class TokenNotFoundError(AuthenticationError):
    """Raised when no authentication token is found."""
    pass

class InvalidTokenError(AuthenticationError):
    """Raised when the authentication token is invalid."""
    pass

class RepositoryError(HfLifecycleError):
    """Base exception for repository operations."""
    pass

class CheckpointError(HfLifecycleError):
    """Base exception for checkpoint operations."""
    pass

class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint is not found."""
    pass

class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint is corrupted or invalid."""
    pass
