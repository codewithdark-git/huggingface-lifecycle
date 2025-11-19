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
