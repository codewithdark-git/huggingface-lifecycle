"""
Authentication module for HuggingFace Lifecycle Manager.
"""
import os
from typing import Optional
import logging

from huggingface_hub import HfApi, login, logout, get_token
from huggingface_hub.utils import LocalTokenNotFoundError
from dotenv import load_dotenv

from hf_lifecycle.exceptions import (
    AuthenticationError,
    TokenNotFoundError,
    InvalidTokenError,
)

# Configure logging
logger = logging.getLogger(__name__)

class AuthManager:
    """
    Manages authentication with the HuggingFace Hub.

    Supports multiple token sources:
    1. Environment variable (HF_TOKEN)
    2. Direct input
    3. Cached credentials (from `huggingface-cli login`)
    """

    def __init__(self, token: Optional[str] = None, profile: Optional[str] = None):
        """
        Initialize the AuthManager.

        Args:
            token: Optional HuggingFace token.
            profile: Optional profile name (not yet fully implemented).
        """
        self._token = token
        self._profile = profile
        load_dotenv()  # Load environment variables from .env file

    def login(self, token: Optional[str] = None, write_to_disk: bool = False) -> None:
        """
        Authenticate with the HuggingFace Hub.

        Args:
            token: The HuggingFace token. If None, tries to find it in env or cache.
            write_to_disk: Whether to save the token to the local cache.

        Raises:
            TokenNotFoundError: If no token is provided or found.
            InvalidTokenError: If the token is invalid.
        """
        token_to_use = token or self._token or os.getenv("HF_TOKEN")

        if not token_to_use:
            # Try to get from cache
            try:
                token_to_use = get_token()
            except LocalTokenNotFoundError:
                pass

        if not token_to_use:
            raise TokenNotFoundError(
                "No HuggingFace token found. Please provide a token, set HF_TOKEN env var, "
                "or login via CLI."
            )

        # Validate token
        try:
            self.validate_token(token_to_use)
        except InvalidTokenError as e:
            raise e
        except Exception as e:
            raise AuthenticationError(f"Unexpected error during validation: {e}")

        # If we got here, token is valid
        self._token = token_to_use
        
        if write_to_disk:
            login(token=self._token)
            logger.info("Token saved to local cache.")
        
        logger.info("Successfully authenticated with HuggingFace Hub.")

    def logout(self) -> None:
        """
        Logout and clear local credentials.
        """
        logout()
        self._token = None
        logger.info("Logged out from HuggingFace Hub.")

    def get_token(self) -> Optional[str]:
        """
        Retrieve the current active token.

        Returns:
            The active token or None if not authenticated.
        """
        if self._token:
            return self._token
        
        token = os.getenv("HF_TOKEN")
        if token:
            return token
            
        try:
            return get_token()
        except LocalTokenNotFoundError:
            return None

    def validate_token(self, token: str) -> None:
        """
        Validate a HuggingFace token.

        Args:
            token: The token to validate.

        Raises:
            InvalidTokenError: If the token is invalid.
        """
        api = HfApi(token=token)
        try:
            user = api.whoami()
            logger.debug(f"Token valid for user: {user.get('name')}")
        except Exception as e:
            # HfApi raises generic exceptions for auth failures often, 
            # but usually HTTPError 401. We catch broad exception to be safe 
            # and re-raise as our custom error.
            logger.error(f"Token validation failed: {e}")
            raise InvalidTokenError(f"Invalid HuggingFace token: {e}")

    def set_profile(self, name: str) -> None:
        """
        Switch the active profile.
        
        Note: This is a placeholder for future multi-account support.
        
        Args:
            name: The name of the profile to switch to.
        """
        self._profile = name
        logger.info(f"Switched to profile: {name}")
