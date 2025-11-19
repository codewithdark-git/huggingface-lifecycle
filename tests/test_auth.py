import pytest
import os
from unittest.mock import patch, MagicMock
from hf_lifecycle.auth import AuthManager
from hf_lifecycle.exceptions import TokenNotFoundError, InvalidTokenError

class TestAuthManager:
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()

    def test_init(self):
        am = AuthManager(token="test_token")
        assert am._token == "test_token"

    @patch("hf_lifecycle.auth.get_token")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_token_none(self, mock_get_token, auth_manager):
        mock_get_token.return_value = None
        assert auth_manager.get_token() is None

    @patch.dict(os.environ, {"HF_TOKEN": "env_token"}, clear=True)
    def test_get_token_env(self, auth_manager):
        assert auth_manager.get_token() == "env_token"

    @patch("hf_lifecycle.auth.get_token")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_token_cache(self, mock_get_token, auth_manager):
        mock_get_token.return_value = "cache_token"
        assert auth_manager.get_token() == "cache_token"

    @patch("hf_lifecycle.auth.HfApi")
    def test_validate_token_valid(self, mock_hf_api, auth_manager):
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.whoami.return_value = {"name": "test_user"}
        
        # Should not raise
        auth_manager.validate_token("valid_token")

    @patch("hf_lifecycle.auth.HfApi")
    def test_validate_token_invalid(self, mock_hf_api, auth_manager):
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.whoami.side_effect = Exception("401 Unauthorized")
        
        with pytest.raises(InvalidTokenError):
            auth_manager.validate_token("invalid_token")

    @patch("hf_lifecycle.auth.login")
    @patch("hf_lifecycle.auth.AuthManager.validate_token")
    def test_login_success(self, mock_validate, mock_login, auth_manager):
        auth_manager.login(token="new_token", write_to_disk=True)
        mock_validate.assert_called_with("new_token")
        mock_login.assert_called_with(token="new_token")
        assert auth_manager._token == "new_token"

    @patch("hf_lifecycle.auth.get_token")
    @patch.dict(os.environ, {}, clear=True)
    def test_login_no_token_found(self, mock_get_token, auth_manager):
        mock_get_token.return_value = None
        with pytest.raises(TokenNotFoundError):
            auth_manager.login()

    @patch("hf_lifecycle.auth.logout")
    def test_logout(self, mock_logout, auth_manager):
        auth_manager._token = "some_token"
        auth_manager.logout()
        assert auth_manager._token is None
        mock_logout.assert_called_once()
