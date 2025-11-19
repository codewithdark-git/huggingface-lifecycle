# API Reference

## Authentication

::: hf_lifecycle.auth.AuthManager
    handler: python
    options:
      members:
        - login
        - logout
        - get_token
        - validate_token
        - set_profile

## Repository Management

::: hf_lifecycle.repo.RepoManager
    handler: python
    options:
      members:
        - create_repo
        - delete_repo
        - list_repos
        - update_card
        - create_branch
        - file_exists

## Exceptions

::: hf_lifecycle.exceptions
