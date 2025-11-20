HuggingFace Lifecycle Manager Documentation
============================================

**huggingface-lifecycle** (import as ``hf_lifecycle``) is a production-ready Python package that provides comprehensive lifecycle management for HuggingFace training workflows.

Features
--------

- **Unified Authentication**: Manage tokens securely across environments
- **Repository Management**: Create, update, and manage HuggingFace Hub repositories
- **Checkpoint Operations**: Intelligent saving, loading, and retention policies
- **Model Registry**: Register custom models and configurations
- **Dataset Management**: Upload, version, and manage datasets
- **Training State Persistence**: Save and restore complete training states
- **Utilities**: Progress tracking, logging, and error handling

Installation
------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install huggingface-lifecycle

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/codewithdark-git/huggingface-lifecycle.git
   cd huggingface-lifecycle
   pip install -e .

Quick Start
-----------

Authentication
~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.auth import AuthManager

   # Initialize (automatically checks HF_TOKEN env var and CLI cache)
   auth = AuthManager()

   # Explicit login
   auth.login(token="hf_...", write_to_disk=True)

Repository Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.repo import RepoManager

   repo_mgr = RepoManager(auth)

   # Create a new private model repository
   url = repo_mgr.create_repo("username/my-new-model", private=True)

   # Update the Model Card
   repo_mgr.update_card("username/my-new-model", "# My Model\\n\\nDescription here.")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   authentication
   repository
   checkpoint
   model_registry
   dataset
   training_state
   utilities
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/manager
   api/auth
   api/repo
   api/checkpoint
   api/model_registry
   api/dataset
   api/training_state
   api/utils
   api/exceptions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
