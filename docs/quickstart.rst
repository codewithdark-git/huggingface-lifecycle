Quick Start Guide
=================

This guide will help you get started with HuggingFace Lifecycle Manager.

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install huggingface-lifecycle

Basic Usage
-----------

1. Authentication
~~~~~~~~~~~~~~~~~

First, authenticate with the HuggingFace Hub:

.. code-block:: python

   from hf_lifecycle.auth import AuthManager

   auth = AuthManager()
   auth.login(token="hf_your_token_here", write_to_disk=True)

You can also set the ``HF_TOKEN`` environment variable:

.. code-block:: bash

   export HF_TOKEN=hf_your_token_here

2. Repository Management
~~~~~~~~~~~~~~~~~~~~~~~~~

Create and manage repositories:

.. code-block:: python

   from hf_lifecycle.repo import RepoManager

   repo_mgr = RepoManager(auth)

   # Create a new repository
   url = repo_mgr.create_repo("username/my-model", private=True)
   print(f"Created: {url}")

   # List your repositories
   repos = repo_mgr.list_repos()
   print(f"My repositories: {repos}")

3. Update Model Cards
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model_card = """
   # My Awesome Model

   This model does amazing things!

   ## Training Details
   - Dataset: custom-dataset
   - Epochs: 10
   """

   repo_mgr.update_card("username/my-model", model_card)

Next Steps
----------

- Learn more about :doc:`authentication`
- Explore :doc:`repository` management
- Check out the :doc:`api/auth` for detailed API documentation
