Repository Management
=====================

The ``RepoManager`` class provides a high-level interface for managing HuggingFace Hub repositories.

Creating Repositories
---------------------

Create Model Repository
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.auth import AuthManager
   from hf_lifecycle.repo import RepoManager

   auth = AuthManager()
   repo_mgr = RepoManager(auth)

   # Create a private model repository
   url = repo_mgr.create_repo(
       repo_id="username/my-model",
       repo_type="model",
       private=True,
       exist_ok=False
   )
   print(f"Created: {url}")

Create Dataset Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   url = repo_mgr.create_repo(
       repo_id="username/my-dataset",
       repo_type="dataset",
       private=True
   )

Listing Repositories
--------------------

List your repositories:

.. code-block:: python

   # List your own repositories
   repos = repo_mgr.list_repos()
   
   # List another user's repositories
   repos = repo_mgr.list_repos(username="other-user")
   
   for repo in repos:
       print(repo)

Updating Model Cards
--------------------

Update the README.md (Model Card):

.. code-block:: python

   model_card = """
   ---
   license: mit
   tags:
   - pytorch
   - transformers
   ---

   # My Awesome Model

   ## Model Description
   This model was trained on custom data.

   ## Training Details
   - Framework: PyTorch
   - Optimizer: AdamW
   - Learning Rate: 2e-5
   """

   repo_mgr.update_card("username/my-model", model_card)

Branch Management
-----------------

Create experimental branches:

.. code-block:: python

   # Create a new branch for experiments
   repo_mgr.create_branch(
       repo_id="username/my-model",
       branch="experiment-1"
   )

Checking File Existence
-----------------------

Check if a file exists in the repository:

.. code-block:: python

   exists = repo_mgr.file_exists(
       repo_id="username/my-model",
       filename="config.json"
   )
   
   if exists:
       print("Config file found!")

   # Check on a specific branch
   exists = repo_mgr.file_exists(
       repo_id="username/my-model",
       filename="checkpoint.pt",
       revision="experiment-1"
   )

Deleting Repositories
---------------------

.. warning::
   Deletion is permanent and cannot be undone!

.. code-block:: python

   repo_mgr.delete_repo(
       repo_id="username/my-model",
       repo_type="model"
   )

Error Handling
--------------

The repository module raises ``RepositoryError`` for various failures:

.. code-block:: python

   from hf_lifecycle.exceptions import RepositoryError

   try:
       repo_mgr.create_repo("username/existing-repo", exist_ok=False)
   except RepositoryError as e:
       print(f"Failed to create repository: {e}")

Best Practices
--------------

1. **Use Private Repositories**: Default to ``private=True`` for safety
2. **Enable exist_ok**: Use ``exist_ok=True`` for idempotent scripts
3. **Update Model Cards**: Always provide comprehensive model documentation
4. **Use Branches**: Create branches for experiments to preserve main branch
