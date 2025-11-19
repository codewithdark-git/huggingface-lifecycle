Dataset Management
==================

The Dataset Manager provides easy-to-use methods for managing datasets on HuggingFace Hub.

Creating Datasets
-----------------

Create Dataset Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.dataset import DatasetManager
   
   dataset_mgr = DatasetManager(repo_manager)
   
   # Create a new dataset repository
   url = dataset_mgr.create_dataset(
       repo_id="username/my-dataset",
       private=True
   )
   
   print(f"Dataset created: {url}")

Uploading Data
--------------

Upload Files
~~~~~~~~~~~~

.. code-block:: python

   # Upload a single file
   dataset_mgr.upload_file(
       repo_id="username/my-dataset",
       file_path="data/train.csv",
       path_in_repo="train.csv"
   )

Upload Folders
~~~~~~~~~~~~~~

.. code-block:: python

   # Upload entire folder
   dataset_mgr.upload_folder(
       repo_id="username/my-dataset",
       folder_path="data/images",
       path_in_repo="images"
   )

Upload DataFrames
~~~~~~~~~~~~~~~~~

One of the most convenient features - directly upload Pandas DataFrames:

.. code-block:: python

   import pandas as pd
   
   # Create a DataFrame
   df = pd.DataFrame({
       "text": ["Hello", "World", "!"],
       "label": [1, 0, 1]
   })
   
   # Upload as Parquet (recommended for large datasets)
   dataset_mgr.upload_dataframe(
       repo_id="username/my-dataset",
       df=df,
       path_in_repo="train.parquet",
       format="parquet"
   )
   
   # Upload as CSV
   dataset_mgr.upload_dataframe(
       repo_id="username/my-dataset",
       df=df,
       path_in_repo="train.csv",
       format="csv"
   )
   
   # Upload as JSON
   dataset_mgr.upload_dataframe(
       repo_id="username/my-dataset",
       df=df,
       path_in_repo="train.json",
       format="json"
   )

Downloading Datasets
--------------------

Download Complete Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Download entire dataset
   path = dataset_mgr.download_dataset(
       repo_id="username/my-dataset",
       local_dir="./data"
   )
   
   print(f"Downloaded to: {path}")

Download Specific Files
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Download only CSV files
   path = dataset_mgr.download_dataset(
       repo_id="username/my-dataset",
       local_dir="./data",
       allow_patterns="*.csv"
   )
   
   # Download multiple patterns
   path = dataset_mgr.download_dataset(
       repo_id="username/my-dataset",
       local_dir="./data",
       allow_patterns=["*.csv", "*.json"]
   )

Download Specific Revision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Download from a specific branch or tag
   path = dataset_mgr.download_dataset(
       repo_id="username/my-dataset",
       local_dir="./data",
       revision="v1.0"
   )

Deleting Datasets
-----------------

.. code-block:: python

   # Delete a dataset repository
   dataset_mgr.delete_dataset("username/my-dataset")

Complete Workflow Example
--------------------------

.. code-block:: python

   from hf_lifecycle.auth import AuthManager
   from hf_lifecycle.repo import RepoManager
   from hf_lifecycle.dataset import DatasetManager
   import pandas as pd
   
   # Setup
   auth = AuthManager()
   repo_mgr = RepoManager(auth)
   dataset_mgr = DatasetManager(repo_mgr)
   
   # Create dataset
   dataset_mgr.create_dataset("username/sentiment-data", private=False)
   
   # Prepare data
   train_df = pd.DataFrame({
       "text": ["Great movie!", "Terrible film", "Loved it!"],
       "sentiment": ["positive", "negative", "positive"]
   })
   
   test_df = pd.DataFrame({
       "text": ["Not bad", "Amazing!"],
       "sentiment": ["neutral", "positive"]
   })
   
   # Upload splits
   dataset_mgr.upload_dataframe(
       "username/sentiment-data",
       train_df,
       "train.parquet",
       format="parquet"
   )
   
   dataset_mgr.upload_dataframe(
       "username/sentiment-data",
       test_df,
       "test.parquet",
       format="parquet"
   )
   
   print("✅ Dataset uploaded successfully!")
   
   # Later: Download for use
   dataset_mgr.download_dataset(
       "username/sentiment-data",
       local_dir="./downloaded_data"
   )

Best Practices
--------------

1. **Use Parquet for large datasets**: More efficient than CSV
2. **Split your data**: Upload train/val/test splits separately
3. **Version your datasets**: Use git tags for dataset versions
4. **Document your data**: Update the dataset card with descriptions
5. **Use private repos**: Keep sensitive data private
