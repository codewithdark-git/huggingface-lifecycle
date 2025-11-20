HFManager API Reference
=======================

The ``HFManager`` class provides a high-level, unified interface for all HuggingFace Lifecycle operations. It wraps authentication, repository management, checkpointing, metadata tracking, and model registration into a single easy-to-use class.

Class Definition
----------------

HFManager
~~~~~~~~~

.. code-block:: python

   class HFManager(
       repo_id: Optional[str] = None,
       local_dir: str = "./outputs",
       checkpoint_dir: str = "./checkpoints",
       hf_token: Optional[str] = None,
       private: bool = False,
       retention_policy: Optional[RetentionPolicy] = None,
       auto_push: bool = False,
       model: Optional[torch.nn.Module] = None,
       optimizer: Optional[torch.optim.Optimizer] = None,
       scheduler: Optional[Any] = None
   )

Initialize the HuggingFace Lifecycle Manager.

**Parameters:**

- **repo_id** (*Optional[str]*): The HuggingFace Hub repository ID (e.g., ``"username/model-name"``). If provided, enables Hub features like pushing checkpoints and registering models.
- **local_dir** (*str*, default=``"./outputs"``): Local directory for storing experiment outputs, metadata, and logs.
- **checkpoint_dir** (*str*, default=``"./checkpoints"``): Directory where checkpoints will be saved.
- **hf_token** (*Optional[str]*): HuggingFace API token. If ``None``, uses the token stored in the environment variable ``HF_TOKEN`` or the local HuggingFace cache.
- **private** (*bool*, default=``False``): If creating a new repository, whether to make it private.
- **retention_policy** (*Optional[RetentionPolicy]*): Policy for retaining checkpoints. Defaults to ``KeepLastN(3)`` if not specified.
- **auto_push** (*bool*, default=``False``): If ``True``, automatically pushes checkpoints to the Hub immediately after saving them.
- **model** (*Optional[torch.nn.Module]*): The PyTorch model to manage. Can also be set later using ``set_model()``.
- **optimizer** (*Optional[torch.optim.Optimizer]*): The optimizer. Can also be set later using ``set_optimizer()``.
- **scheduler** (*Optional[Any]*): The learning rate scheduler. Can also be set later using ``set_scheduler()``.

Core Methods
------------

track_hyperparameters
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def track_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None

Track static hyperparameters for the experiment (e.g., learning rate, batch size, architecture config).

**Parameters:**

- **hyperparameters** (*Dict[str, Any]*): A dictionary of hyperparameter names and values.

**Example:**

.. code-block:: python

   manager.track_hyperparameters({
       "learning_rate": 0.001,
       "batch_size": 32,
       "epochs": 10,
       "optimizer": "AdamW"
   })

log_metrics
~~~~~~~~~~~

.. code-block:: python

   def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None

Log dynamic metrics during training (e.g., loss, accuracy).

**Parameters:**

- **metrics** (*Dict[str, float]*): A dictionary of metric names and numeric values.
- **step** (*Optional[int]*): The current training step or epoch.

**Example:**

.. code-block:: python

   manager.log_metrics({"train_loss": 0.5, "val_acc": 0.92}, step=100)

save_checkpoint
~~~~~~~~~~~~~~~

.. code-block:: python

   def save_checkpoint(
       self,
       model: Optional[torch.nn.Module] = None,
       optimizer: Optional[torch.optim.Optimizer] = None,
       scheduler: Optional[Any] = None,
       epoch: Optional[int] = None,
       step: Optional[int] = None,
       metrics: Optional[Dict[str, float]] = None,
       custom_state: Optional[Dict[str, Any]] = None,
       config: Optional[Any] = None,
       name: Optional[str] = None,
       push: Optional[bool] = None
   ) -> str

Save a training checkpoint. This saves the model state, optimizer state, scheduler state, and metadata.

**Parameters:**

- **model** (*Optional[torch.nn.Module]*): The model to save. Uses ``self.model`` if not provided.
- **optimizer** (*Optional[torch.optim.Optimizer]*): The optimizer to save. Uses ``self.optimizer`` if not provided.
- **scheduler** (*Optional[Any]*): The scheduler to save. Uses ``self.scheduler`` if not provided.
- **epoch** (*Optional[int]*): Current epoch number.
- **step** (*Optional[int]*): Current step number.
- **metrics** (*Optional[Dict[str, float]]*): Metrics associated with this checkpoint (useful for ``KeepBestM`` retention policy).
- **custom_state** (*Optional[Dict[str, Any]]*): Any additional custom state to save.
- **config** (*Optional[Any]*): Model configuration object.
- **name** (*Optional[str]*): Custom name for the checkpoint. If ``None``, generates a name like ``checkpoint-step-{step}``.
- **push** (*Optional[bool]*): Whether to push this checkpoint to the Hub immediately. If ``None``, uses the ``auto_push`` setting from initialization.

**Returns:**

- (*str*): The absolute path to the saved checkpoint file.

**Example:**

.. code-block:: python

   manager.save_checkpoint(
       epoch=5,
       step=5000,
       metrics={"val_loss": 0.3},
       push=True  # Force push this checkpoint
   )

save_final_model
~~~~~~~~~~~~~~~~

.. code-block:: python

   def save_final_model(
       self,
       model: Optional[torch.nn.Module] = None,
       name: str = "final_model",
       format: str = "pt",
       config: Optional[Any] = None
   ) -> str

Save the final trained model in the root directory (not in ``checkpoints/``).

**Parameters:**

- **model** (*Optional[torch.nn.Module]*): The model to save. Uses ``self.model`` if not provided.
- **name** (*str*, default=``"final_model"``): Base filename for the model (without extension).
- **format** (*str*, default=``"pt"``): Format to save. Options: ``"pt"`` (PyTorch) or ``"safetensors"``.
- **config** (*Optional[Any]*): Model configuration to save alongside the model as ``config.json``.

**Returns:**

- (*str*): The absolute path to the saved model file.

**Example:**

.. code-block:: python

   manager.save_final_model(format="safetensors", config=my_config)

push
~~~~

.. code-block:: python

   def push(
       self,
       push_checkpoints: bool = True,
       push_metadata: bool = True,
       push_final_model: bool = False,
       commit_message: str = "Upload training artifacts"
   ) -> None

Push all tracked artifacts to the HuggingFace Hub. This is useful for batch uploading at the end of training.

**Parameters:**

- **push_checkpoints** (*bool*, default=``True``): Whether to upload all saved checkpoints.
- **push_metadata** (*bool*, default=``True``): Whether to upload the ``experiment_metadata.json`` file.
- **push_final_model** (*bool*, default=``False``): Whether to upload the final model file and config.
- **commit_message** (*str*): Commit message for the upload.

**Example:**

.. code-block:: python

   manager.push(push_final_model=True, commit_message="Training complete")

cleanup_checkpoints
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def cleanup_checkpoints(self, dry_run: bool = False) -> List[str]

Apply the configured retention policy (e.g., ``KeepLastN``) to delete old checkpoints from the local disk.

**Parameters:**

- **dry_run** (*bool*, default=``False``): If ``True``, returns the list of checkpoints that *would* be deleted without actually deleting them.

**Returns:**

- (*List[str]*): List of checkpoint names that were deleted (or would be deleted).

register_custom_model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def register_custom_model(
       self,
       model: Optional[torch.nn.Module] = None,
       config: Optional[Any] = None,
       repo_id: Optional[str] = None,
       model_name: Optional[str] = None,
       description: str = "",
       push_to_hub: bool = False,
       commit_message: str = "Upload custom model"
   ) -> Optional[str]

Register a custom model architecture with HuggingFace ``AutoClasses``. This allows users to load your model using ``AutoModel.from_pretrained()``.

**Parameters:**

- **model** (*Optional[torch.nn.Module]*): The custom model instance.
- **config** (*Optional[Any]*): The custom configuration object (must inherit from ``PretrainedConfig``).
- **repo_id** (*Optional[str]*): Repository ID to register to. Uses ``self.repo_id`` if not provided.
- **model_name** (*Optional[str]*): Name of the model class.
- **description** (*str*): Description for the model card.
- **push_to_hub** (*bool*, default=``False``): Whether to push the registered code and model to the Hub.
- **commit_message** (*str*): Commit message.

**Returns:**

- (*Optional[str]*): The URL of the uploaded model if ``push_to_hub`` is True, else ``None``.

load_checkpoint
~~~~~~~~~~~~~~~

.. code-block:: python

   def load_checkpoint(
       self,
       name: str,
       model: Optional[torch.nn.Module] = None,
       optimizer: Optional[torch.optim.Optimizer] = None,
       scheduler: Optional[Any] = None,
       map_location: Optional[Any] = None
   ) -> Dict[str, Any]

Load a specific checkpoint by name.

**Parameters:**

- **name** (*str*): The name of the checkpoint to load (e.g., ``"checkpoint-step-1000"``).
- **model** (*Optional[torch.nn.Module]*): Model to load state into. Uses ``self.model`` if not provided.
- **optimizer** (*Optional[torch.optim.Optimizer]*): Optimizer to load state into. Uses ``self.optimizer`` if not provided.
- **scheduler** (*Optional[Any]*): Scheduler to load state into. Uses ``self.scheduler`` if not provided.
- **map_location** (*Optional[Any]*): Device to map tensors to (e.g., ``"cpu"`` or ``"cuda"``).

**Returns:**

- (*Dict[str, Any]*): The loaded checkpoint dictionary containing model state, optimizer state, etc.

load_latest_checkpoint
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def load_latest_checkpoint(
       self,
       model: Optional[torch.nn.Module] = None,
       optimizer: Optional[torch.optim.Optimizer] = None,
       scheduler: Optional[Any] = None,
       map_location: Optional[Any] = None
   ) -> Optional[Dict[str, Any]]

Load the most recent checkpoint found in the ``checkpoint_dir``. This is useful for automatically resuming training.

**Parameters:**

- **model** (*Optional[torch.nn.Module]*): Model to load state into. Uses ``self.model`` if not provided.
- **optimizer** (*Optional[torch.optim.Optimizer]*): Optimizer to load state into. Uses ``self.optimizer`` if not provided.
- **scheduler** (*Optional[Any]*): Scheduler to load state into. Uses ``self.scheduler`` if not provided.
- **map_location** (*Optional[Any]*): Device to map tensors to.

**Returns:**

- (*Optional[Dict[str, Any]]*): The loaded checkpoint dictionary, or ``None`` if no checkpoints are found.

**Example:**

.. code-block:: python

   # Automatically resume from latest checkpoint if available
   manager.load_latest_checkpoint()

list_checkpoints
~~~~~~~~~~~~~~~~

.. code-block:: python

   def list_checkpoints(self) -> List[Dict[str, Any]]

List all available checkpoints in the ``checkpoint_dir``, sorted by creation time (newest first).

**Returns:**

- (*List[Dict[str, Any]]*): A list of dictionaries, each containing metadata about a checkpoint (name, path, metrics, timestamp).

Helper Methods
--------------

- **set_model(model)**: Update the internal model reference.
- **set_optimizer(optimizer)**: Update the internal optimizer reference.
- **set_scheduler(scheduler)**: Update the internal scheduler reference.
- **get_summary() -> str**: Returns a formatted string summary of the experiment metadata.
- **save_metadata(filename)**: Manually save the metadata to a JSON file.
