"""
High-level manager for HuggingFace Lifecycle.
"""
from typing import Optional, Dict, Any, List
import torch
from pathlib import Path

from hf_lifecycle.repo import RepoManager
from hf_lifecycle.metadata import MetadataTracker
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.model_registry import ModelRegistry
from hf_lifecycle.retention import RetentionPolicy, KeepLastN


class HFManager:
    """
    High-level manager that simplifies the HuggingFace Lifecycle API.
    
    This class wraps all the individual managers (RepoManager, MetadataTracker,
    CheckpointManager, ModelRegistry) into a single, easy-to-use interface.
    
    Example:
        ```python
        manager = HFManager(
            repo_id="username/my-model",
            local_dir="./outputs",
            hf_token="your_token"
        )
        
        # Track hyperparameters
        manager.track_hyperparameters({"lr": 0.001, "batch_size": 32})
        
        # Log metrics during training
        manager.log_metrics({"loss": 0.5, "accuracy": 0.9})
        
        # Save checkpoint
        manager.save_checkpoint(model, optimizer, epoch=1)
        
        # Save final model
        manager.save_final_model(model, format="safetensors")
        ```
    """
    
    def __init__(
        self,
        repo_id: Optional[str] = None,
        local_dir: str = "./outputs",
        checkpoint_dir: str = "./checkpoints",
        hf_token: Optional[str] = None,
        private: bool = False,
        retention_policy: Optional[RetentionPolicy] = None,
        auto_push: bool = False,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize the HFManager.
        
        Args:
            repo_id: HuggingFace Hub repository ID (e.g., "username/model-name").
            local_dir: Local directory for outputs and metadata.
            checkpoint_dir: Directory for saving checkpoints.
            hf_token: HuggingFace API token for Hub operations.
            private: Whether to create a private repository.
            retention_policy: Checkpoint retention policy. Defaults to KeepLastN(3).
            auto_push: If True, automatically push checkpoints to Hub after saving.
            model: PyTorch model (can be set later).
            optimizer: Optimizer (can be set later).
            scheduler: Learning rate scheduler (can be set later).
        """
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_push = auto_push
        
        # Store model, optimizer, scheduler references
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Initialize sub-managers
        self.repo_manager = RepoManager(
            repo_id=repo_id,
            token=hf_token,
            private=private
        ) if repo_id else None
        
        self.metadata_tracker = MetadataTracker()
        
        self.checkpoint_manager = CheckpointManager(
            repo_manager=self.repo_manager,
            local_dir=str(self.checkpoint_dir),
            retention_policy=retention_policy or KeepLastN(3)
        )
        
        self.model_registry = ModelRegistry(
            repo_manager=self.repo_manager
        ) if repo_id else None
    
    def set_model(self, model: torch.nn.Module):
        """Set or update the model reference."""
        self.model = model
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set or update the optimizer reference."""
        self.optimizer = optimizer
    
    def set_scheduler(self, scheduler: Any):
        """Set or update the scheduler reference."""
        self.scheduler = scheduler
    
    # ========================================================================
    # Metadata Tracking Methods
    # ========================================================================
    
    def track_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Track hyperparameters for the experiment.
        
        Args:
            hyperparameters: Dictionary of hyperparameters.
        """
        self.metadata_tracker.track_hyperparameters(hyperparameters)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current training step.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number.
        """
        self.metadata_tracker.track_metrics(metrics, step=step)
    
    def get_summary(self) -> str:
        """Get a summary of the experiment."""
        return self.metadata_tracker.get_summary()
    
    def save_metadata(self, filename: str = "experiment_metadata.json"):
        """Save metadata to a JSON file."""
        self.metadata_tracker.save_metadata(filename)
    
    # ========================================================================
    # Checkpoint Management Methods
    # ========================================================================
    
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
        push: Optional[bool] = None,
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save (uses self.model if not provided).
            optimizer: Optimizer to save (uses self.optimizer if not provided).
            scheduler: Scheduler to save (uses self.scheduler if not provided).
            epoch: Current epoch number.
            step: Current step number.
            metrics: Dictionary of metrics.
            custom_state: Any custom state to save.
            config: Model configuration.
            name: Custom checkpoint name.
            push: If True, push to Hub after saving. If None, uses self.auto_push.
            
        Returns:
            Path to saved checkpoint.
        """
        model = model or self.model
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler
        
        if model is None:
            raise ValueError("No model provided. Set model via __init__ or set_model().")
        
        checkpoint_path = self.checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            custom_state=custom_state,
            config=config,
            name=name
        )
        
        # Push to Hub if enabled
        should_push = push if push is not None else self.auto_push
        if should_push and self.repo_id:
            # Extract checkpoint name from path
            checkpoint_name = Path(checkpoint_path).stem
            self.checkpoint_manager.upload_to_hub(
                repo_id=self.repo_id,
                checkpoint_name=checkpoint_name
            )
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        name: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint by name.
        
        Args:
            name: Checkpoint name.
            model: Model to load state into (uses self.model if not provided).
            optimizer: Optimizer to load state into (uses self.optimizer if not provided).
            scheduler: Scheduler to load state into (uses self.scheduler if not provided).
            map_location: Device to map tensors to.
            
        Returns:
            Checkpoint dictionary.
        """
        model = model or self.model
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler
        
        return self.checkpoint_manager.load(
            name=name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location
        )
    
    def load_latest_checkpoint(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into (uses self.model if not provided).
            optimizer: Optimizer to load state into (uses self.optimizer if not provided).
            scheduler: Scheduler to load state into (uses self.scheduler if not provided).
            map_location: Device to map tensors to.
            
        Returns:
            Checkpoint dictionary, or None if no checkpoints exist.
        """
        model = model or self.model
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler
        
        return self.checkpoint_manager.load_latest(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location
        )
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoint_manager.list_checkpoints()
    
    def cleanup_checkpoints(self, dry_run: bool = False) -> List[str]:
        """
        Apply retention policy to remove old checkpoints.
        
        Args:
            dry_run: If True, only show what would be deleted.
            
        Returns:
            List of deleted checkpoint names.
        """
        return self.checkpoint_manager.cleanup(dry_run=dry_run)
    
    def save_final_model(
        self,
        model: Optional[torch.nn.Module] = None,
        name: str = "final_model",
        format: str = "pt",
        config: Optional[Any] = None,
    ) -> str:
        """
        Save the final trained model to the root directory.
        
        Args:
            model: Model to save (uses self.model if not provided).
            name: Model filename (without extension).
            format: Format to save ('pt' or 'safetensors').
            config: Model configuration.
            
        Returns:
            Path to saved model.
        """
        model = model or self.model
        
        if model is None:
            raise ValueError("No model provided. Set model via __init__ or set_model().")
        
        return self.checkpoint_manager.save_final_model(
            model=model,
            name=name,
            format=format,
            config=config
        )
    
    # ========================================================================
    # Hub Push Methods
    # ========================================================================
    
    def push(
        self,
        push_checkpoints: bool = True,
        push_metadata: bool = True,
        push_final_model: bool = False,
        commit_message: str = "Upload training artifacts",
    ) -> None:
        """
        Push all tracked items to HuggingFace Hub.
        
        Args:
            push_checkpoints: Whether to push all checkpoints.
            push_metadata: Whether to push metadata JSON file.
            push_final_model: Whether to push the final model file.
            commit_message: Commit message for all uploads.
        """
        if not self.repo_id:
            raise ValueError("No repo_id provided. Cannot push to Hub.")
        
        if not self.repo_manager:
            raise ValueError("RepoManager not initialized. Cannot push to Hub.")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Push checkpoints
        if push_checkpoints:
            checkpoints = self.list_checkpoints()
            logger.info(f"Pushing {len(checkpoints)} checkpoint(s) to Hub...")
            for ckpt in checkpoints:
                try:
                    self.checkpoint_manager.upload_to_hub(
                        repo_id=self.repo_id,
                        checkpoint_name=ckpt["name"],
                        commit_message=commit_message
                    )
                except Exception as e:
                    logger.warning(f"Failed to push checkpoint {ckpt['name']}: {e}")
        
        # Push metadata
        if push_metadata:
            from huggingface_hub import HfApi
            api = HfApi()
            metadata_file = self.local_dir / "experiment_metadata.json"
            
            if metadata_file.exists():
                logger.info("Pushing metadata to Hub...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(metadata_file),
                        path_in_repo="experiment_metadata.json",
                        repo_id=self.repo_id,
                        commit_message=commit_message
                    )
                except Exception as e:
                    logger.warning(f"Failed to push metadata: {e}")
            else:
                logger.warning("Metadata file not found. Skipping metadata push.")
        
        # Push final model
        if push_final_model:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Check for both .pt and .safetensors
            final_model_files = []
            for ext in [".pt", ".safetensors"]:
                model_file = self.checkpoint_dir.parent / f"final_model{ext}"
                if model_file.exists():
                    final_model_files.append(model_file)
            
            config_file = self.checkpoint_dir.parent / "config.json"
            if config_file.exists():
                final_model_files.append(config_file)
            
            if final_model_files:
                logger.info("Pushing final model to Hub...")
                for file in final_model_files:
                    try:
                        api.upload_file(
                            path_or_fileobj=str(file),
                            path_in_repo=file.name,
                            repo_id=self.repo_id,
                            commit_message=commit_message
                        )
                    except Exception as e:
                        logger.warning(f"Failed to push {file.name}: {e}")
            else:
                logger.warning("No final model files found. Skipping final model push.")
        
        logger.info(f"✓ Pushed artifacts to {self.repo_id}")
    
    def register_custom_model(
        self,
        model: Optional[torch.nn.Module] = None,
        config: Optional[Any] = None,
        repo_id: Optional[str] = None,
        model_name: Optional[str] = None,
        description: str = "",
        push_to_hub: bool = False,
        commit_message: str = "Upload custom model",
    ) -> Optional[str]:
        """
        Register a custom model architecture with HuggingFace transformers.
        
        Args:
            model: Custom model to register (uses self.model if not provided).
            config: Custom config (must inherit from PretrainedConfig).
            repo_id: Repository ID (uses self.repo_id if not provided).
            model_name: Name for the model.
            description: Model description.
            push_to_hub: Whether to push to Hub after registration.
            commit_message: Commit message for the upload.
            
        Returns:
            URL of uploaded model if push_to_hub=True, else None.
        """
        if self.model_registry is None:
            raise ValueError("Model registry not available. Provide repo_id during initialization.")
        
        model = model or self.model
        repo_id = repo_id or self.repo_id
        
        if model is None:
            raise ValueError("No model provided. Set model via __init__ or set_model().")
        
        return self.model_registry.register_custom_model(
            model=model,
            config=config,
            repo_id=repo_id,
            model_name=model_name,
            description=description,
            push_to_hub=push_to_hub,
            commit_message=commit_message
        )
