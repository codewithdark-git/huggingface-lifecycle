"""
Checkpoint management for HuggingFace Lifecycle Manager.
"""
import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging

from huggingface_hub import HfApi, snapshot_download, upload_folder, hf_hub_download
try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from hf_lifecycle.repo import RepoManager
from hf_lifecycle.retention import RetentionPolicy, KeepLastN
from hf_lifecycle.exceptions import (
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointCorruptedError,
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and retention for training workflows.
    """

    def __init__(
        self,
        repo_manager: RepoManager,
        local_dir: str = "./checkpoints",
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        """
        Initialize the CheckpointManager.

        Args:
            repo_manager: Repository manager for uploading checkpoints.
            local_dir: Local directory for storing checkpoints.
            retention_policy: Policy for managing checkpoint retention.
                Defaults to KeepLastN(3).
        """
        self.repo_manager = repo_manager
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.retention_policy = retention_policy or KeepLastN(3)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        custom_state: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint with model, optimizer, and other training state.

        Args:
            model: PyTorch model to save.
            optimizer: Optimizer state to save.
            scheduler: Learning rate scheduler state to save.
            epoch: Current epoch number.
            step: Current step number.
            metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.9}).
            metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.9}).
            custom_state: Any custom state dictionary to save.
            config: Model configuration object (optional).
            name: Custom checkpoint name. If None, auto-generates based on step/epoch.

        Returns:
            Path to the saved checkpoint.

        Raises:
            CheckpointError: If save fails.
        """
        try:
            # Generate checkpoint name
            if name is None:
                if step is not None:
                    name = f"checkpoint-step-{step}"
                elif epoch is not None:
                    name = f"checkpoint-epoch-{epoch}"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    name = f"checkpoint-{timestamp}"

            # Build checkpoint dictionary
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                "custom_state": custom_state or {},
                "timestamp": datetime.now().isoformat(),
            }

            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            # Save checkpoint directly
            checkpoint_path = self.local_dir / f"{name}.pt"
            torch.save(checkpoint, checkpoint_path)

            # Save metadata separately for easy inspection
            metadata = {
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                "timestamp": checkpoint["timestamp"],
            }

            # Add config if available
            model_config = config
            if model_config is None and hasattr(model, "config"):
                model_config = model.config
            
            if model_config is not None:
                if hasattr(model_config, "to_dict"):
                    metadata.update(model_config.to_dict())
                elif isinstance(model_config, dict):
                    metadata.update(model_config)
            metadata_path = self.local_dir / f"{name}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved checkpoint: {name}")
            return str(checkpoint_path)

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def save_final_model(
        self,
        model: torch.nn.Module,
        name: str = "final_model",
        format: str = "pt",
        config: Optional[Any] = None,
    ) -> str:
        """
        Save the final model to the root directory (parent of checkpoints dir).

        Args:
            model: PyTorch model to save.
            name: Name of the file (without extension).
            format: Format to save in ('pt' or 'safetensors').
            config: Optional model configuration to save as config.json.

        Returns:
            Path to the saved model file.
        """
        try:
            root_dir = self.local_dir.parent
            
            # Save config if provided or available on model
            model_config = config
            if model_config is None and hasattr(model, "config"):
                model_config = model.config
            
            if model_config is not None:
                config_path = root_dir / "config.json"
                # User wants full details (including defaults), so we prefer to_dict() over save_pretrained()
                if hasattr(model_config, "to_dict"):
                    with open(config_path, "w") as f:
                        json.dump(model_config.to_dict(), f, indent=2)
                elif isinstance(model_config, dict):
                    with open(config_path, "w") as f:
                        json.dump(model_config, f, indent=2)
                elif hasattr(model_config, "save_pretrained"):
                    model_config.save_pretrained(root_dir)
                logger.info(f"Saved config to {config_path}")

            # Save model weights
            if format == "safetensors":
                if not SAFETENSORS_AVAILABLE:
                    raise ImportError("safetensors is not installed. Install it with 'pip install safetensors'.")
                
                path = root_dir / f"{name}.safetensors"
                save_safetensors(model.state_dict(), path)
            else:
                path = root_dir / f"{name}.pt"
                torch.save(model.state_dict(), path)
            
            logger.info(f"Saved final model to {path}")
            return str(path)

        except Exception as e:
            raise CheckpointError(f"Failed to save final model: {e}")

    def load(
        self,
        name: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint by name.

        Args:
            name: Name of the checkpoint to load.
            model: Model to load state into (if provided).
            optimizer: Optimizer to load state into (if provided).
            scheduler: Scheduler to load state into (if provided).
            map_location: Device to map tensors to.

        Returns:
            Checkpoint dictionary containing all saved state.

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist.
            CheckpointCorruptedError: If checkpoint is corrupted.
        """
        checkpoint_path = self.local_dir / f"{name}.pt"

        if not checkpoint_path.exists():
            # Fallback for old directory structure
            old_path = self.local_dir / name / "checkpoint.pt"
            if old_path.exists():
                checkpoint_path = old_path
            else:
                raise CheckpointNotFoundError(f"Checkpoint not found: {name}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Load states if objects provided
            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded model state from {name}")

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"Loaded optimizer state from {name}")

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info(f"Loaded scheduler state from {name}")

            return checkpoint

        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to load checkpoint {name}: {e}")

    def load_latest(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            map_location: Device to map tensors to.

        Returns:
            Checkpoint dictionary, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.warning("No checkpoints found")
            return None

        # Sort by step (or timestamp if no step)
        latest = max(checkpoints, key=lambda x: x.get("step", 0))
        return self.load(
            latest["name"], model, optimizer, scheduler, map_location
        )

    def load_best(
        self,
        metric: str,
        mode: str = "min",
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint based on a metric.

        Args:
            metric: Metric name to use for comparison.
            mode: 'min' for lower is better, 'max' for higher is better.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            map_location: Device to map tensors to.

        Returns:
            Checkpoint dictionary, or None if no checkpoints with metric exist.
        """
        checkpoints = self.list_checkpoints()
        valid_ckpts = [
            ckpt
            for ckpt in checkpoints
            if ckpt.get("metrics") and metric in ckpt["metrics"]
        ]

        if not valid_ckpts:
            logger.warning(f"No checkpoints found with metric '{metric}'")
            return None

        # Find best
        reverse = mode == "max"
        best = sorted(
            valid_ckpts, key=lambda x: x["metrics"][metric], reverse=reverse
        )[0]

        logger.info(f"Loading best checkpoint by {metric}: {best['name']}")
        return self.load(best["name"], model, optimizer, scheduler, map_location)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.

        Returns:
            List of checkpoint metadata dictionaries.
        """
        checkpoints = []

        checkpoints = []

        if not self.local_dir.exists():
            return checkpoints

        for metadata_path in self.local_dir.glob("*.json"):
            if not metadata_path.is_file():
                continue
            
            # Skip if it's not a checkpoint metadata file (heuristic: must have corresponding .pt)
            name = metadata_path.stem
            checkpoint_path = self.local_dir / f"{name}.pt"
            
            if not checkpoint_path.exists():
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata["name"] = name
                checkpoints.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {name}: {e}")

        return checkpoints

    def cleanup(self, dry_run: bool = False) -> List[str]:
        """
        Apply retention policy to remove old checkpoints.

        Args:
            dry_run: If True, only show what would be deleted without deleting.

        Returns:
            List of deleted checkpoint names.
        """
        checkpoints = self.list_checkpoints()
        to_keep = set(self.retention_policy.select_checkpoints_to_keep(checkpoints))
        to_delete = [ckpt["name"] for ckpt in checkpoints if ckpt["name"] not in to_keep]

        if dry_run:
            logger.info(f"[DRY RUN] Would delete {len(to_delete)} checkpoints: {to_delete}")
            return to_delete

        deleted = []
        deleted = []
        for name in to_delete:
            try:
                # Try deleting flat files
                pt_path = self.local_dir / f"{name}.pt"
                json_path = self.local_dir / f"{name}.json"
                
                if pt_path.exists():
                    pt_path.unlink()
                if json_path.exists():
                    json_path.unlink()
                    
                # Try deleting directory (old structure)
                dir_path = self.local_dir / name
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                
                deleted.append(name)
                logger.info(f"Deleted checkpoint: {name}")
            except Exception as e:
                logger.error(f"Failed to delete {name}: {e}")

        return deleted

    def upload_to_hub(
        self,
        repo_id: str,
        checkpoint_name: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> None:
        """
        Upload checkpoint(s) to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub.
            checkpoint_name: Specific checkpoint to upload. If None, uploads all.
            commit_message: Git commit message for the upload.

        Raises:
            CheckpointNotFoundError: If specified checkpoint doesn't exist.
            CheckpointError: If upload fails.
        """
        try:
            api = self.repo_manager._api
            
            if checkpoint_name:
                # Upload specific checkpoint
                pt_path = self.local_dir / f"{checkpoint_name}.pt"
                json_path = self.local_dir / f"{checkpoint_name}.json"
                
                if not pt_path.exists():
                    # Check for old directory structure
                    dir_path = self.local_dir / checkpoint_name
                    if dir_path.exists() and dir_path.is_dir():
                         # Upload all files in checkpoint directory
                        for file_path in dir_path.rglob("*"):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(dir_path)
                                path_in_repo = f"checkpoints/{checkpoint_name}/{relative_path}"
                                
                                api.upload_file(
                                    path_or_fileobj=str(file_path),
                                    path_in_repo=path_in_repo,
                                    repo_id=repo_id,
                                    commit_message=commit_message
                                    or f"Upload checkpoint {checkpoint_name}",
                                )
                        logger.info(f"Uploaded checkpoint {checkpoint_name} to {repo_id}")
                        return

                    raise CheckpointNotFoundError(
                        f"Checkpoint not found: {checkpoint_name}"
                    )
                
                # Upload flat files
                api.upload_file(
                    path_or_fileobj=str(pt_path),
                    path_in_repo=f"checkpoints/{checkpoint_name}.pt",
                    repo_id=repo_id,
                    commit_message=commit_message or f"Upload checkpoint {checkpoint_name}",
                )
                
                if json_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(json_path),
                        path_in_repo=f"checkpoints/{checkpoint_name}.json",
                        repo_id=repo_id,
                        commit_message=commit_message or f"Upload checkpoint {checkpoint_name}",
                    )
                
                logger.info(f"Uploaded checkpoint {checkpoint_name} to {repo_id}")
            else:
                # Upload all checkpoints
                checkpoints = self.list_checkpoints()
                for ckpt in checkpoints:
                    self.upload_to_hub(repo_id, ckpt["name"], commit_message)
                
                logger.info(f"Uploaded {len(checkpoints)} checkpoints to {repo_id}")
                
        except CheckpointNotFoundError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to upload checkpoint to Hub: {e}")

    def download_from_hub(
        self,
        repo_id: str,
        checkpoint_name: str,
        revision: Optional[str] = None,
    ) -> str:
        """
        Download a checkpoint from HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub.
            checkpoint_name: Name of checkpoint to download.
            revision: Git revision (branch, tag, commit) to download from.

        Returns:
            Path to downloaded checkpoint directory.

        Raises:
            CheckpointError: If download fails.
        """
        try:
            from huggingface_hub import hf_hub_download
            
            # Try downloading flat files
            try:
                pt_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"checkpoints/{checkpoint_name}.pt",
                    revision=revision,
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False
                )
                
                # Try downloading metadata if it exists
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=f"checkpoints/{checkpoint_name}.json",
                        revision=revision,
                        local_dir=self.local_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception:
                    pass # Metadata might not exist
                    
                logger.info(f"Downloaded checkpoint {checkpoint_name} from {repo_id}")
                return str(Path(pt_path).parent)
                
            except Exception:
                # Fallback to directory structure download
                pass

            # Download specific checkpoint directory from hub
            cache_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"checkpoints/{checkpoint_name}/*",
                revision=revision,
            )
            
            # Copy to local checkpoint directory
            checkpoint_dir = self.local_dir / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            src_dir = Path(cache_dir) / "checkpoints" / checkpoint_name
            if src_dir.exists():
                for item in src_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, checkpoint_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(
                            item, checkpoint_dir / item.name, dirs_exist_ok=True
                        )
            
            logger.info(f"Downloaded checkpoint {checkpoint_name} from {repo_id}")
            return str(checkpoint_dir)
            
        except Exception as e:
            raise CheckpointError(f"Failed to download checkpoint from Hub: {e}")
