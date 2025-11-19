"""
Complete Training Example with HuggingFace Lifecycle Manager

This example demonstrates a full training workflow using all major features:
- Authentication
- Checkpoint management with retention policies
- Training state management with early stopping
- Metadata tracking
- Model registration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from hf_lifecycle.auth import AuthManager
from hf_lifecycle.repo import RepoManager
from hf_lifecycle.checkpoint import CheckpointManager
from hf_lifecycle.retention import CombinedRetentionPolicy, KeepLastN, KeepBestM
from hf_lifecycle.training_state import TrainingStateManager, EarlyStopping
from hf_lifecycle.metadata import MetadataTracker
from hf_lifecycle.model_registry import ModelRegistry


# Simple model for demonstration
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # ========================================================================
    # SETUP
    # ========================================================================
    print("=" * 70)
    print("HuggingFace Lifecycle Manager - Complete Training Example")
    print("=" * 70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Using device: {device}")
    
    # Hyperparameters
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "hidden_size": 128,
        "early_stopping_patience": 5,
    }
    
    # ========================================================================
    # AUTHENTICATION
    # ========================================================================
    print("\n🔐 Step 1: Authentication")
    auth = AuthManager()
    # auth.login(token="your_token_here")  # Uncomment to login
    print("✓ Authentication ready")
    
    # ========================================================================
    # METADATA TRACKING
    # ========================================================================
    print("\n📊 Step 2: Metadata Tracking")
    metadata_tracker = MetadataTracker()
    metadata_tracker.capture_system_info()
    metadata_tracker.capture_environment()
    metadata_tracker.track_hyperparameters(hyperparameters)
    print("✓ Captured system and environment metadata")
    
    # ========================================================================
    # CHECKPOINT MANAGEMENT
    # ========================================================================
    print("\n💾 Step 3: Checkpoint Management")
    repo_mgr = RepoManager(auth)
    
    # Combined retention policy: keep last 3 AND best 2 by validation loss
    retention_policy = CombinedRetentionPolicy([
        KeepLastN(3),
        KeepBestM(2, metric="val_loss", mode="min")
    ])
    
    ckpt_mgr = CheckpointManager(
        repo_manager=repo_mgr,
        local_dir="./checkpoints",
        retention_policy=retention_policy
    )
    print("✓ Checkpoint manager initialized with retention policy")
    
    # ========================================================================
    # TRAINING STATE MANAGEMENT
    # ========================================================================
    print("\n🔄 Step 4: Training State Management")
    state_mgr = TrainingStateManager()
    early_stop = EarlyStopping(
        patience=hyperparameters["early_stopping_patience"],
        mode="min"
    )
    print("✓ Training state manager and early stopping initialized")
    
    # ========================================================================
    # MODEL AND DATA
    # ========================================================================
    print("\n🤖 Step 5: Model and Data Preparation")
    model = SimpleClassifier(
        input_size=784,
        hidden_size=hyperparameters["hidden_size"],
        num_classes=10
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_loader = create_dummy_data(num_samples=1000)
    val_loader = create_dummy_data(num_samples=200)
    print("✓ Model and data ready")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n🚀 Step 6: Training")
    print("-" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(hyperparameters["epochs"]):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        metadata_tracker.track_metrics(metrics, step=epoch)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{hyperparameters['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=epoch,
            metrics=metrics
        )
        
        # Save training state
        state_mgr.save_state(
            path="training_state.pt",
            epoch=epoch,
            step=epoch,
            best_metric=val_loss,
            custom_state={"learning_rate": scheduler.get_last_lr()[0]}
        )
        
        # Track best model
        if state_mgr.is_best(val_loss, mode="min"):
            best_val_loss = val_loss
            print(f"  ⭐ New best model! Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if early_stop.step(val_loss):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        # Cleanup old checkpoints
        if (epoch + 1) % 5 == 0:
            deleted = ckpt_mgr.cleanup(dry_run=False)
            if deleted:
                print(f"  🗑️  Cleaned up {len(deleted)} old checkpoints")
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    # Final metrics
    metadata_tracker.track_metrics({
        "final_val_loss": val_loss,
        "final_val_accuracy": val_acc,
        "best_val_loss": best_val_loss,
    })
    
    # Save metadata
    metadata_tracker.save_metadata("experiment_metadata.json")
    print("\n📊 Metadata saved to: experiment_metadata.json")
    
    # Show summary
    print("\n" + metadata_tracker.get_summary())
    
    # ========================================================================
    # MODEL REGISTRATION (Optional)
    # ========================================================================
    print("\n📦 Step 7: Model Registration (Optional)")
    print("To register your model to HuggingFace Hub:")
    print("  registry = ModelRegistry(repo_mgr)")
    print("  registry.register_model(")
    print("      model=model,")
    print("      repo_id='username/my-model',")
    print("      description='Simple classifier trained with hf-lifecycle',")
    print("      metrics=metadata_tracker.metadata['metrics']['final']")
    print("  )")
    
    print("\n" + "=" * 70)
    print("🎉 Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
