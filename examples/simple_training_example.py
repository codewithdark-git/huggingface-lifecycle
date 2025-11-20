"""
Simplified Training Example using HFManager High-Level API

This example demonstrates how to use the HFManager class for easy experiment tracking,
checkpoint management, and model registration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import PretrainedConfig, PreTrainedModel

from hf_lifecycle import HFManager


# ========================================================================
# Custom Model (inheriting from PreTrainedModel for Hub compatibility)
# ========================================================================

class SimpleConfig(PretrainedConfig):
    model_type = "simple-classifier"
    
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        num_classes=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes


class SimpleClassifier(PreTrainedModel):
    config_class = SimpleConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ========================================================================
# Dummy Dataset
# ========================================================================

def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# ========================================================================
# Training Functions
# ========================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(dataloader), correct / total


# ========================================================================
# Main Training Script
# ========================================================================

def main():
    print("="*70)
    print("HuggingFace Lifecycle Manager - Simplified API Example")
    print("="*70)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 784
    hidden_size = 128
    num_classes = 10
    batch_size = 32
    learning_rate = 0.001
    epochs = 20
    
    # ========================================================================
    # Step 1: Initialize HFManager (One-Stop Solution!)
    # ========================================================================
    print("\n🚀 Step 1: Initializing HFManager")
    
    manager = HFManager(
        repo_id=None,  # Set to "username/model-name" to enable Hub features
        local_dir="./outputs",
        checkpoint_dir="./checkpoints",
        hf_token=None,  # Set your HF token if using Hub
        auto_push=False,  # Set to True to auto-push checkpoints after each save
    )
    
    print("✓ HFManager initialized")
    
    # ========================================================================
    # Step 2: Track Hyperparameters
    # ========================================================================
    print("\n📊 Step 2: Tracking Hyperparameters")
    
    manager.track_hyperparameters({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "early_stopping_patience": 5
    })
    
    print("✓ Hyperparameters tracked")
    
    # ========================================================================
    # Step 3: Prepare Model and Data
    # ========================================================================
    print("\n🔧 Step 3: Preparing Model and Data")
    
    # Create model
    config = SimpleConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes
    )
    model = SimpleClassifier(config).to(device)
    
    # Register model with manager
    manager.set_model(model)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    manager.set_optimizer(optimizer)
    manager.set_scheduler(scheduler)
    
    # Create data
    X_train, y_train = create_dummy_data(1000, input_size, num_classes)
    X_val, y_val = create_dummy_data(200, input_size, num_classes)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    
    print("✓ Model and data ready")
    
    # ========================================================================
    # Step 4: Training Loop
    # ========================================================================
    print("\n🚀 Step 4: Training")
    print("-"*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        manager.log_metrics(metrics, step=epoch)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save checkpoint every epoch
        manager.save_checkpoint(
            epoch=epoch,
            step=epoch,
            metrics=metrics,
            config=config,
            push=False  # Set to True to push this checkpoint to Hub immediately
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
    
    # ========================================================================
    # Step 5: Save Final Model
    # ========================================================================
    print("\n💾 Step 5: Saving Final Model")
    
    manager.save_final_model(
        name="final_model",
        format="safetensors",  # Options: 'pt' or 'safetensors'
        config=config
    )
    
    print("✓ Saved final model (final_model.safetensors)")
    
    # ========================================================================
    # Step 6: Cleanup Old Checkpoints
    # ========================================================================
    print("\n🧹 Step 6: Cleaning Up Old Checkpoints")
    
    deleted = manager.cleanup_checkpoints()
    print(f"✓ Deleted {len(deleted)} old checkpoints")
    
    # ========================================================================
    # Step 7: Save Metadata
    # ========================================================================
    print("\n📄 Step 7: Saving Experiment Metadata")
    
    manager.save_metadata("experiment_metadata.json")
    print("✓ Metadata saved")
    
    # Show summary
    print("\n" + "="*70)
    print(manager.get_summary())
    print("="*70)
    
    # ========================================================================
    # Optional: Push to HuggingFace Hub
    # ========================================================================
    print("\n🚀 Optional: Push to HuggingFace Hub")
    print("To push all artifacts to HuggingFace Hub:")
    print("  1. Set repo_id='username/model-name' and hf_token='your_token' when initializing HFManager")
    print("  2. Call: manager.push(push_checkpoints=True, push_metadata=True, push_final_model=True)")
    print("  3. Or set auto_push=True to auto-push each checkpoint after saving")
    print("")
    print("  # Example:")
    print("  # manager.push(")
    print("  #     push_checkpoints=True,  # Push all checkpoints")
    print("  #     push_metadata=True,     # Push metadata JSON")
    print("  #     push_final_model=True,  # Push final model & config")
    print("  # )")
    
    # ========================================================================
    # Optional: Model Registration
    # ========================================================================
    print("\n📦 Optional: Model Registration")
    print("To register your model to HuggingFace Hub:")
    print("  1. Set repo_id='username/model-name' when initializing HFManager")
    print("  2. Call: manager.register_custom_model(push_to_hub=True)")
    
    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()
