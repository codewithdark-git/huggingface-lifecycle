Training State Management
=========================

The training state module provides tools for managing complete training state, enabling exact resumption and reproducibility.

TrainingStateManager
--------------------

Save Complete State
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.training_state import TrainingStateManager
   
   state_mgr = TrainingStateManager()
   
   # Save complete training state including RNG states
   state_mgr.save_state(
       path="training_state.pt",
       epoch=10,
       step=1000,
       best_metric=0.85,
       custom_state={"learning_rate": 0.001}
   )

Load and Resume
~~~~~~~~~~~~~~~

.. code-block:: python

   # Load state and restore RNG for reproducibility
   state = state_mgr.load_state("training_state.pt", restore_rng=True)
   
   start_epoch = state["epoch"] + 1
   best_metric = state["best_metric"]
   
   # Continue training from exact same point
   for epoch in range(start_epoch, num_epochs):
       train_one_epoch(model, optimizer)

Track Best Metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check if current metric is best
   if state_mgr.is_best(current_loss, mode="min"):
       print("New best model!")
       save_checkpoint(model, "best_model.pt")

Early Stopping
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.training_state import EarlyStopping
   
   early_stop = EarlyStopping(patience=5, min_delta=0.001, mode="min")
   
   for epoch in range(num_epochs):
       val_loss = validate(model)
       
       if early_stop.step(val_loss):
           print(f"Early stopping triggered at epoch {epoch}")
           break

Save/Load Early Stopping State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save early stopping state
   es_state = early_stop.state_dict()
   torch.save(es_state, "early_stop_state.pt")
   
   # Load early stopping state
   new_early_stop = EarlyStopping(patience=5, mode="min")
   new_early_stop.load_state_dict(torch.load("early_stop_state.pt"))

Complete Training Loop Example
-------------------------------

.. code-block:: python

   from hf_lifecycle.training_state import TrainingStateManager, EarlyStopping
   
   state_mgr = TrainingStateManager()
   early_stop = EarlyStopping(patience=10, mode="min")
   
   # Try to resume from checkpoint
   try:
       state = state_mgr.load_state("training_state.pt", restore_rng=True)
       start_epoch = state["epoch"] + 1
       print(f"Resuming from epoch {start_epoch}")
   except FileNotFoundError:
       start_epoch = 0
       print("Starting fresh training")
   
   for epoch in range(start_epoch, num_epochs):
       # Training
       train_loss = train_one_epoch(model, optimizer)
       val_loss = validate(model)
       
       # Save state
       state_mgr.save_state(
           path="training_state.pt",
           epoch=epoch,
           step=epoch * steps_per_epoch,
           best_metric=val_loss,
       )
       
       # Check early stopping
       if early_stop.step(val_loss):
           print("Early stopping!")
           break
       
       # Save best model
       if state_mgr.is_best(val_loss, mode="min"):
           torch.save(model.state_dict(), "best_model.pt")

Reproducibility
---------------

The `TrainingStateManager` saves and restores RNG states for:

- PyTorch (CPU and CUDA)
- NumPy
- Python's random module

This ensures that when you resume training, you get the exact same random behavior as if training had never stopped.

.. code-block:: python

   # Training will produce identical results when resumed
   state_mgr.save_state("state.pt", epoch=5, step=1000)
   
   # Later...
   state_mgr.load_state("state.pt", restore_rng=True)
   # Next random operations will be identical to continuing from epoch 5
