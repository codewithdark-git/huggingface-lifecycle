Checkpoint Management
=====================

The ``CheckpointManager`` provides intelligent checkpoint saving, loading, and retention for training workflows.

Creating a Checkpoint Manager
------------------------------

.. code-block:: python

   from hf_lifecycle.checkpoint import CheckpointManager
   from hf_lifecycle.retention import KeepLastN
   
   ckpt_mgr = CheckpointManager(
       repo_manager=repo_mgr,
       local_dir="./checkpoints",
       retention_policy=KeepLastN(3)
   )

Saving Checkpoints
------------------

Basic Save
~~~~~~~~~~

.. code-block:: python

   ckpt_mgr.save(
       model=model,
       optimizer=optimizer,
       epoch=10,
       step=1000
   )

Save with Metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   ckpt_mgr.save(
       model=model,
       optimizer=optimizer,
       scheduler=scheduler,
       epoch=10,
       step=1000,
       metrics={"loss": 0.25, "accuracy": 0.95}
   )

Custom Checkpoint Name
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ckpt_mgr.save(
       model=model,
       name="best-model-v1"
   )

Loading Checkpoints
-------------------

Load Latest
~~~~~~~~~~~

.. code-block:: python

   checkpoint = ckpt_mgr.load_latest(
       model=model,
       optimizer=optimizer,
       scheduler=scheduler
   )
   
   # Access checkpoint data
   epoch = checkpoint["epoch"]
   step = checkpoint["step"]
   metrics = checkpoint["metrics"]

Load Best by Metric
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load checkpoint with minimum loss
   checkpoint = ckpt_mgr.load_best(
       metric="loss",
       mode="min",
       model=model
   )
   
   # Load checkpoint with maximum accuracy
   checkpoint = ckpt_mgr.load_best(
       metric="accuracy",
       mode="max",
       model=model
   )

Load Specific Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   checkpoint = ckpt_mgr.load(
       name="checkpoint-step-1000",
       model=model,
       optimizer=optimizer
   )

Retention Policies
------------------

Keep Last N
~~~~~~~~~~~

Keep only the N most recent checkpoints:

.. code-block:: python

   from hf_lifecycle.retention import KeepLastN
   
   policy = KeepLastN(n=5)  # Keep last 5 checkpoints

Keep Best M
~~~~~~~~~~~

Keep the M best checkpoints based on a metric:

.. code-block:: python

   from hf_lifecycle.retention import KeepBestM
   
   # Keep 3 checkpoints with lowest loss
   policy = KeepBestM(m=3, metric="loss", mode="min")
   
   # Keep 2 checkpoints with highest accuracy
   policy = KeepBestM(m=2, metric="accuracy", mode="max")

Combined Policies
~~~~~~~~~~~~~~~~~

Combine multiple retention policies:

.. code-block:: python

   from hf_lifecycle.retention import CombinedRetentionPolicy, KeepLastN, KeepBestM
   
   # Keep last 3 OR best 2 by loss (union)
   policy = CombinedRetentionPolicy([
       KeepLastN(3),
       KeepBestM(2, metric="loss", mode="min")
   ])

Custom Retention Policy
~~~~~~~~~~~~~~~~~~~~~~~

Create custom policies using a callback:

.. code-block:: python

   from hf_lifecycle.retention import CustomRetentionPolicy
   
   def keep_every_10th_epoch(checkpoints):
       return [
           ckpt["name"] 
           for ckpt in checkpoints 
           if ckpt.get("epoch", 0) % 10 == 0
       ]
   
   policy = CustomRetentionPolicy(keep_every_10th_epoch)

Applying Retention Policies
----------------------------

Dry Run
~~~~~~~

Preview what would be deleted:

.. code-block:: python

   deleted = ckpt_mgr.cleanup(dry_run=True)
   print(f"Would delete: {deleted}")

Actual Cleanup
~~~~~~~~~~~~~~

Apply retention policy and delete old checkpoints:

.. code-block:: python

   deleted = ckpt_mgr.cleanup(dry_run=False)
   print(f"Deleted: {deleted}")

Complete Training Example
--------------------------

.. code-block:: python

   from hf_lifecycle.checkpoint import CheckpointManager
   from hf_lifecycle.retention import CombinedRetentionPolicy, KeepLastN, KeepBestM
   
   # Setup
   policy = CombinedRetentionPolicy([
       KeepLastN(3),  # Keep last 3 checkpoints
       KeepBestM(2, metric="loss", mode="min")  # Keep 2 best
   ])
   
   ckpt_mgr = CheckpointManager(
       repo_manager=repo_mgr,
       retention_policy=policy
   )
   
   # Training loop
   for epoch in range(100):
       train_loss = train_one_epoch(model, optimizer)
       val_loss = validate(model)
       
       # Save checkpoint
       ckpt_mgr.save(
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           epoch=epoch,
           step=epoch * steps_per_epoch,
           metrics={"train_loss": train_loss, "val_loss": val_loss}
       )
       
       # Cleanup old checkpoints
       if epoch % 10 == 0:
           ckpt_mgr.cleanup()
   
   # Resume training later
   checkpoint = ckpt_mgr.load_latest(model=model, optimizer=optimizer)
   if checkpoint:
       start_epoch = checkpoint["epoch"] + 1
       print(f"Resuming from epoch {start_epoch}")

Best Practices
--------------

1. **Save Regularly**: Save checkpoints at regular intervals (every N steps or epochs)
2. **Include Metrics**: Always include relevant metrics for intelligent retention
3. **Use Combined Policies**: Combine KeepLastN and KeepBestM to balance recency and quality
4. **Periodic Cleanup**: Run cleanup periodically, not after every save
5. **Dry Run First**: Use dry_run=True to verify what will be deleted
6. **Custom State**: Use custom_state parameter for any additional training state
