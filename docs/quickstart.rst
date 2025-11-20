Quick Start Guide
=================

This guide will help you get started with HuggingFace Lifecycle Manager using the high-level ``HFManager`` API.

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install huggingface-lifecycle

Basic Usage
-----------

1. Initialize HFManager
~~~~~~~~~~~~~~~~~~~~~~~

The ``HFManager`` class provides a unified interface for all lifecycle operations:

.. code-block:: python

   from hf_lifecycle import HFManager

   manager = HFManager(
       repo_id="username/my-model",
       local_dir="./outputs",
       checkpoint_dir="./checkpoints",
       hf_token="your_token",
       auto_push=False  # Set to True to auto-push checkpoints
   )

You can also set the ``HF_TOKEN`` environment variable:

.. code-block:: bash

   export HF_TOKEN=hf_your_token_here

2. Track Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   manager.track_hyperparameters({
       "learning_rate": 0.001,
       "batch_size": 32,
       "epochs": 10
   })

3. Training Loop
~~~~~~~~~~~~~~~~

.. code-block:: python

   for epoch in range(epochs):
       # Train and validate
       train_loss, train_acc = train_epoch(...)
       val_loss, val_acc = evaluate(...)
       
       # Log metrics
       manager.log_metrics({
           "train_loss": train_loss,
           "train_accuracy": train_acc,
           "val_loss": val_loss,
           "val_accuracy": val_acc
       }, step=epoch)
       
       # Save checkpoint
       manager.save_checkpoint(
           model=model,
           optimizer=optimizer,
           epoch=epoch,
           metrics={"val_loss": val_loss},
           config=config
       )

4. Save Final Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   manager.save_final_model(
       model=model,
       format="safetensors",  # or "pt"
       config=config
   )

5. Push to Hub
~~~~~~~~~~~~~~

Push all artifacts to HuggingFace Hub at once:

.. code-block:: python

   manager.push(
       push_checkpoints=True,
       push_metadata=True,
       push_final_model=True
   )

Next Steps
----------

- Check out the :doc:`api/manager` reference for complete API documentation
- See :doc:`examples <../examples/simple_training_example>` for complete working examples
