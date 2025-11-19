Utilities and Helpers
=====================

The utilities module provides helper functions for common tasks in ML workflows.

File Operations
---------------

Checksum Calculation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.utils import get_file_checksum
   
   checksum = get_file_checksum("model.pt", algorithm="sha256")
   print(f"SHA256: {checksum}")

Disk Space Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.utils import get_disk_space, check_disk_space
   
   # Check available space
   space = get_disk_space()
   print(f"Free: {space['free'] / 1e9:.2f} GB")
   
   # Verify sufficient space before saving
   if check_disk_space(required_bytes=10 * 1024**3):  # 10 GB
       print("Sufficient space available")

Retry Logic
-----------

Exponential Backoff
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.utils import retry_with_backoff
   
   @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
   def upload_to_hub(file_path):
       # This will retry up to 3 times with exponential backoff
       api.upload_file(file_path)

Device Information
------------------

.. code-block:: python

   from hf_lifecycle.utils import get_device_info
   
   info = get_device_info()
   print(f"CUDA available: {info['cuda_available']}")
   print(f"GPU count: {info['cuda_device_count']}")
   for name in info['device_names']:
       print(f"  - {name}")

Formatting Utilities
--------------------

.. code-block:: python

   from hf_lifecycle.utils import format_bytes, format_time
   
   print(format_bytes(1536000000))  # "1.4 GB"
   print(format_time(3725))  # "1h 2m 5s"

Timing Operations
-----------------

Timer Context Manager
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.utils import Timer
   
   with Timer("Training epoch") as timer:
       train_one_epoch(model, dataloader)
   
   print(f"Elapsed: {timer.elapsed:.2f}s")

Logging
-------

Setup Logging
~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.logger import setup_logger, set_log_level
   import logging
   
   # Setup with file output
   logger = setup_logger(
       name="my_training",
       level=logging.INFO,
       log_file="training.log"
   )
   
   logger.info("Training started")
   
   # Change log level
   set_log_level(logging.DEBUG)
