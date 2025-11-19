Utilities API
=============

.. automodule:: hf_lifecycle.utils
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

.. autofunction:: hf_lifecycle.utils.get_file_checksum
.. autofunction:: hf_lifecycle.utils.get_disk_space
.. autofunction:: hf_lifecycle.utils.check_disk_space
.. autofunction:: hf_lifecycle.utils.retry_with_backoff
.. autofunction:: hf_lifecycle.utils.get_device_info
.. autofunction:: hf_lifecycle.utils.format_bytes
.. autofunction:: hf_lifecycle.utils.format_time

Timer
-----

.. autoclass:: hf_lifecycle.utils.Timer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __enter__, __exit__

Logging API
===========

.. automodule:: hf_lifecycle.logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: hf_lifecycle.logger.setup_logger
.. autofunction:: hf_lifecycle.logger.get_logger
.. autofunction:: hf_lifecycle.logger.set_log_level
