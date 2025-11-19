Exceptions API
==============

.. automodule:: hf_lifecycle.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Hierarchy
-------------------

.. code-block:: text

   HfLifecycleError
   ├── AuthenticationError
   │   ├── TokenNotFoundError
   │   └── InvalidTokenError
   └── RepositoryError

Base Exception
--------------

.. autoclass:: hf_lifecycle.exceptions.HfLifecycleError
   :members:
   :show-inheritance:

Authentication Exceptions
-------------------------

.. autoclass:: hf_lifecycle.exceptions.AuthenticationError
   :members:
   :show-inheritance:

.. autoclass:: hf_lifecycle.exceptions.TokenNotFoundError
   :members:
   :show-inheritance:

.. autoclass:: hf_lifecycle.exceptions.InvalidTokenError
   :members:
   :show-inheritance:

Repository Exceptions
---------------------

.. autoclass:: hf_lifecycle.exceptions.RepositoryError
   :members:
   :show-inheritance:
