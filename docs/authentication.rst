Authentication
==============

The ``AuthManager`` class handles authentication with the HuggingFace Hub, supporting multiple token sources.

Token Sources
-------------

The authentication system checks for tokens in the following order:

1. **Direct Input**: Token passed to ``login()`` method
2. **Environment Variable**: ``HF_TOKEN`` environment variable
3. **CLI Cache**: Token saved by ``huggingface-cli login``

Basic Authentication
--------------------

Using Environment Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export HF_TOKEN=hf_your_token_here

.. code-block:: python

   from hf_lifecycle.auth import AuthManager

   auth = AuthManager()
   token = auth.get_token()  # Automatically retrieves from env

Direct Login
~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.auth import AuthManager

   auth = AuthManager()
   auth.login(token="hf_your_token_here", write_to_disk=True)

The ``write_to_disk=True`` parameter saves the token to the local cache.

Token Validation
----------------

Tokens are automatically validated when you call ``login()``:

.. code-block:: python

   from hf_lifecycle.auth import AuthManager
   from hf_lifecycle.exceptions import InvalidTokenError

   auth = AuthManager()
   
   try:
       auth.login(token="invalid_token")
   except InvalidTokenError as e:
       print(f"Token validation failed: {e}")

Logout
------

Clear your credentials:

.. code-block:: python

   auth.logout()

Profile Management
------------------

Switch between different accounts (placeholder for future implementation):

.. code-block:: python

   auth.set_profile("work")
   auth.set_profile("personal")

Error Handling
--------------

The authentication module raises specific exceptions:

- ``TokenNotFoundError``: No token found in any source
- ``InvalidTokenError``: Token validation failed
- ``AuthenticationError``: General authentication error

Example:

.. code-block:: python

   from hf_lifecycle.auth import AuthManager
   from hf_lifecycle.exceptions import TokenNotFoundError

   auth = AuthManager()
   
   try:
       auth.login()  # No token provided
   except TokenNotFoundError:
       print("Please provide a token or set HF_TOKEN environment variable")
