Model Registry
==============

The Model Registry simplifies sharing custom models on HuggingFace Hub with automatic model card generation.

Registering a Model
-------------------

Basic Registration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hf_lifecycle.model_registry import ModelRegistry
   
   registry = ModelRegistry(repo_manager)
   
   # Register your custom model
   url = registry.register_model(
       model=my_model,
       repo_id="username/my-awesome-model",
       description="A custom model for sentiment analysis",
       private=False
   )
   
   print(f"Model registered at: {url}")

With Metrics and Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   url = registry.register_model(
       model=my_model,
       repo_id="username/my-model",
       description="Fine-tuned BERT for NER",
       metrics={"f1": 0.95, "accuracy": 0.93},
       datasets=["conll2003", "ontonotes"],
       tags=["ner", "bert", "token-classification"],
   )

With Tokenizer and Config
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import AutoTokenizer, AutoConfig
   
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   config = AutoConfig.from_pretrained("bert-base-uncased")
   
   registry.register_model(
       model=my_model,
       repo_id="username/my-model",
       tokenizer=tokenizer,
       config=config,
       description="Custom BERT variant"
   )

Model Card Generation
---------------------

Automatic Model Cards
~~~~~~~~~~~~~~~~~~~~~

The registry automatically generates professional model cards with:

- YAML frontmatter (license, tags, datasets)
- Model description
- Architecture information
- Training metrics in tables
- Usage examples
- Timestamps

Custom Model Card
~~~~~~~~~~~~~~~~~

.. code-block:: python

   model_card = registry.generate_model_card(
       model_name="Sentiment Analyzer",
       description="Fine-tuned on movie reviews",
       architecture="distilbert",
       datasets=["imdb", "sst2"],
       metrics={"accuracy": 0.94, "f1": 0.93},
       tags=["sentiment-analysis", "distilbert"],
       license="mit",
       language=["en"]
   )
   
   print(model_card)

Loading Models
--------------

Load from Hub
~~~~~~~~~~~~~

.. code-block:: python

   # Load using AutoModel
   model = registry.load_model("username/my-model")
   
   # Load with specific model class
   from transformers import BertForSequenceClassification
   
   model = registry.load_model(
       "username/my-model",
       model_class=BertForSequenceClassification
   )

Load Specific Revision
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load from a specific branch or tag
   model = registry.load_model(
       "username/my-model",
       revision="v1.0"
   )

Complete Example
----------------

.. code-block:: python

   from hf_lifecycle.auth import AuthManager
   from hf_lifecycle.repo import RepoManager
   from hf_lifecycle.model_registry import ModelRegistry
   import torch.nn as nn
   
   # Setup
   auth = AuthManager()
   auth.login(token="hf_...")
   
   repo_mgr = RepoManager(auth)
   registry = ModelRegistry(repo_mgr)
   
   # Your custom model
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Linear(768, 2)
       
       def forward(self, x):
           return self.fc(x)
   
   model = MyModel()
   
   # Register to Hub
   url = registry.register_model(
       model=model,
       repo_id="myusername/custom-classifier",
       description="A simple binary classifier",
       metrics={"accuracy": 0.89},
       datasets=["custom-dataset"],
       tags=["classification", "pytorch"],
       private=False
   )
   
   print(f"✅ Model published: {url}")
