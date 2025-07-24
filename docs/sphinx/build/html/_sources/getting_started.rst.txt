Getting Started
===============

This guide will help you install VCS Metrics and understand the basic requirements for using the library.

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~~~

Install VCS Metrics from PyPI:

.. code-block:: bash

   pip install vcs-metrics

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/vcs-metrics.git
   cd vcs-metrics

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

PyTorch Installation
~~~~~~~~~~~~~~~~~~~~

VCS Metrics requires PyTorch >= 1.9.0 but doesn't install it automatically to avoid conflicts with existing installations. 

.. note::
   **PyTorch Requirements**: VCS Metrics requires PyTorch version 1.9.0 or higher. Please visit the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ to download the appropriate version for your system configuration (CPU/GPU, operating system, etc.). In Google Colab, PyTorch is pre-installed, so no additional installation is needed.

Requirements
------------

- Python 3.8+
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- PyTorch >= 1.9.0 (install separately)

Core Function Requirements
--------------------------

VCS Metrics requires two types of functions that you need to provide:

1. **Segmenter Function**: Splits text into meaningful units
2. **Embedding Function**: Converts text segments into numerical vectors

Segmenter Function
~~~~~~~~~~~~~~~~~~

The segmenter function takes a string and returns a list of strings (segments).

**Function Signature:**

.. code-block:: python

   def segmenter_function(text: str) -> List[str]:
       """
       Split text into segments for analysis.
       
       Args:
           text: Input text to segment
           
       Returns:
           List of text segments
       """
       pass

**Available Libraries and Tools:**

You can use various libraries and models to build your segmenter function:

* **Traditional Libraries**: NLTK, spaCy for sentence and clause segmentation
* **Modern Models**: `Segment Any Text (SAT) <https://github.com/segment-any-text/wtpsplit>`_ for state-of-the-art text segmentation
* **Research**: We recommend researching current state-of-the-art segmentation technologies, as poor segmentation can significantly affect VCS performance

**Author Recommendation (2025):**

.. warning::
   **Technology Evolution**: This recommendation is current as of 2025. As better segmentation models emerge, this recommendation may become outdated. Always research the latest state-of-the-art options.

For 2025, we recommend using **Segment Any Text (SAT)** for optimal segmentation performance:

.. code-block:: python

   import re
   import string
   import contractions
   # Note: You need to download and initialize SAT model first
   # from wtpsplit import SaT
   # sat_adapted = SaT("sat-12l-sm")  # or appropriate model variant
   
   # Define punctuation set (excluding apostrophes for contractions)
   punctuations = set(string.punctuation) - {"'"}
   
   def sat_segmenter(text: str) -> list[str]:
       """
       Advanced text segmenter using Segment Any Text (SAT) model.
       
       This function:
       1. Expands contractions (can't -> cannot)
       2. Removes punctuation (except apostrophes)
       3. Fixes spacing around remaining punctuation
       4. Uses SAT model for intelligent segmentation
       """
       # Expand contractions for better processing
       text = contractions.fix(text)
       
       def remove_punctuation(text_str: str) -> str:
           """Remove punctuation except apostrophes."""
           return text_str.translate(str.maketrans('', '', ''.join(punctuations)))
       
       def fix_punctuation_spacing(text_str: str) -> str:
           """Add space after sentence-ending punctuation if missing."""
           return re.sub(r'([.!?])(?=[^\s])', r'\1 ', text_str)
       
       # Clean and prepare text
       text = remove_punctuation(text)
       text = fix_punctuation_spacing(text)
       
       # Use SAT model for segmentation
       sentences = sat_adapted.split(text)
       
       # Clean and filter segments
       sentences = [s.strip() for s in sentences if s.strip()]
       
       return sentences

Embedding Function
~~~~~~~~~~~~~~~~~~

The embedding function takes a list of strings and returns a PyTorch tensor with embeddings.

**Function Signature:**

.. code-block:: python

   def embedding_function(texts: List[str]) -> torch.Tensor:
       """
       Convert text segments to embeddings.
       
       Args:
           texts: List of text segments to embed
           
       Returns:
           PyTorch tensor of shape (len(texts), embedding_dim)
       """
       pass

**Finding SOTA Embedding Models:**

Visit the `Massive Text Embedding Benchmark (MTEB) <https://huggingface.co/spaces/mteb/leaderboard>`_ to find state-of-the-art embedding models. You can choose from:

* **English Models**: For English-only text analysis
* **Multilingual Models**: For multi-language support
* **Different Model Sizes**: From lightweight to high-performance variants

**Author Recommendation (2025):**

.. warning::
   **Technology Evolution**: This recommendation is current as of 2025. As better embedding models emerge, this recommendation may become outdated. Always check MTEB leaderboard for the latest best-performing models.

For 2025, we recommend **nv-embed-v2** for optimal embedding performance:

.. code-block:: python

   import torch
   import torch.nn.functional as F
   # Note: You need to download and initialize nv-embed-v2 model first
   # model_nv = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
   
   def nv_embed_embedding_fn(texts: list[str], instruction: str = "", model=None,
                             batch_size: int = 8, max_length: int = 32768) -> torch.Tensor:
       """
       High-performance embedding function using nv-embed-v2.
       
       Args:
           texts: List of text segments to embed
           instruction: Optional instruction for the embedding model
           model: Pre-initialized nv-embed-v2 model
           batch_size: Number of texts to process at once
           max_length: Maximum token length per text
       """
       if model is None:
           model = model_nv  # Use pre-initialized global model
       
       device = next(model.parameters()).device
       all_embs = []
       
       # Process in batches to manage memory
       for i in range(0, len(texts), batch_size):
           batch = texts[i: i + batch_size]
           
           # Generate embeddings
           emb_np = model.encode(batch, instruction=instruction, max_length=max_length)
           emb = torch.tensor(emb_np, device=device, dtype=torch.float)
           
           # Normalize embeddings for cosine similarity
           emb = F.normalize(emb, p=2, dim=1)
           all_embs.append(emb)
       
       return torch.cat(all_embs, dim=0)

**Important Setup Requirements:**

.. note::
   **Model Initialization**: Both SAT and nv-embed-v2 require you to download and initialize the models before creating your segmenter or embedding functions. Use SOTA models for best VCS results.

.. warning::
   **GPU Requirements**: SAT and nv-embed-v2 require GPU access for optimal performance. For CPU-only testing, consider using smaller models from MTEB leaderboard for embeddings and traditional libraries like NLTK or spaCy for segmentation.

Quick Start Example
-------------------

Here's a complete working example using Sentence Transformers:

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt

   try:
       from vcs import compute_vcs_score, visualize_metrics_summary
       print("✅ Main function imported successfully!")
   except ImportError as e:
       print(f"❌ Import failed: {e}")

   # Example texts
   reference_text = """
   The quick brown fox jumps over the lazy dog.
   It was a beautiful sunny day in the forest.
   The fox was looking for food for its family.
   """

   generated_text = """
   A brown fox jumped over a sleeping dog.
   The weather was nice and sunny in the woods.
   The fox needed to find food for its cubs.
   """

   # Load a pre-trained sentence transformer model
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')

   def sbert_embedding_function(texts):
       """Sentence-BERT embedding function."""
       embeddings = model.encode(texts)
       return torch.tensor(embeddings, dtype=torch.float32)

   def simple_segmenter(text):
       """Simple sentence segmenter."""
       return [s.strip() for s in text.split('.') if s.strip()]

   # Test with better embeddings
   print("🧠 Testing with Sentence-BERT embeddings...")
   result_sbert = compute_vcs_score(
       reference_text=reference_text,
       generated_text=generated_text,
       segmenter_fn=simple_segmenter,
       embedding_fn_las=sbert_embedding_function,
       embedding_fn_gas=sbert_embedding_function,
       return_all_metrics=True,
       return_internals=True
   )

   print("🎯 VCS Results with Sentence-BERT:")
   print(f"VCS Score: {result_sbert['VCS']:.4f}")
   print(f"GAS Score: {result_sbert['GAS']:.4f}")
   print(f"LAS Score: {result_sbert['LAS']:.4f}")
   print(f"NAS Score: {result_sbert['NAS']:.4f}")


   if 'internals' in result_sbert:
       internals = result_sbert['internals']
       visualize_metrics_summary(internals)

.. note::
   **Scale Consideration**: This example uses a small caption to illustrate the concept, but VCS is designed for analyzing really long captions and should be used for them. There is no size limit to caption length - any large generated caption length can be processed.

Configuration Parameters
------------------------

VCS Metrics provides several configuration parameters to control the granularity and strictness of the comparison:

.. code-block:: python

   from vcs import (
       DEFAULT_CONTEXT_CUTOFF_VALUE,    # 0.6
       DEFAULT_CONTEXT_WINDOW_CONTROL,  # 4.0
       DEFAULT_LCT,                     # 0
       DEFAULT_CHUNK_SIZE,              # 1
   )

   result = compute_vcs_score(
       reference_text=ref_text,
       generated_text=gen_text,
       segmenter_fn=segmenter,
       embedding_fn_las=embedder,
       chunk_size=2,                    # Group segments in pairs
       context_cutoff_value=0.7,        # More restrictive matching
       context_window_control=3.0,      # Larger context windows
       lct=1,                          # Allow some narrative reordering
   )

Next Steps
----------

- Read the :doc:`usage` guide for detailed examples
- Explore the :doc:`api` reference for all available functions
- Check out the visualization capabilities for analysis and reporting

Troubleshooting
---------------

**ImportError: No module named 'torch'**
   Install PyTorch separately from the official website: https://pytorch.org/get-started/locally/

**SAT or nv-embed-v2 model not found**
   Download and initialize the models first before creating your functions

**GPU memory issues**
   Try smaller batch sizes or use CPU-compatible models for testing

**Poor VCS scores**
   Experiment with different segmentation strategies and embedding models from MTEB leaderboard