API Reference
=============

This page contains the complete API documentation for VCS Metrics, automatically generated from docstrings.

Core Functions
--------------

Main Scoring Function
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.compute_vcs_score

Visualization Functions
-----------------------

Configuration and Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.visualize_config

.. autofunction:: vcs.visualize_metrics_summary

Text Analysis
~~~~~~~~~~~~~

.. autofunction:: vcs.visualize_text_chunks

.. autofunction:: vcs.visualize_similarity_matrix

Alignment Analysis
~~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.visualize_mapping_windows

.. autofunction:: vcs.visualize_best_match

Metric Visualizations
~~~~~~~~~~~~~~~~~~~~~

Local Alignment Score (LAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vcs.visualize_las

Narrative Alignment Score (NAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vcs.visualize_distance_nas

.. autofunction:: vcs.visualize_line_nas

.. autofunction:: vcs.visualize_line_nas_precision_calculations

.. autofunction:: vcs.visualize_line_nas_recall_calculations

Window Regularization
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vcs.visualize_window_regularizer

Report Generation
~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.create_vcs_pdf_report

Configuration Constants
-----------------------

Default configuration values that can be imported and used:

.. autodata:: vcs.DEFAULT_CONTEXT_CUTOFF_VALUE
   :annotation: = 0.6

   Default context cutoff value for best match finding. Controls when context windows are applied during the best match selection process.

.. autodata:: vcs.DEFAULT_CONTEXT_WINDOW_CONTROL
   :annotation: = 4.0

   Default context window control parameter. Controls the size of context windows when they are applied during best match finding.

.. autodata:: vcs.DEFAULT_LCT
   :annotation: = 0

   Default Local Chronology Tolerance (LCT) value. Controls how much flexibility is allowed in narrative chronological ordering.

.. autodata:: vcs.DEFAULT_CHUNK_SIZE
   :annotation: = 1

   Default chunk size for grouping text segments. Determines how many consecutive text segments are grouped together to form analysis chunks.

Package Information
-------------------

.. autodata:: vcs.__version__
   :annotation: = "1.0.0"

   Current version of the VCS Metrics package.

.. autodata:: vcs.__author__
   :annotation: = "Harsh Dubey"

   Primary author of the VCS Metrics package.

.. autodata:: vcs.__email__
   :annotation: = "chulwoo.pack@sdstate.edu"

   Contact email for the VCS Metrics package.

Function Return Types
---------------------

VCS Score Result
~~~~~~~~~~~~~~~~

The :func:`vcs.compute_vcs_score` function returns a dictionary with the following structure:

**Minimal return (default):**

.. code-block:: python

   {
       'VCS': float  # Video Comprehension Score (0.0 to 1.0)
   }

**With return_all_metrics=True:**

.. code-block:: python

   {
       'VCS': float,                    # Video Comprehension Score
       'GAS': float,                    # Global Alignment Score
       'GAS-LAS-Scaled': float,         # Scaled combination of GAS and LAS
       'Precision LAS': float,          # LAS precision component
       'Recall LAS': float,             # LAS recall component
       'LAS': float,                    # Local Alignment Score (F1)
       'Precision NAS-D': float,        # Distance-based NAS precision
       'Recall NAS-D': float,           # Distance-based NAS recall
       'NAS-D': float,                  # Distance-based NAS
       'Precision NAS-L': float,        # Line-based NAS precision
       'Recall NAS-L': float,           # Line-based NAS recall
       'NAS-L': float,                  # Line-based NAS
       'NAS-F1': float,                 # Combined NAS-D and NAS-L
       'Window-Regularizer': float,     # Regularization factor
       'NAS': float,                    # Final regularized NAS
   }

**With return_internals=True:**

Adds an ``'internals'`` key containing detailed calculation data for visualization and analysis.

Visualization Return Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most visualization functions return ``matplotlib.figure.Figure`` objects:

.. code-block:: python

   import matplotlib.pyplot as plt
   from vcs import visualize_similarity_matrix

   result = compute_vcs_score(...)
   fig = visualize_similarity_matrix(result['internals'])
   fig.show()  # Display the plot
   fig.savefig('output.png')  # Save to file

Some functions return multiple figures or dictionaries of figures:

- :func:`vcs.visualize_text_chunks`: Returns ``Dict[str, List[plt.Figure]]``
- :func:`vcs.visualize_best_match`: Returns ``Dict[str, plt.Figure]``
- :func:`vcs.visualize_line_nas_precision_calculations`: Returns ``List[plt.Figure]``
- :func:`vcs.visualize_line_nas_recall_calculations`: Returns ``List[plt.Figure]``

Error Handling
--------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. py:exception:: ValueError

   Raised when:
   - Embedding functions are not callable
   - Both embedding functions are None
   - Unknown metric names in visualization functions

.. py:exception:: TypeError

   Raised when:
   - Segmenter function is not callable
   - Segmenter function doesn't return a list of strings

Examples of Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vcs import compute_vcs_score

   try:
       result = compute_vcs_score(
           reference_text="",  # Empty text might cause issues
           generated_text="Some text",
           segmenter_fn=None,  # This will raise TypeError
           embedding_fn_las=lambda x: "not a tensor"  # This will cause issues
       )
   except ValueError as e:
       print(f"Configuration error: {e}")
   except TypeError as e:
       print(f"Function type error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Type Hints
----------

For type checking and IDE support, the package uses the following types:

.. code-block:: python

   from typing import List, Callable, Dict, Any, Union
   import torch

   # Function signatures
   SegmenterFunction = Callable[[str], List[str]]
   EmbeddingFunction = Callable[[List[str]], torch.Tensor]
   VCSResult = Dict[str, Any]

See Also
--------

- :doc:`getting_started` - Installation and basic setup
- :doc:`usage` - Detailed usage examples and tutorials
- `GitHub Repository <https://github.com/yourusername/vcs-metrics>`_ - Source code and issue tracking