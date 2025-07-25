API Reference
=============

This page contains the complete API documentation for VCS Metrics, automatically generated from docstrings.

Core Functions
--------------

Main Scoring Function
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.compute_vcs_score

.. code-block:: python

   # Access this documentation interactively in your notebook or Python session
   help(vcs.compute_vcs_score)
   
   # Or use ? in Jupyter/Colab for quick reference
   vcs.compute_vcs_score?

Visualization Functions
-----------------------

Configuration and Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vcs.visualize_config

.. code-block:: python

   # Access this documentation interactively in your notebook or Python session
   help(vcs.visualize_config)
   
   # Or use ? in Jupyter/Colab for quick reference
   vcs.visualize_config?

.. autofunction:: vcs.visualize_metrics_summary

.. code-block:: python

   # Access this documentation interactively in your notebook or Python session
   help(vcs.visualize_metrics_summary)
   
   # Or use ? in Jupyter/Colab for quick reference
   vcs.visualize_metrics_summary?

Text Analysis
~~~~~~~~~~~~~

.. autofunction:: vcs.visualize_text_chunks

.. code-block:: python

   # Access this documentation interactively in your notebook or Python session
   help(vcs.visualize_text_chunks)
   
   # Or use ? in Jupyter/Colab for quick reference
   vcs.visualize_text_chunks?

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

.. code-block:: python

   # Access this documentation interactively in your notebook or Python session
   help(vcs.create_vcs_pdf_report)
   
   # Or use ? in Jupyter/Colab for quick reference
   vcs.create_vcs_pdf_report?

Configuration Constants
-----------------------

Default parameter values available for import:

.. py:data:: vcs.DEFAULT_CONTEXT_CUTOFF_VALUE

   **Default:** ``0.6`` | **Type:** ``float`` | **Range:** ``0.0 - 1.0``
   
   Context cutoff threshold for best match finding.

.. py:data:: vcs.DEFAULT_CONTEXT_WINDOW_CONTROL

   **Default:** ``4.0`` | **Type:** ``float`` | **Range:** ``1.0 - ∞``
   
   Context window size control parameter.

.. py:data:: vcs.DEFAULT_LCT

   **Default:** ``0`` | **Type:** ``int`` | **Range:** ``0 - ∞``
   
   Local Chronology Tolerance for narrative ordering flexibility.

.. py:data:: vcs.DEFAULT_CHUNK_SIZE

   **Default:** ``1`` | **Type:** ``int`` | **Range:** ``1 - ∞``
   
   Number of text segments grouped into analysis chunks.

Package Information
-------------------

Access package metadata programmatically:

.. code-block:: python

   import vcs
   
   # Print package information
   print(f"VCS Metrics v{vcs.__version__}")
   print(f"Author: {vcs.__author__}")
   print(f"Contact: {vcs.__email__}")


See Also
--------

- :doc:`getting_started` - Installation and basic setup
- `GitHub Repository <https://github.com/Multimodal-Intelligence-Lab/Video-Comprehension-Score>`_ - Source code and issue tracking