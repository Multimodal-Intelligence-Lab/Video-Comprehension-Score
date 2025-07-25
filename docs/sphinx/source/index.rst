VCS Metrics Documentation
=========================

.. raw:: html

   <div style="background: linear-gradient(135deg, #0d9488, #0f766e); color: white; padding: 2rem; text-align: center; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);">
       <h1 style="color: white; border: none; margin: 0; font-size: 2.5rem; font-weight: 700;">VCS Metrics</h1>
       <p style="color: #ccfbf1; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Video Comprehension Score - A comprehensive metric for evaluating narrative similarity</p>
   </div>

Recent advances in Large Video Language Models (LVLMs) have significantly enhanced automated video understanding, enabling detailed, long-form narratives of complex video content. However, accurately evaluating whether these models genuinely comprehend the video's narrative—its events, entities, interactions, and chronological coherence—remains challenging.

**Why Existing Metrics Fall Short:**

- **N-gram Metrics (e.g., BLEU, ROUGE, CIDEr)**: Primarily measure lexical overlap, penalizing valid linguistic variations and inadequately evaluating narrative chronology.
- **Embedding-based Metrics (e.g., BERTScore, SBERT)**: Improve semantic sensitivity but struggle with extended context, detailed content alignment, and narrative sequencing.
- **LLM-based Evaluations**: Often inconsistent, lacking clear criteria for narrative structure and chronology assessments.

Moreover, traditional benchmarks largely rely on question-answering tasks, which only test isolated events or entities rather than holistic video comprehension. A model answering specific questions correctly does not necessarily demonstrate understanding of the overall narrative or the intricate interplay of events.

**Introducing VCS (Video Comprehension Score):**

VCS is a Python library specifically designed to overcome these challenges by evaluating narrative comprehension through direct comparison of extensive, detailed video descriptions generated by LVLMs against human-written references. Unlike traditional metrics, VCS assesses whether models capture the overall narrative structure, event sequencing, and thematic coherence, not just lexical or isolated semantic matches.

**Core Components of VCS:**

- 🌍 **Global Alignment Score (GAS)**: Captures overall thematic alignment, tolerating stylistic variations without penalizing valid linguistic differences.
- 🎯 **Local Alignment Score (LAS)**: Checks detailed semantic correspondence at a chunk-level, allowing minor descriptive variations while penalizing significant inaccuracies or omissions.
- 📖 **Narrative Alignment Score (NAS)**: Evaluates chronological consistency, balancing the need for both strict event sequencing and permissible narrative flexibility.

Initially developed for evaluating video comprehension by comparing generated and human-written video narratives, VCS is versatile enough for broader applications, including document-level narrative comparisons, analysis of extensive narrative content, and various other narrative similarity tasks.

To understand how VCS works in detail, please read our research paper or explore our interactive playground.

.. raw:: html

   <p>
   <a href="https://arxiv.org/abs/placeholder-link" target="_blank">📄 Research Paper</a>
   ·
   <a href="https://multimodal-intelligence-lab.github.io/Video-Comprehension-Score/" target="_blank">📓 Interactive Playground</a>
   </p>

.. image:: https://img.shields.io/pypi/v/video-comprehension-score.svg
   :target: https://pypi.org/project/video-comprehension-score/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

Explore the comprehensive capabilities that make VCS a powerful narrative evaluation toolkit:

**🧮 Comprehensive Metric Suite**
   Computes VCS along with detailed breakdowns: GAS (global thematic similarity), LAS with precision/recall components, and NAS with distance-based and line-based sub-metrics. Access all internal calculations including penalty systems, mapping windows, and alignment paths.

**📊 Advanced Visualization Engine**
   11 specialized visualization functions including similarity heatmaps, alignment analysis, best-match visualizations, narrative flow diagrams, and precision/recall breakdowns. Each metric component can be visualized with publication-quality plots.

**📋 Professional PDF Reports**
   Generate comprehensive multi-page PDF reports with all metrics, visualizations, and analysis details. Supports both complete reports and customizable selective reports. Professional formatting suitable for research publications.

**⚙️ Flexible Configuration System**
   Fine-tune evaluation with configurable parameters: chunk sizes, similarity thresholds, context windows, and Local Chronology Tolerance (LCT). Supports custom segmentation and embedding functions for domain-specific applications.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

   PyPI Package <https://pypi.org/project/video-comprehension-score/>
   GitHub Repository <https://github.com/Multimodal-Intelligence-Lab/Video-Comprehension-Score>
   Report Issues <https://github.com/Multimodal-Intelligence-Lab/Video-Comprehension-Score/issues>

Citation
--------

If you use VCS Metrics in your research, please cite:

.. code-block:: bibtex

   @software{vcs_metrics,
     title = {VCS Metrics: Video Comprehension Score for Text Similarity},
     author = {Harsh Dubey and Mukhtiar Ali and Sugam Mishra and Chulwoo Pack},
     year = {2024},
     institution = {South Dakota State University},
     note = {Python package for narrative similarity evaluation}
   }


License
-------

This project is licensed under the MIT License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`