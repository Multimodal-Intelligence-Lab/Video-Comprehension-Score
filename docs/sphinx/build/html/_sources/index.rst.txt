VCS Metrics Documentation
=========================

.. raw:: html

   <div style="background: linear-gradient(135deg, #0d9488, #0f766e); color: white; padding: 2rem; text-align: center; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);">
       <h1 style="color: white; border: none; margin: 0; font-size: 2.5rem; font-weight: 700;">VCS Metrics</h1>
       <p style="color: #ccfbf1; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Video Comprehension Score - A comprehensive metric for evaluating narrative similarity</p>
   </div>

VCS Metrics is a Python library that provides a comprehensive approach to measuring narrative similarity between reference and generated text. Originally developed to evaluate the video comprehension ability of large video language models by comparing their long video descriptions to human-written captions, VCS can also be used for document-level similarity analysis or comparing similarity between two long stories and other narrative content.

VCS focuses on whether the overall story and narrative structure matches rather than worrying about every small detail. It achieves this by checking the global, local, and chronological semantic alignment between two given long paragraphs. There is no size limit in terms of paragraphs that can be processed.

To understand how VCS works in detail, please read our `research paper <#>`_ or explore our interactive `playground <#>`_.

.. image:: https://img.shields.io/pypi/v/vcs-metrics.svg
   :target: https://pypi.org/project/vcs-metrics/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

🎯 **Comprehensive Scoring**: Evaluates narrative similarity through Global Alignment Score (GAS), Local Alignment Score (LAS), and chronological Narrative Alignment Score (NAS)

📊 **Rich Visualizations**: Generate detailed plots, heatmaps, and PDF reports for analysis

🔧 **Flexible Configuration**: VCS is configurable to perform less granular or more granular comparisons and can be configured to control the strictness of comparison, from strict to less strict matching

📈 **Detailed Analytics**: Access to internal calculations for deep analysis and debugging

🎨 **Publication Ready**: Professional visualizations and reports suitable for research and presentations

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

   GitHub Repository <https://github.com/yourusername/vcs-metrics>
   PyPI Package <https://pypi.org/project/vcs-metrics/>

Citation
--------

If you use VCS Metrics in your research, please cite:

.. code-block:: bibtex

   @software{vcs_metrics,
     title = {VCS Metrics: Video Comprehension Score for Text Similarity},
     author = {Harsh Dubey and Chulwoo Pack},
     year = {2024},
     url = {https://github.com/yourusername/vcs-metrics}
   }

Support
-------

- **Documentation**: You're reading it! 📖
- **Issues**: `GitHub Issues <https://github.com/yourusername/vcs-metrics/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/yourusername/vcs-metrics/discussions>`_

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/yourusername/vcs-metrics/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`