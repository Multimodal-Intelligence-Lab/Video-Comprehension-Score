Usage Guide
===========

This guide provides comprehensive examples of how to use VCS Metrics for different scenarios and text types.

Basic Usage
-----------

Simple Text Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from sentence_transformers import SentenceTransformer
   import nltk
   from vcs import compute_vcs_score

   # Setup (run once)
   nltk.download('punkt')
   model = SentenceTransformer('all-MiniLM-L6-v2')

   # Define your functions
   def segment_text(text):
       return nltk.sent_tokenize(text)

   def embed_text(texts):
       return torch.tensor(model.encode(texts))

   # Compare texts
   reference = "The cat sat on the mat. It was very comfortable."
   generated = "A cat was sitting on a mat and seemed comfortable."

   result = compute_vcs_score(
       reference_text=reference,
       generated_text=generated,
       segmenter_fn=segment_text,
       embedding_fn_las=embed_text
   )

   print(f"VCS Score: {result['VCS']:.4f}")

Getting All Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get detailed breakdown of all metrics
   result = compute_vcs_score(
       reference_text=reference,
       generated_text=generated,
       segmenter_fn=segment_text,
       embedding_fn_las=embed_text,
       return_all_metrics=True
   )

   # Print all metrics
   for metric, value in result.items():
       print(f"{metric}: {value:.4f}")

Advanced Configuration
---------------------

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fine-tune the analysis parameters
   result = compute_vcs_score(
       reference_text=reference,
       generated_text=generated,
       segmenter_fn=segment_text,
       embedding_fn_las=embed_text,
       chunk_size=2,                    # Group sentences in pairs
       context_cutoff_value=0.7,        # More restrictive matching
       context_window_control=3.0,      # Different context window size
       lct=1,                          # Allow some narrative reordering
       return_all_metrics=True
   )

Different Embedding Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use different models for global vs local analysis
   global_model = SentenceTransformer('all-mpnet-base-v2')  # High quality
   local_model = SentenceTransformer('all-MiniLM-L6-v2')   # Fast

   def global_embeddings(texts):
       return torch.tensor(global_model.encode(texts))

   def local_embeddings(texts):
       return torch.tensor(local_model.encode(texts))

   result = compute_vcs_score(
       reference_text=reference,
       generated_text=generated,
       segmenter_fn=segment_text,
       embedding_fn_las=local_embeddings,
       embedding_fn_gas=global_embeddings
   )

Visualization Examples
----------------------

Basic Visualizations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vcs import (
       visualize_config, visualize_metrics_summary, 
       visualize_similarity_matrix, visualize_las
   )

   # Get internals for visualization
   result = compute_vcs_score(
       reference_text=reference,
       generated_text=generated,
       segmenter_fn=segment_text,
       embedding_fn_las=embed_text,
       return_internals=True,
       return_all_metrics=True
   )

   internals = result['internals']

   # Show configuration
   config_fig = visualize_config(internals)
   config_fig.show()

   # Overview of all metrics
   summary_fig = visualize_metrics_summary(internals)
   summary_fig.show()

   # Similarity matrix heatmap
   sim_fig = visualize_similarity_matrix(internals)
   sim_fig.show()

   # Local Alignment Score details
   las_fig = visualize_las(internals)
   las_fig.show()

Advanced Visualizations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vcs import (
       visualize_text_chunks, visualize_mapping_windows,
       visualize_distance_nas, visualize_line_nas,
       visualize_best_match
   )

   # Text chunks (may return multiple pages)
   chunk_figs = visualize_text_chunks(internals)
   for i, fig in enumerate(chunk_figs['reference_chunks']):
       fig.suptitle(f'Reference Chunks - Page {i+1}')
       fig.show()

   # Mapping windows showing alignment constraints
   windows_fig = visualize_mapping_windows(internals)
   windows_fig.show()

   # Distance-based narrative analysis
   distance_fig = visualize_distance_nas(internals)
   distance_fig.show()

   # Line-based narrative analysis
   line_fig = visualize_line_nas(internals)
   line_fig.show()

   # Best match analysis (returns multiple figures)
   match_figs = visualize_best_match(internals)
   match_figs['precision_details'].show()
   match_figs['recall_details'].show()
   match_figs['summary_table'].show()

Creating PDF Reports
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vcs import create_vcs_pdf_report

   # Generate comprehensive PDF report
   create_vcs_pdf_report(
       internals=internals,
       output_file="complete_analysis.pdf",
       metrics_to_include="all"
   )

   # Generate focused report with specific metrics
   create_vcs_pdf_report(
       internals=internals,
       output_file="summary_report.pdf",
       metrics_to_include=["Config", "Overview", "LAS", "NAS Distance"]
   )

Domain-Specific Examples
-----------------------

News Article Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For news articles, use strict chronological order
   def analyze_news_articles(reference_article, generated_summary):
       return compute_vcs_score(
           reference_text=reference_article,
           generated_text=generated_summary,
           segmenter_fn=segment_text,
           embedding_fn_las=embed_text,
           chunk_size=1,        # Sentence-level analysis
           lct=0,              # Strict chronological order
           return_all_metrics=True,
           return_internals=True
       )

   news_result = analyze_news_articles(original_article, ai_summary)

Dialog/Conversation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom segmentation for dialog
   def segment_dialog(text):
       lines = text.split('\n')
       return [line.strip() for line in lines if line.strip()]

   def analyze_dialog(reference_dialog, generated_dialog):
       return compute_vcs_score(
           reference_text=reference_dialog,
           generated_text=generated_dialog,
           segmenter_fn=segment_dialog,
           embedding_fn_las=embed_text,
           chunk_size=1,
           lct=1,              # Allow some reordering in conversations
           return_all_metrics=True
       )

   # Example dialog texts
   ref_dialog = """Speaker A: Hello, how are you?
   Speaker B: I'm doing well, thanks!
   Speaker A: What are your plans for today?
   Speaker B: I'm going to the library."""

   gen_dialog = """A: Hi, how are you doing?
   B: I'm good, thank you!
   A: Any plans for today?
   B: I'll visit the library."""

   dialog_result = analyze_dialog(ref_dialog, gen_dialog)

Creative Writing Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For creative writing, allow more narrative flexibility
   def analyze_creative_writing(reference_story, generated_story):
       return compute_vcs_score(
           reference_text=reference_story,
           generated_text=generated_story,
           segmenter_fn=segment_text,
           embedding_fn_las=embed_text,
           chunk_size=2,        # Group sentences for narrative flow
           lct=2,              # Allow creative reordering
           context_cutoff_value=0.5,  # More permissive matching
           return_all_metrics=True,
           return_internals=True
       )

Batch Processing
---------------

Processing Multiple Text Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from tqdm import tqdm

   def batch_vcs_analysis(text_pairs, output_file="results.csv"):
       """Process multiple text pairs and save results."""
       results = []
       
       for i, (ref_text, gen_text) in enumerate(tqdm(text_pairs, desc="Processing")):
           try:
               result = compute_vcs_score(
                   reference_text=ref_text,
                   generated_text=gen_text,
                   segmenter_fn=segment_text,
                   embedding_fn_las=embed_text,
                   return_all_metrics=True
               )
               
               # Extract key metrics
               results.append({
                   'pair_id': i,
                   'vcs_score': result['VCS'],
                   'gas_score': result['GAS'],
                   'las_score': result['LAS'],
                   'nas_score': result['NAS'],
                   'ref_length': len(segment_text(ref_text)),
                   'gen_length': len(segment_text(gen_text))
               })
               
           except Exception as e:
               print(f"Error processing pair {i}: {e}")
               results.append({
                   'pair_id': i,
                   'vcs_score': None,
                   'error': str(e)
               })
       
       # Save results
       df = pd.DataFrame(results)
       df.to_csv(output_file, index=False)
       return df

   # Example usage
   text_pairs = [
       ("Reference 1", "Generated 1"),
       ("Reference 2", "Generated 2"),
       # ... more pairs
   ]
   
   results_df = batch_vcs_analysis(text_pairs)
   print(results_df.describe())

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import itertools

   def parameter_grid_search(ref_text, gen_text, param_grid):
       """Test different parameter combinations."""
       results = []
       
       # Generate all combinations
       keys = param_grid.keys()
       values = param_grid.values()
       
       for combination in itertools.product(*values):
           params = dict(zip(keys, combination))
           
           try:
               result = compute_vcs_score(
                   reference_text=ref_text,
                   generated_text=gen_text,
                   segmenter_fn=segment_text,
                   embedding_fn_las=embed_text,
                   **params,
                   return_all_metrics=True
               )
               
               results.append({
                   **params,
                   'vcs_score': result['VCS'],
                   'gas_score': result['GAS'],
                   'las_score': result['LAS'],
                   'nas_score': result['NAS']
               })
               
           except Exception as e:
               print(f"Error with params {params}: {e}")
       
       return pd.DataFrame(results)

   # Define parameter grid
   param_grid = {
       'chunk_size': [1, 2, 3],
       'lct': [0, 1, 2],
       'context_cutoff_value': [0.5, 0.6, 0.7]
   }

   # Run grid search
   grid_results = parameter_grid_search(reference, generated, param_grid)
   
   # Find best parameters
   best_params = grid_results.loc[grid_results['vcs_score'].idxmax()]
   print(f"Best VCS score: {best_params['vcs_score']:.4f}")
   print(f"Best parameters: {best_params.to_dict()}")

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   def memory_efficient_analysis(ref_text, gen_text):
       """Analyze text with memory optimization."""
       result = compute_vcs_score(
           reference_text=ref_text,
           generated_text=gen_text,
           segmenter_fn=segment_text,
           embedding_fn_las=embed_text,
           return_internals=True,
           return_all_metrics=True
       )
       
       # Generate and save visualizations without keeping in memory
       internals = result['internals']
       
       # Save key visualizations
       for viz_name, viz_func in [
           ('config', visualize_config),
           ('summary', visualize_metrics_summary),
           ('similarity', visualize_similarity_matrix),
           ('las', visualize_las)
       ]:
           fig = viz_func(internals)
           fig.savefig(f'{viz_name}.png', dpi=300, bbox_inches='tight')
           plt.close(fig)  # Free memory
       
       # Generate PDF report
       create_vcs_pdf_report(
           internals=internals,
           output_file="analysis_report.pdf",
           metrics_to_include="all"
       )
       
       # Return just the scores (not internals) to save memory
       return {k: v for k, v in result.items() if k != 'internals'}

Large Text Handling
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_long_text(ref_text, gen_text, max_segments=50):
       """Handle very long texts by chunking."""
       ref_segments = segment_text(ref_text)
       gen_segments = segment_text(gen_text)
       
       # If texts are too long, analyze in chunks
       if len(ref_segments) > max_segments or len(gen_segments) > max_segments:
           print(f"Text too long ({len(ref_segments)}, {len(gen_segments)} segments). Chunking...")
           
           # Simple chunking strategy
           ref_chunks = [' '.join(ref_segments[i:i+max_segments]) 
                        for i in range(0, len(ref_segments), max_segments)]
           gen_chunks = [' '.join(gen_segments[i:i+max_segments])
                        for i in range(0, len(gen_segments), max_segments)]
           
           # Analyze each chunk pair
           chunk_results = []
           for ref_chunk, gen_chunk in zip(ref_chunks, gen_chunks):
               result = compute_vcs_score(
                   reference_text=ref_chunk,
                   generated_text=gen_chunk,
                   segmenter_fn=segment_text,
                   embedding_fn_las=embed_text,
                   return_all_metrics=True
               )
               chunk_results.append(result)
           
           # Aggregate results (simple average)
           avg_result = {}
           for key in chunk_results[0].keys():
               avg_result[key] = sum(r[key] for r in chunk_results) / len(chunk_results)
           
           return avg_result
       else:
           # Normal analysis for reasonable-sized texts
           return compute_vcs_score(
               reference_text=ref_text,
               generated_text=gen_text,
               segmenter_fn=segment_text,
               embedding_fn_las=embed_text,
               return_all_metrics=True
           )

Error Handling and Debugging
----------------------------

Robust Analysis Function
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_vcs_analysis(ref_text, gen_text, **kwargs):
       """VCS analysis with comprehensive error handling."""
       try:
           # Validate inputs
           if not ref_text.strip():
               raise ValueError("Reference text is empty")
           if not gen_text.strip():
               raise ValueError("Generated text is empty")
           
           # Attempt analysis
           result = compute_vcs_score(
               reference_text=ref_text,
               generated_text=gen_text,
               segmenter_fn=segment_text,
               embedding_fn_las=embed_text,
               return_all_metrics=True,
               **kwargs
           )
           
           # Validate results
           if result['VCS'] < 0 or result['VCS'] > 1:
               print(f"Warning: VCS score {result['VCS']:.4f} is outside expected range [0, 1]")
           
           return {
               'success': True,
               'result': result,
               'error': None
           }
           
       except Exception as e:
           return {
               'success': False,
               'result': None,
               'error': str(e),
               'error_type': type(e).__name__
           }

   # Usage with error handling
   analysis = robust_vcs_analysis(reference, generated)

   if analysis['success']:
       result = analysis['result']
       print(f"VCS Score: {result['VCS']:.4f}")
   else:
       print(f"Analysis failed: {analysis['error']}")

Best Practices
--------------

Choosing Segmentation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different segmentation strategies
   strategies = {
       'sentences': lambda t: nltk.sent_tokenize(t),
       'simple_split': lambda t: [s.strip() for s in t.split('.') if s.strip()],
       'paragraphs': lambda t: [p.strip() for p in t.split('\n\n') if p.strip()],
       'fixed_length': lambda t: [t[i:i+100] for i in range(0, len(t), 100)]
   }

   for name, segmenter in strategies.items():
       try:
           result = compute_vcs_score(
               reference_text=reference,
               generated_text=generated,
               segmenter_fn=segmenter,
               embedding_fn_las=embed_text
           )
           print(f"{name}: VCS = {result['VCS']:.4f}")
       except Exception as e:
           print(f"{name}: Error - {e}")

Model Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare different embedding models
   models = {
       'all-MiniLM-L6-v2': 'Fast, good quality',
       'all-mpnet-base-v2': 'Slower, higher quality',
       'all-distilroberta-v1': 'Balanced speed/quality'
   }

   for model_name, description in models.items():
       print(f"\nTesting {model_name} ({description})")
       
       model = SentenceTransformer(model_name)
       embedder = lambda texts: torch.tensor(model.encode(texts))
       
       result = compute_vcs_score(
           reference_text=reference,
           generated_text=generated,
           segmenter_fn=segment_text,
           embedding_fn_las=embedder
       )
       
       print(f"VCS Score: {result['VCS']:.4f}")

Next Steps
----------

- Explore the :doc:`api` reference for detailed function documentation
- Check out the `GitHub repository <https://github.com/yourusername/vcs-metrics>`_ for examples and updates
- Join the discussion on `GitHub Discussions <https://github.com/yourusername/vcs-metrics/discussions>`_