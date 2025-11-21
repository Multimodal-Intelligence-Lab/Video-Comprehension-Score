from typing import List, Callable, Dict, Any

import numpy as np
import torch

from typing import List, Callable, Dict, Any
import numpy as np
import torch

from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_Rn,
    DEFAULT_CHUNK_SIZE,
)
from ._utils import _validate_seg_embed_functions
from ._segmenting import _segment_and_chunk_texts, _build_similarity_matrix
from ._alignment_windows import _get_alignment_windows
from ._alignment_based_matching import _calculate_alignment_based_matches

from ._metrics import (
    _compute_global_sas_metrics,
    _compute_local_sas_metrics,
    _compute_sas_metrics,
    _compute_nas_metrics,
    _compute_vcs_metrics,
)

def compute_vcs_score(
    reference_text: str,
    generated_text: str,
    segmenter_fn: Callable[[str], List[str]],
    embedding_fn_global_sas: Callable[[List[str]], torch.Tensor],
    embedding_fn_local_sas: Callable[[List[str]], torch.Tensor] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_cutoff_value: float = DEFAULT_CONTEXT_CUTOFF_VALUE,
    context_window_control: float = DEFAULT_CONTEXT_WINDOW_CONTROL,
    Rn: int = DEFAULT_Rn,
    return_all_metrics: bool = False,
    return_internals: bool = False,
) -> Dict[str, Any]:
    """Compute Video Comprehension Score (VCS) between reference and generated text.

    The VCS metric combines Global Semantic Alignment Score (Global_SAS),
    Local Semantic Alignment Score (Local_SAS), and Narrative Alignment Score (NAS)
    to provide a comprehensive measure of how well a generated text preserves the
    narrative structure and semantic content of a reference text.

    **Key Metrics Computed:**

    * **Global_SAS (Global Semantic Alignment Score)**: Measures overall semantic
      similarity between the full reference and generated texts using document-level
      embeddings (Global_SAS Embedding).
    * **Local_SAS (Local Semantic Alignment Score)**: Evaluates segment-by-segment
      semantic similarity using optimal alignment between text chunks (Local_SAS Embedding).
    * **NAS (Narrative Alignment Score)**: Assesses how well the narrative flow and
      chronological structure are preserved, combining Global NAS and Local NAS measures.
    * **SAS (Semantic Alignment Score)**: Scaled combination of Global_SAS and Local_SAS.
    * **VCS (Video Comprehension Score)**: The final combined score that balances all
      three metrics to provide an overall narrative similarity assessment.
    
    Parameters
    ----------
    reference_text : str
        The reference text to compare against. This should be the "ground truth" or
        original text that serves as the comparison baseline.
    generated_text : str
        The generated text to evaluate. This is the text being assessed for how well
        it preserves the content and structure of the reference.
    segmenter_fn : callable
        Function to segment text into meaningful units for comparison. Must take a
        string as input and return a list of strings. Common choices include sentence
        segmentation, clause segmentation, or custom domain-specific segmentation.

        Example: ``lambda text: text.split('.')`` for simple sentence splitting.
    embedding_fn_global_sas : callable
        Function to compute Global_SAS Embedding (document-level embeddings) for Global_SAS
        calculation. Must take a list of strings and return a torch.Tensor of shape
        (n_items, embedding_dim) where each row is an embedding.

        Example: A function that uses sentence transformers or other semantic models.
    embedding_fn_local_sas : callable, optional
        Function to compute Local_SAS Embedding (segment-level embeddings) for Local_SAS
        calculation. If None, uses ``embedding_fn_global_sas`` for both Global_SAS and
        Local_SAS calculations. Should follow the same signature as ``embedding_fn_global_sas``.
    chunk_size : int, default=1
        Number of consecutive segments to group together for analysis. Larger values
        create bigger comparison units but may lose fine-grained alignment details.
        
        - ``chunk_size=1``: Compare individual segments (most precise)
        - ``chunk_size=2``: Compare pairs of segments  
        - ``chunk_size=3+``: Compare larger groups
    context_cutoff_value : float, default=0.6
        Threshold that controls when context windows are applied during best match 
        finding. Must be between 0 and 1. Higher values make context windows less 
        likely to be applied, leading to more restrictive matching.
    context_window_control : float, default=4.0
        Controls the size of context windows when they are applied. Larger values 
        create smaller context windows (more restrictive), while smaller values 
        create larger context windows (more permissive).
    Rn : int, default=0
        NAS Regularizer - allows flexibility in narrative ordering. 
        Higher values permit more deviation from strict chronological order:
        
        - ``Rn=0``: Strict chronological order required
        - ``Rn=1``: Small deviations allowed
        - ``Rn=2+``: More flexible chronological matching
    return_all_metrics : bool, default=False
        If True, returns all intermediate metrics (GAS, LAS, NAS components) in 
        addition to the final VCS score. Useful for detailed analysis.
    return_internals : bool, default=False
        If True, includes detailed internal calculations and intermediate results.
        Required for generating visualizations and detailed analysis reports.
    
    Returns
    -------
    dict
        Dictionary containing VCS score and optionally other metrics and internals.
        
        **Minimal return (default):**
        
        * ``'VCS'`` : float
            The Video Comprehension Score (0.0 to 1.0, higher is better)
            
        **With return_all_metrics=True:**

        * ``'VCS'`` : float - Video Comprehension Score
        * ``'Global_SAS'`` : float - Global Semantic Alignment Score
        * ``'SAS'`` : float - Semantic Alignment Score (scaled Global_SAS and Local_SAS)
        * ``'Precision Local_SAS'`` : float - Local_SAS precision component
        * ``'Recall Local_SAS'`` : float - Local_SAS recall component
        * ``'Local_SAS'`` : float - Local Semantic Alignment Score (F1 of precision/recall)
        * ``'Precision Global NAS'`` : float - Global NAS precision
        * ``'Recall Global NAS'`` : float - Global NAS recall
        * ``'Global NAS'`` : float - Global Narrative Alignment Score
        * ``'Precision Local NAS'`` : float - Local NAS precision
        * ``'Recall Local NAS'`` : float - Local NAS recall
        * ``'Local NAS'`` : float - Local Narrative Alignment Score
        * ``'NAS'`` : float - Final Narrative Alignment Score
            
        **With return_internals=True:**
        
        * ``'internals'`` : dict
            Detailed calculation data for visualization and analysis, containing:
            
            - ``'texts'``: Original and processed text data
            - ``'similarity'``: Similarity matrix and related data  
            - ``'mapping_windows'``: Alignment window information
            - ``'alignment'``: Detailed alignment results
            - ``'metrics'``: Breakdown of all metric calculations
            - ``'config'``: Configuration parameters used
            - ``'best_match'``: Detailed matching information
    
    Raises
    ------
    ValueError
        If embedding functions are not callable, or if both embedding functions are None.
    TypeError
        If segmenter_fn is not callable or doesn't return a list of strings.
    
    Examples
    --------
    **Basic Usage (Minimal Parameters):**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_global_sas=your_embedding_function
        )
        print(f"VCS Score: {result['VCS']:.4f}")

    **With Return Controls:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_global_sas=your_embedding_function,
            return_all_metrics=True,
            return_internals=True
        )

    **With Core Configuration Parameters:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_global_sas=your_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            Rn=1
        )

    **Different Embedding Functions for Global and Local SAS:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_global_sas=your_global_embedding_function,
            embedding_fn_local_sas=your_local_embedding_function
        )

    **Complete Configuration (All Parameters):**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_global_sas=your_global_embedding_function,
            embedding_fn_local_sas=your_local_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            Rn=1,
            return_all_metrics=True,
            return_internals=True
        )
    
    See Also
    --------
    visualize_metrics_summary : Create overview visualization of all metrics
    visualize_similarity_matrix : Visualize the similarity matrix between segments
    visualize_mapping_windows : Show alignment windows used for matching
    create_vcs_pdf_report : Generate comprehensive PDF analysis report
    """
    if embedding_fn_local_sas is None and embedding_fn_global_sas is not None:
        embedding_fn_local_sas = embedding_fn_global_sas
    elif embedding_fn_global_sas is None and embedding_fn_local_sas is not None:
        embedding_fn_global_sas = embedding_fn_local_sas
    if embedding_fn_local_sas is None or embedding_fn_global_sas is None:
        raise ValueError("Provide at least one embedding function (global or local).")

    _validate_seg_embed_functions(segmenter_fn, embedding_fn_local_sas, embedding_fn_global_sas)

    # ===== METHOD: EMBED =====
    # Global Embed (GE)
    global_sas = _compute_global_sas_metrics(reference_text, generated_text, embedding_fn_global_sas)

    # Local Embed (LE): Segmenting and Chunking
    ref_chunks, gen_chunks = _segment_and_chunk_texts(
        reference_text, generated_text, chunk_size, segmenter_fn
    )

    # ===== METHOD: ALIGNMENT =====
    # Build similarity matrix for alignment
    sim_matrix, ref_len, gen_len = _build_similarity_matrix(
        ref_chunks, gen_chunks, embedding_fn_local_sas
    )

    # Alignment Window (AW)
    prec_align_windows, rec_align_windows = _get_alignment_windows(ref_len, gen_len)

    # Alignment-Based Matching (ABM): Precision direction (generated -> reference)
    precision_aligned_matches, precision_aligned_indices, precision_sim_values, precision_match_details = (
        _calculate_alignment_based_matches(
            sim_matrix, prec_align_windows, "precision",
            context_cutoff_value, context_window_control
        )
    )

    # Alignment-Based Matching (ABM): Recall direction (reference -> generated)
    recall_aligned_matches, recall_aligned_indices, recall_sim_values, recall_match_details = (
        _calculate_alignment_based_matches(
            sim_matrix, rec_align_windows, "recall",
            context_cutoff_value, context_window_control
        )
    )

    # ===== METHOD: SAS =====
    # Local-SAS (lSAS): Semantic-Alignment Score at local/chunk level
    local_sas_metrics, local_sas_internals = _compute_local_sas_metrics(
        precision_sim_values, recall_sim_values,
        precision_aligned_indices, recall_aligned_indices,
        ref_len, gen_len
    )

    # Combine Global_SAS + Local_SAS → SAS
    sas_metrics, sas_internals = _compute_sas_metrics(
        global_sas, local_sas_metrics
    )

    # ===== METHOD: NAS =====
    # Narrative Alignment Score (combines Global NAS and Local NAS)
    nas_metrics, nas_internals = _compute_nas_metrics(
        sim_matrix, ref_len, gen_len,
        precision_aligned_matches, precision_aligned_indices, precision_sim_values,
        recall_aligned_matches, recall_aligned_indices, recall_sim_values,
        prec_align_windows, rec_align_windows,
        ref_chunks, gen_chunks,
        Rn=Rn
    )

    # ===== METHOD: VCS =====
    # Final VCS computation combining SAS and NAS
    combined = _compute_vcs_metrics(
        sas_metrics["SAS"], nas_metrics
    )

    if return_all_metrics:
        output: Dict[str, Any] = {**sas_metrics, **nas_metrics, **combined}
    else:
        output = {
            "VCS": combined["VCS"],
        }
    
    if return_internals:
        internals = {
            "texts": {
                "reference_chunks": ref_chunks,
                "generated_chunks": gen_chunks,
                "reference_length": ref_len,
                "generated_length": gen_len,
            },
            "similarity": {
                "matrix": sim_matrix.tolist() if isinstance(sim_matrix, np.ndarray) else sim_matrix,
            },
            "alignment_windows": {
                "precision": prec_align_windows,
                "recall": rec_align_windows,
            },
            "alignment": {
                "precision": {
                    "matches": precision_aligned_matches,
                    "indices": precision_aligned_indices.tolist() if isinstance(precision_aligned_indices, np.ndarray) else precision_aligned_indices,
                    "similarity_values": precision_sim_values.tolist() if isinstance(precision_sim_values, np.ndarray) else precision_sim_values,
                    "aligned_segments": nas_internals["aligned_precision"] if "aligned_precision" in nas_internals else [],
                },
                "recall": {
                    "matches": recall_aligned_matches,
                    "indices": recall_aligned_indices.tolist() if isinstance(recall_aligned_indices, np.ndarray) else recall_aligned_indices,
                    "similarity_values": recall_sim_values.tolist() if isinstance(recall_sim_values, np.ndarray) else recall_sim_values,
                    "aligned_segments": nas_internals["aligned_recall"] if "aligned_recall" in nas_internals else [],
                }
            },
            "metrics": {
                "global_sas": {
                    "value": global_sas,
                },
                "local_sas": {
                    "precision": local_sas_metrics["Precision Local_SAS"],
                    "recall": local_sas_metrics["Recall Local_SAS"],
                    "f1": local_sas_metrics["Local_SAS"],
                    "precision_internals": local_sas_internals["precision"],
                    "recall_internals": local_sas_internals["recall"],
                },
                "sas": {
                    "value": sas_metrics["SAS"],
                    "global_sas_internals": sas_internals.get("global_sas_internals", {}),
                    "local_sas_internals": sas_internals.get("local_sas_internals", {}),
                },
                "global_nas": {
                    "precision": {
                        "value": nas_metrics["Precision Global NAS"],
                        "alignment_window_height": nas_internals["precision_global_nas_internals"]["alignment_window_height"],
                        "max_penalty": nas_internals["precision_global_nas_internals"]["max_penalty"],
                        "total_penalty": nas_internals["precision_global_nas_internals"]["total_penalty"],
                        "penalties": nas_internals["precision_global_nas_internals"]["penalties"],
                        "in_window": nas_internals["precision_global_nas_internals"]["in_window"],
                        "in_rn_zone": nas_internals["precision_global_nas_internals"]["in_rn_zone"],
                    },
                    "recall": {
                        "value": nas_metrics["Recall Global NAS"],
                        "alignment_window_height": nas_internals["recall_global_nas_internals"]["alignment_window_height"],
                        "max_penalty": nas_internals["recall_global_nas_internals"]["max_penalty"],
                        "total_penalty": nas_internals["recall_global_nas_internals"]["total_penalty"],
                        "penalties": nas_internals["recall_global_nas_internals"]["penalties"],
                        "in_window": nas_internals["recall_global_nas_internals"]["in_window"],
                        "in_rn_zone": nas_internals["recall_global_nas_internals"]["in_rn_zone"],
                    },
                    "f1": nas_metrics["Global NAS"],
                },
                "local_nas": {
                    "precision": {
                        "value": nas_metrics["Precision Local NAS"],
                        "actual_line_length": nas_internals["precision_local_nas_internals"]["actual_line_length"],
                        "floor_ideal_line_length": nas_internals["precision_local_nas_internals"]["floor_ideal_line_length"],
                        "ceil_ideal_line_length": nas_internals["precision_local_nas_internals"]["ceil_ideal_line_length"],
                        "average_ideal_line_length": nas_internals["precision_local_nas_internals"]["average_ideal_line_length"],
                        "segments": nas_internals["precision_local_nas_internals"]["segments"],
                        "floor_path": nas_internals["precision_local_nas_internals"]["floor_path"],
                        "ceil_path": nas_internals["precision_local_nas_internals"]["ceil_path"],
                        "actual_path": nas_internals["precision_local_nas_internals"]["actual_path"]
                    },
                    "recall": {
                        "value": nas_metrics["Recall Local NAS"],
                        "actual_line_length": nas_internals["recall_local_nas_internals"]["actual_line_length"],
                        "floor_ideal_line_length": nas_internals["recall_local_nas_internals"]["floor_ideal_line_length"],
                        "ceil_ideal_line_length": nas_internals["recall_local_nas_internals"]["ceil_ideal_line_length"],
                        "average_ideal_line_length": nas_internals["recall_local_nas_internals"]["average_ideal_line_length"],
                        "segments": nas_internals["recall_local_nas_internals"]["segments"],
                        "floor_path": nas_internals["recall_local_nas_internals"]["floor_path"],
                        "ceil_path": nas_internals["recall_local_nas_internals"]["ceil_path"],
                        "actual_path": nas_internals["recall_local_nas_internals"]["actual_path"]
                    },
                    "f1": nas_metrics["Local NAS"],
                },
                "nas": {
                    "value": nas_metrics["NAS"],
                },
                "vcs": {
                    "value": combined["VCS"],
                },
            },
            "config": {
                "chunk_size": chunk_size,
                "context_cutoff_value": context_cutoff_value,
                "context_window_control": context_window_control,
                "Rn": Rn,
            },
            "best_match": {
                "precision": precision_match_details,
                "recall": recall_match_details
            }
        }
        output["internals"] = internals
    
    return output