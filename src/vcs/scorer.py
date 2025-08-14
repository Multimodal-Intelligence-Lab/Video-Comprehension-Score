from typing import List, Callable, Dict, Any

import numpy as np
import torch

from typing import List, Callable, Dict, Any
import numpy as np
import torch

from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_LCT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAS_BLEND_FACTOR,
    DEFAULT_NAS_COVERAGE_CUTOFF,
)
from ._utils import _validate_seg_embed_functions
from ._segmenting import _segment_and_chunk_texts, _build_similarity_matrix
from ._mapping_windows import _get_mapping_windows
from ._matching import _calculate_row_col_matches_context

from ._metrics import (
    _compute_gas_metrics,
    _compute_las_metrics,
    _compute_nas_metrics,
    _compute_vcs_metrics,
)

def compute_vcs_score(
    reference_text: str,
    generated_text: str,
    segmenter_fn: Callable[[str], List[str]],
    embedding_fn_las: Callable[[List[str]], torch.Tensor],
    embedding_fn_gas: Callable[[List[str]], torch.Tensor] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_cutoff_value: float = DEFAULT_CONTEXT_CUTOFF_VALUE,
    context_window_control: float = DEFAULT_CONTEXT_WINDOW_CONTROL,
    lct: int = DEFAULT_LCT,
    nas_blend_factor: float = DEFAULT_NAS_BLEND_FACTOR,
    nas_coverage_cutoff: float = DEFAULT_NAS_COVERAGE_CUTOFF,
    return_all_metrics: bool = False,
    return_internals: bool = False,
) -> Dict[str, Any]:
    """Compute Video Comprehension Score (VCS) between reference and generated text.
    
    The VCS metric combines Global Alignment Score (GAS), Local Alignment Score (LAS),
    and Narrative Alignment Score (NAS) to provide a comprehensive measure of how well
    a generated text preserves the narrative structure and semantic content of a
    reference text.
    
    **Key Metrics Computed:**
    
    * **GAS (Global Alignment Score)**: Measures overall semantic similarity between 
      the full reference and generated texts using document-level embeddings.
    * **LAS (Local Alignment Score)**: Evaluates segment-by-segment semantic similarity
      using optimal alignment between text chunks.
    * **NAS (Narrative Alignment Score)**: Assesses how well the narrative flow and 
      chronological structure are preserved, combining distance-based and line-based 
      alignment measures.
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
    embedding_fn_las : callable
        Function to compute embeddings for Local Alignment Score calculation. Must 
        take a list of strings (text segments) and return a torch.Tensor of shape 
        (n_segments, embedding_dim) where each row is the embedding for one segment.
        
        Example: A function that uses sentence transformers or other semantic models.
    embedding_fn_gas : callable, optional
        Function to compute embeddings for Global Alignment Score calculation. If 
        None, uses ``embedding_fn_las`` for both GAS and LAS calculations. Should 
        follow the same signature as ``embedding_fn_las``.
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
    lct : int, default=0
        Local Chronology Tolerance - allows flexibility in narrative ordering. 
        Higher values permit more deviation from strict chronological order:
        
        - ``lct=0``: Strict chronological order required
        - ``lct=1``: Small deviations allowed
        - ``lct=2+``: More flexible chronological matching
    nas_blend_factor : float, default=0.0
        Controls how window penalty and imbalance penalty are blended in NAS 
        regularization. Must be between -1.0 and 1.0:
        
        - ``-1.0``: Follow imbalance penalty exclusively (prioritize length differences)
        - ``0.0``: Balanced blend of both penalties (recommended)
        - ``+1.0``: Follow window penalty exclusively (prioritize coverage patterns)
    nas_coverage_cutoff : float, default=0.5
        Coverage threshold above which NAS is considered meaningless due to 
        excessive mapping window overlap. Must be between 0.0 and 1.0:
        
        - ``0.3-0.4``: Strict coverage requirements
        - ``0.5-0.6``: Balanced coverage assessment (recommended)
        - ``0.7-0.8``: Permissive coverage allowance
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
        * ``'GAS'`` : float - Global Alignment Score
        * ``'GAS-LAS-Scaled'`` : float - Scaled combination of GAS and LAS
        * ``'Precision LAS'`` : float - LAS precision component
        * ``'Recall LAS'`` : float - LAS recall component  
        * ``'LAS'`` : float - Local Alignment Score (F1 of precision/recall)
        * ``'Precision NAS-D'`` : float - Distance-based NAS precision
        * ``'Recall NAS-D'`` : float - Distance-based NAS recall
        * ``'NAS-D'`` : float - Distance-based Narrative Alignment Score
        * ``'Precision NAS-L'`` : float - Line-based NAS precision
        * ``'Recall NAS-L'`` : float - Line-based NAS recall
        * ``'NAS-L'`` : float - Line-based Narrative Alignment Score
        * ``'NAS-F1'`` : float - Combined NAS-D and NAS-L score
        * ``'Length-Regularizer'`` : float - Regularization factor for length imbalance
        * ``'NAS'`` : float - Final regularized Narrative Alignment Score
            
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
            embedding_fn_las=your_embedding_function
        )
        print(f"VCS Score: {result['VCS']:.4f}")
    
    **With Return Controls:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_las=your_embedding_function,
            return_all_metrics=True,
            return_internals=True
        )
    
    **With Core Configuration Parameters:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_las=your_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            lct=1,
            nas_blend_factor=0.2,
            nas_coverage_cutoff=0.6
        )
    
    **Different Embedding Functions for GAS and LAS:**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_las=your_local_embedding_function,
            embedding_fn_gas=your_global_embedding_function
        )
    
    **Complete Configuration (All Parameters):**
    
    .. code-block:: python
    
        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn_las=your_embedding_function,
            embedding_fn_gas=your_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            lct=1,
            nas_blend_factor=0.2,
            nas_coverage_cutoff=0.6,
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
    if embedding_fn_las is None and embedding_fn_gas is not None:
        embedding_fn_las = embedding_fn_gas
    elif embedding_fn_gas is None and embedding_fn_las is not None:
        embedding_fn_gas = embedding_fn_las
    if embedding_fn_las is None or embedding_fn_gas is None:
        raise ValueError("Provide at least one embedding function (LAS or GAS).")

    _validate_seg_embed_functions(segmenter_fn, embedding_fn_las, embedding_fn_gas)
    
    gas_val = _compute_gas_metrics(reference_text, generated_text, embedding_fn_gas)

    ref_chunks, gen_chunks = _segment_and_chunk_texts(
        reference_text, generated_text, chunk_size, segmenter_fn
    )

    sim_matrix, ref_len, gen_len = _build_similarity_matrix(
        ref_chunks, gen_chunks, embedding_fn_las
    )

    prec_map_windows, rec_map_windows = _get_mapping_windows(ref_len, gen_len)

    precision_matches, precision_indices, precision_sim_values, precision_match_details = (
        _calculate_row_col_matches_context(
            sim_matrix, prec_map_windows, "precision",
            context_cutoff_value, context_window_control
        )
    )
    recall_matches, recall_indices, recall_sim_values, recall_match_details = (
        _calculate_row_col_matches_context(
            sim_matrix, rec_map_windows, "recall",
            context_cutoff_value, context_window_control
        )
    )

    las_metrics = _compute_las_metrics(precision_sim_values, recall_sim_values)
    nas_metrics, nas_internals = _compute_nas_metrics(
        sim_matrix, ref_len, gen_len,
        precision_matches, precision_indices, precision_sim_values,
        recall_matches, recall_indices, recall_sim_values,
        prec_map_windows, rec_map_windows,
        ref_chunks, gen_chunks,
        lct=lct,
        nas_blend_factor=nas_blend_factor,
        nas_coverage_cutoff=nas_coverage_cutoff
    )
    combined = _compute_vcs_metrics(
        gas_val, nas_metrics["NAS"], las_metrics["LAS"]
    )

    if return_all_metrics:
        output: Dict[str, Any] = {**las_metrics, **nas_metrics, **combined}
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
            "mapping_windows": {
                "precision": prec_map_windows,
                "recall": rec_map_windows,
            },
            "alignment": {
                "precision": {
                    "matches": precision_matches,
                    "indices": precision_indices.tolist() if isinstance(precision_indices, np.ndarray) else precision_indices,
                    "similarity_values": precision_sim_values.tolist() if isinstance(precision_sim_values, np.ndarray) else precision_sim_values,
                    "aligned_segments": nas_internals["aligned_precision"] if "aligned_precision" in nas_internals else [],
                },
                "recall": {
                    "matches": recall_matches,
                    "indices": recall_indices.tolist() if isinstance(recall_indices, np.ndarray) else recall_indices,
                    "similarity_values": recall_sim_values.tolist() if isinstance(recall_sim_values, np.ndarray) else recall_sim_values,
                    "aligned_segments": nas_internals["aligned_recall"] if "aligned_recall" in nas_internals else [],
                }
            },
            "metrics": {
                "gas": {
                    "value": gas_val,
                },
                "las": {
                    "precision": las_metrics["Precision LAS"],
                    "recall": las_metrics["Recall LAS"],
                    "f1": las_metrics["LAS"],
                },
                "nas": {
                    "nas_d": {
                        "precision": {
                            "value": nas_metrics["Precision NAS-D"],
                            "mapping_window_height": nas_internals["precision_nas_internals"]["mapping_window_height"],
                            "max_penalty": nas_internals["precision_nas_internals"]["max_penalty"],
                            "total_penalty": nas_internals["precision_nas_internals"]["total_penalty"],
                            "penalties": nas_internals["precision_nas_internals"]["penalties"],
                            "in_window": nas_internals["precision_nas_internals"]["in_window"],
                            "in_lct_zone": nas_internals["precision_nas_internals"]["in_lct_zone"],
                        },
                        "recall": {
                            "value": nas_metrics["Recall NAS-D"],
                            "mapping_window_height": nas_internals["recall_nas_internals"]["mapping_window_height"],
                            "max_penalty": nas_internals["recall_nas_internals"]["max_penalty"],
                            "total_penalty": nas_internals["recall_nas_internals"]["total_penalty"],
                            "penalties": nas_internals["recall_nas_internals"]["penalties"],
                            "in_window": nas_internals["recall_nas_internals"]["in_window"],
                            "in_lct_zone": nas_internals["recall_nas_internals"]["in_lct_zone"],
                        },
                        "f1": nas_metrics["NAS-D"],
                    },
                    "nas_l": {
                        "precision": {
                            "value": nas_metrics["Precision NAS-L"],
                            "actual_line_length": nas_internals["precision_line_internals"]["actual_line_length"],
                            "floor_ideal_line_length": nas_internals["precision_line_internals"]["floor_ideal_line_length"],
                            "ceil_ideal_line_length": nas_internals["precision_line_internals"]["ceil_ideal_line_length"],
                            "average_ideal_line_length": nas_internals["precision_line_internals"]["average_ideal_line_length"],
                            "segments": nas_internals["precision_line_internals"]["segments"],
                            "floor_path": nas_internals["precision_line_internals"]["floor_path"],
                            "ceil_path": nas_internals["precision_line_internals"]["ceil_path"],
                            "actual_path": nas_internals["precision_line_internals"]["actual_path"]
                        },
                        "recall": {
                            "value": nas_metrics["Recall NAS-L"],
                            "actual_line_length": nas_internals["recall_line_internals"]["actual_line_length"],
                            "floor_ideal_line_length": nas_internals["recall_line_internals"]["floor_ideal_line_length"],
                            "ceil_ideal_line_length": nas_internals["recall_line_internals"]["ceil_ideal_line_length"],
                            "average_ideal_line_length": nas_internals["recall_line_internals"]["average_ideal_line_length"],
                            "segments": nas_internals["recall_line_internals"]["segments"],
                            "floor_path": nas_internals["recall_line_internals"]["floor_path"],
                            "ceil_path": nas_internals["recall_line_internals"]["ceil_path"],
                            "actual_path": nas_internals["recall_line_internals"]["actual_path"]
                        },
                        "f1": nas_metrics["NAS-L"],
                    },
                    "regularizer": {
                        "value": nas_metrics["Length-Regularizer"],
                        "total_mapping_window_area": nas_internals["regularizer_internals"]["total_mapping_window_area"],
                        "timeline_area": nas_internals["regularizer_internals"]["timeline_area"],
                        "min_area": nas_internals["regularizer_internals"]["min_area"],
                    },
                    "nas_f1": nas_metrics["NAS-F1"],
                    "regularized_nas": nas_metrics["NAS"],
                },
                "vcs": {
                    "value": combined["VCS"],
                    "gas_las_scaled": combined["GAS-LAS-Scaled"],
                },
            },
            "config": {
                "chunk_size": chunk_size,
                "context_cutoff_value": context_cutoff_value,
                "context_window_control": context_window_control,
                "lct": lct,
                "nas_blend_factor": nas_blend_factor,
                "nas_coverage_cutoff": nas_coverage_cutoff,
            },
            "best_match": {
                "precision": precision_match_details,
                "recall": recall_match_details
            }
        }
        output["internals"] = internals
    
    return output