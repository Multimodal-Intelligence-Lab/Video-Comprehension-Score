from typing import List, Callable, Dict, Any

import numpy as np
import torch

from ._config import (
    DEFAULT_CONTEXT_CUTOFF_VALUE,
    DEFAULT_CONTEXT_WINDOW_CONTROL,
    DEFAULT_Rn,
    DEFAULT_CHUNK_SIZE,
)
from ._validation import (
    _resolve_chunk_texts,
    _validate_chunk_embeddings,
    _validate_doc_embedding,
    _validate_parameters,
    _validate_seg_embed_functions,
    _validate_texts,
)
from ._segmenting import (
    _segment_and_chunk_texts,
    _build_similarity_matrix,
    _similarity_from_chunk_embeddings,
)
from ._windows import _get_alignment_windows
from ._matching import _calculate_alignment_based_matches
from ._sas import (
    _compute_global_sas_metrics,
    _compute_local_sas_metrics,
    _compute_sas_metrics,
    _global_sas_from_embeddings,
)
from ._nas import _compute_nas_metrics
from ._utils import _resolve_version
from ._vcs import _compute_vcs_metrics

def compute_vcs_score(
    reference_text: str,
    generated_text: str,
    segmenter_fn: Callable[[str], List[str]],
    embedding_fn: Callable[[List[str]], torch.Tensor],
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
      embeddings.
    * **Local_SAS (Local Semantic Alignment Score)**: Evaluates segment-by-segment
      semantic similarity using optimal alignment between text chunks.
    * **NAS (Narrative Alignment Score)**: Assesses how well the narrative flow and
      chronological structure are preserved, combining Global NAS and Local NAS measures.
    * **SAS (Semantic Alignment Score)**: Scaled combination of Global_SAS and Local_SAS.
    * **VCS (Video Comprehension Score)**: The final combined score that balances all
      three metrics to provide an overall narrative similarity assessment.

    **Reading the score:**

    VCS compounds its components. With SAS = NAS = x the score is
    ``max(0, 2x - 1) / x``, so headline numbers run lower than raw cosine
    intuition suggests:

    ========  ======
    SAS=NAS   VCS
    ========  ======
    <= 0.5    0.0
    0.6       0.33
    0.7       0.57
    0.8       0.75
    0.9       0.89
    1.0       1.0
    ========  ======

    Calibrate expectations (and any pass/fail thresholds) on this scale.
    VCS = 0.0 is a gate, not a fine-grained judgment: content (SAS) and
    order (NAS) must both clear it. To rank candidates below the gate,
    use ``'VCS Margin'`` from ``return_all_metrics``.

    Repeated content is tolerated by design: generated chunks that
    paraphrase the same reference chunk share that chunk's credit through
    the load-sharing penalty, but correct content stated twice does not
    score below correct content stated once.

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
    embedding_fn : callable
        Function to compute embeddings, used for BOTH the document-level
        (Global_SAS) and chunk-level (Local_SAS) measurements — one
        instrument, so the SAS combination compares like with like. Must
        take a list of strings and return a torch.Tensor of shape
        (n_items, embedding_dim) where each row is an embedding.

        Example: A function that uses sentence transformers or other semantic models.
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
        Measured in CHUNKS (after ``chunk_size`` grouping, not in
        sentences). Higher values permit more deviation from strict
        chronological order:

        - ``Rn=0``: Strict chronological order required
        - ``Rn=1``: Small deviations allowed
        - ``Rn=2+``: More flexible chronological matching

        Keep Rn small relative to the chunk count unless deliberately
        relaxing order: at grid scale it disables the NAS dial entirely
        (an intended escape hatch when order should not matter), and on
        small grids a moderate Rn already blunts direction sensitivity.
    return_all_metrics : bool, default=False
        If True, returns all intermediate metrics (GAS, LAS, NAS components) in 
        addition to the final VCS score. Useful for detailed analysis.
    return_internals : bool, default=False
        If True, includes detailed internal calculations and intermediate results.
        Useful for downstream analysis and custom tooling.
    
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
        * ``'VCS Margin'`` : float - SAS + NAS - 1 in [-1, 1]; ranks
          candidates below the VCS zero gate (VCS > 0 iff margin > 0)
        * ``'Config'`` : str - library version + knob values used
          (``vcs=...|chunk_size=...|rn=...|context_cutoff=...|context_window_control=...``).
          Knob values are formatted as passed (``0.6``, ``1`` and ``1.0``
          stay distinct) — compare Configs from runs that pass knobs the
          same way.
            
        **With return_internals=True:**
        
        * ``'internals'`` : dict
            Detailed calculation data for analysis, containing:
            
            - ``'texts'``: Original and processed text data
            - ``'similarity'``: Similarity matrix and related data
            - ``'alignment_windows'``: Alignment window information
            - ``'alignment'``: Detailed alignment results
            - ``'metrics'``: Breakdown of all metric calculations
            - ``'config'``: Configuration parameters used
            - ``'best_match'``: Detailed matching information
    
    Raises
    ------
    ValueError
        If any input is invalid: missing/non-callable functions, empty or
        non-str texts, out-of-range parameters (``chunk_size`` < 1,
        ``context_cutoff_value`` outside [0, 1],
        ``context_window_control`` <= 0, ``Rn`` < 0, bools where numbers are
        expected), a segmenter that produces no chunks, an embedding
        function that returns anything other than a finite 2-D
        ``torch.Tensor`` with one row per input text, or an embedding row
        with L2 norm 0 (which cannot be normalized).

    Notes
    -----
    VCS uses raw dot products as similarities, so embedding rows are
    L2-normalized internally: rows already within ``1e-12`` of unit norm
    pass through bit-untouched, other rows are divided by their norms.
    Scores are therefore scale-invariant in each embedding row. (float32
    rows normalized by the caller rarely measure within ``1e-12`` in
    float64 and are simply renormalized — results are identical either
    way.)

    Examples
    --------
    **Basic Usage (Minimal Parameters):**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn=your_embedding_function
        )
        print(f"VCS Score: {result['VCS']:.4f}")

    **With Return Controls:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn=your_embedding_function,
            return_all_metrics=True,
            return_internals=True
        )

    **With Core Configuration Parameters:**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn=your_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            Rn=1
        )

    **Complete Configuration (All Parameters):**

    .. code-block:: python

        result = compute_vcs_score(
            reference_text="Your reference text here",
            generated_text="Your generated text here",
            segmenter_fn=your_segmenter_function,
            embedding_fn=your_embedding_function,
            chunk_size=2,
            context_cutoff_value=0.7,
            context_window_control=3.0,
            Rn=1,
            return_all_metrics=True,
            return_internals=True
        )

    """
    _validate_seg_embed_functions(segmenter_fn, embedding_fn)
    _validate_texts(reference_text, generated_text)
    _validate_parameters(chunk_size, context_cutoff_value, context_window_control, Rn)

    # ===== METHOD: EMBED =====
    # Global Embed (GE)
    global_sas = _compute_global_sas_metrics(reference_text, generated_text, embedding_fn)

    # Local Embed (LE): Segmenting and Chunking
    ref_chunks, gen_chunks = _segment_and_chunk_texts(
        reference_text, generated_text, chunk_size, segmenter_fn
    )

    # ===== METHOD: ALIGNMENT =====
    # Build similarity matrix for alignment
    sim_matrix, ref_len, gen_len = _build_similarity_matrix(
        ref_chunks, gen_chunks, embedding_fn
    )

    return _run_vcs_pipeline(
        global_sas, sim_matrix, ref_len, gen_len, ref_chunks, gen_chunks,
        chunk_size, context_cutoff_value, context_window_control, Rn,
        return_all_metrics, return_internals,
    )


def compute_vcs_from_embeddings(
    reference_doc_embedding: torch.Tensor,
    generated_doc_embedding: torch.Tensor,
    reference_chunk_embeddings: torch.Tensor,
    generated_chunk_embeddings: torch.Tensor,
    *,
    reference_chunks: List[str] | None = None,
    generated_chunks: List[str] | None = None,
    context_cutoff_value: float = DEFAULT_CONTEXT_CUTOFF_VALUE,
    context_window_control: float = DEFAULT_CONTEXT_WINDOW_CONTROL,
    Rn: int = DEFAULT_Rn,
    return_all_metrics: bool = False,
    return_internals: bool = False,
) -> Dict[str, Any]:
    """Compute VCS directly from pre-computed embeddings.

    Identical to :func:`compute_vcs_score` from the similarity computation
    onward — given embeddings equal to what ``embedding_fn`` would have
    produced, the two entry points return exactly equal results. Use this
    when embeddings are already available (batch pipelines, cached
    embeddings, sweeping VCS parameters without re-embedding).

    One-instrument contract: document and chunk embeddings should come
    from the SAME embedder, like :func:`compute_vcs_score`'s single
    ``embedding_fn`` — the SAS combination assumes Global_SAS and
    Local_SAS measure with the same instrument.

    Parameters
    ----------
    reference_doc_embedding : torch.Tensor
        Document-level embedding of the full reference text, shape
        ``(embedding_dim,)`` (a single-row 2-D tensor is also accepted).
        Used for Global SAS via cosine similarity.
    generated_doc_embedding : torch.Tensor
        Document-level embedding of the full generated text, same shape
        rules as ``reference_doc_embedding``.
    reference_chunk_embeddings : torch.Tensor
        Embeddings of the reference chunks in narrative order, shape
        ``(n_reference_chunks, embedding_dim)``. Similarities are raw dot
        products, so rows are L2-normalized internally (rows already at
        unit norm pass through bit-untouched).
    generated_chunk_embeddings : torch.Tensor
        Embeddings of the generated chunks in narrative order, shape
        ``(n_generated_chunks, embedding_dim)``.
    reference_chunks : list of str, optional
        Chunk texts for the reference side. Texts appear only inside
        ``internals`` (aligned-segment listings); they never affect any
        score. When omitted, ``"<chunk i>"`` placeholders are used.
    generated_chunks : list of str, optional
        Chunk texts for the generated side; same rules.
    context_cutoff_value : float, default=0.6
        See :func:`compute_vcs_score`.
    context_window_control : float, default=4.0
        See :func:`compute_vcs_score`.
    Rn : int, default=0
        See :func:`compute_vcs_score`.
    return_all_metrics : bool, default=False
        See :func:`compute_vcs_score`.
    return_internals : bool, default=False
        See :func:`compute_vcs_score`. Note: since chunking already
        happened upstream, ``internals['config']['chunk_size']`` is None
        for this entry point and the ``'Config'`` string reads
        ``chunk_size=none``.

    Returns
    -------
    dict
        Same structure as :func:`compute_vcs_score`.

    Raises
    ------
    ValueError
        If any embedding is not a finite torch.Tensor of the documented
        shape, any embedding row has L2 norm 0, the two document embeddings
        (or the two chunk-embedding matrices) disagree on embedding_dim,
        chunk texts are provided with the wrong length, or a knob parameter
        is out of range.
    """
    # chunk_size has no meaning here (chunks arrive pre-built): validate the
    # remaining knobs against the same rules as compute_vcs_score.
    _validate_parameters(1, context_cutoff_value, context_window_control, Rn)

    ref_doc_vec = _validate_doc_embedding(reference_doc_embedding, "reference_doc_embedding")
    gen_doc_vec = _validate_doc_embedding(generated_doc_embedding, "generated_doc_embedding")
    if ref_doc_vec.shape != gen_doc_vec.shape:
        raise ValueError(
            f"document embeddings disagree on embedding_dim: "
            f"{ref_doc_vec.shape[0]} vs {gen_doc_vec.shape[0]}"
        )

    ref_tensor = _validate_chunk_embeddings(reference_chunk_embeddings, "reference_chunk_embeddings")
    gen_tensor = _validate_chunk_embeddings(generated_chunk_embeddings, "generated_chunk_embeddings")
    if ref_tensor.shape[1] != gen_tensor.shape[1]:
        raise ValueError(
            f"chunk embeddings disagree on embedding_dim: "
            f"{ref_tensor.shape[1]} vs {gen_tensor.shape[1]}"
        )

    ref_len, gen_len = ref_tensor.shape[0], gen_tensor.shape[0]
    ref_chunks = _resolve_chunk_texts(reference_chunks, ref_len, "reference_chunks")
    gen_chunks = _resolve_chunk_texts(generated_chunks, gen_len, "generated_chunks")

    # Same two ops compute_vcs_score performs on freshly produced embeddings
    global_sas = _global_sas_from_embeddings(ref_doc_vec, gen_doc_vec)
    sim_matrix = _similarity_from_chunk_embeddings(ref_tensor, gen_tensor)

    return _run_vcs_pipeline(
        global_sas, sim_matrix, ref_len, gen_len, ref_chunks, gen_chunks,
        None, context_cutoff_value, context_window_control, Rn,
        return_all_metrics, return_internals,
    )


def _run_vcs_pipeline(
    global_sas: float,
    sim_matrix: np.ndarray,
    ref_len: int,
    gen_len: int,
    ref_chunks: List[str],
    gen_chunks: List[str],
    chunk_size: int | None,
    context_cutoff_value: float,
    context_window_control: float,
    Rn: int,
    return_all_metrics: bool,
    return_internals: bool,
) -> Dict[str, Any]:
    """Shared metric pipeline: everything downstream of the similarity
    matrix. Both public entry points feed this with identical inputs, so
    their outputs are identical by construction. chunk_size is None when
    chunks arrived pre-embedded (compute_vcs_from_embeddings)."""
    # Alignment Window (AW)
    prec_align_windows, rec_align_windows = _get_alignment_windows(ref_len, gen_len)

    # Alignment-Based Matching (ABM): Precision direction (generated -> reference)
    precision_aligned_matches, precision_aligned_indices, precision_sim_values, precision_match_details = (
        _calculate_alignment_based_matches(
            sim_matrix, prec_align_windows, "precision",
            context_cutoff_value, context_window_control,
            collect_details=return_internals,
        )
    )

    # Alignment-Based Matching (ABM): Recall direction (reference -> generated)
    recall_aligned_matches, recall_aligned_indices, recall_sim_values, recall_match_details = (
        _calculate_alignment_based_matches(
            sim_matrix, rec_align_windows, "recall",
            context_cutoff_value, context_window_control,
            collect_details=return_internals,
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
        global_sas, local_sas_metrics, local_sas_internals
    )

    # ===== METHOD: NAS =====
    # Narrative Alignment Score (combines Global NAS and Local NAS)
    nas_metrics, nas_internals = _compute_nas_metrics(
        sim_matrix, ref_len, gen_len,
        precision_aligned_matches, precision_aligned_indices, precision_sim_values,
        recall_aligned_matches, recall_aligned_indices, recall_sim_values,
        prec_align_windows, rec_align_windows,
        ref_chunks, gen_chunks,
        Rn=Rn,
        collect_details=return_internals,
    )

    # ===== METHOD: VCS =====
    # Final VCS computation combining SAS and NAS
    combined = _compute_vcs_metrics(
        sas_metrics["SAS"], nas_metrics
    )

    if return_all_metrics:
        # The embedder's identity is the caller's to report — the library
        # never sees a model name, so the Config string covers exactly the
        # knobs the library controls (plus its own version).
        config_string = (
            f"vcs={_resolve_version()}"
            f"|chunk_size={chunk_size if chunk_size is not None else 'none'}"
            f"|rn={Rn}"
            f"|context_cutoff={context_cutoff_value}"
            f"|context_window_control={context_window_control}"
        )
        output: Dict[str, Any] = {**sas_metrics, **nas_metrics, **combined,
                                  "Config": config_string}
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
                        "max_penalty": nas_internals["precision_global_nas_internals"]["max_penalty"],
                        "total_penalty": nas_internals["precision_global_nas_internals"]["total_penalty"],
                        "penalties": nas_internals["precision_global_nas_internals"]["penalties"],
                        "in_window": nas_internals["precision_global_nas_internals"]["in_window"],
                        "in_rn_zone": nas_internals["precision_global_nas_internals"]["in_rn_zone"],
                    },
                    "recall": {
                        "value": nas_metrics["Recall Global NAS"],
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
                    "margin": combined["VCS Margin"],
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