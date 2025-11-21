import numpy as np
from typing import List, Dict, Any, Tuple
from ..._utils import _calculate_f1

from ._semantic_load_penalty._precision_penalty._precision_penalty import _apply_semantic_load_penalty_precision
from ._semantic_load_penalty._recall_penalty._recall_penalty import _apply_semantic_load_penalty_recall

def _compute_local_sas_metrics(
    precision_sim_values: np.ndarray,
    recall_sim_values: np.ndarray,
    precision_indices: np.ndarray,
    recall_indices: np.ndarray,
    ref_len: int,
    gen_len: int
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute Local SAS metrics with bidirectional semantic load sharing penalty.

    This approach measures semantic coverage by comparing demand vs supply
    for each shared chunk, regardless of overall length balance. Both directions
    are penalized when load sharing occurs.

    Args:
        precision_sim_values: Similarity values for precision direction (gen->ref)
        recall_sim_values: Similarity values for recall direction (ref->gen)
        precision_indices: Matched reference indices for each generated chunk
        recall_indices: Matched generated indices for each reference chunk
        ref_len: Number of reference chunks (m)
        gen_len: Number of generated chunks (n)

    Returns:
        Tuple containing:
        - Dict with Precision Local SAS, Recall Local SAS, and combined Local SAS scores
        - Dict with internal calculations for both directions
    """
    
    # Apply semantic load sharing penalty to both directions
    adjusted_precision_sim, precision_internals = _apply_semantic_load_penalty_precision(
        precision_sim_values, precision_indices, ref_len
    )
    adjusted_recall_sim, recall_internals = _apply_semantic_load_penalty_recall(
        recall_sim_values, recall_indices, gen_len
    )
    
    # Calculate final Local_SAS scores
    precision_local_sas = float(np.mean(adjusted_precision_sim)) if adjusted_precision_sim.size else 0.0
    recall_local_sas = float(np.mean(adjusted_recall_sim)) if adjusted_recall_sim.size else 0.0
    f1_local_sas = _calculate_f1(precision_local_sas, recall_local_sas)

    metrics = {
        "Precision Local_SAS": precision_local_sas,
        "Recall Local_SAS": recall_local_sas,
        "Local_SAS": f1_local_sas
    }
    
    internals = {
        "precision": {
            "original_similarities": precision_sim_values.tolist(),
            "adjusted_similarities": adjusted_precision_sim.tolist(),
            "original_las": float(np.mean(precision_sim_values)) if precision_sim_values.size else 0.0,
            "adjusted_las": precision_las,
            **precision_internals
        },
        "recall": {
            "original_similarities": recall_sim_values.tolist(),
            "adjusted_similarities": adjusted_recall_sim.tolist(),
            "original_las": float(np.mean(recall_sim_values)) if recall_sim_values.size else 0.0,
            "adjusted_las": recall_las,
            **recall_internals
        }
    }
    
    return metrics, internals