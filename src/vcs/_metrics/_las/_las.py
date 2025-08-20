import numpy as np
from typing import List, Dict
from ..._utils import _calculate_f1

def _compute_las_metrics(
    precision_sim_values: np.ndarray,
    recall_sim_values: np.ndarray,
    precision_indices: np.ndarray,
    recall_indices: np.ndarray,
    ref_len: int,
    gen_len: int
) -> Dict[str, float]:
    """
    Compute LAS metrics with semantic load sharing penalty applied only to the 
    direction with more chunks (where length imbalance causes load sharing).
    
    Args:
        precision_sim_values: Similarity values for precision direction (gen->ref)
        recall_sim_values: Similarity values for recall direction (ref->gen)
        precision_indices: Matched reference indices for each generated chunk
        recall_indices: Matched generated indices for each reference chunk
        ref_len: Number of reference chunks (m)
        gen_len: Number of generated chunks (n)
    
    Returns:
        Dict with Precision LAS, Recall LAS, and combined LAS scores
    """
    
    # Apply penalty only in the direction with more chunks
    if gen_len > ref_len:
        # More generated chunks - penalize precision direction
        I = ref_len / gen_len
        precision_provider_loads = _calculate_provider_loads(precision_indices, ref_len)
        adjusted_precision_sim = _apply_load_sharing_penalty(
            precision_sim_values, precision_indices, precision_provider_loads, I
        )
        adjusted_recall_sim = recall_sim_values  # No penalty
        
    elif ref_len > gen_len:
        # More reference chunks - penalize recall direction  
        I = gen_len / ref_len
        recall_provider_loads = _calculate_provider_loads(recall_indices, gen_len)
        adjusted_recall_sim = _apply_load_sharing_penalty(
            recall_sim_values, recall_indices, recall_provider_loads, I
        )
        adjusted_precision_sim = precision_sim_values  # No penalty
        
    else:
        # Balanced chunks - no penalty needed
        adjusted_precision_sim = precision_sim_values
        adjusted_recall_sim = recall_sim_values
    
    # Calculate final LAS scores
    precision_las = float(np.mean(adjusted_precision_sim)) if adjusted_precision_sim.size else 0.0
    recall_las = float(np.mean(adjusted_recall_sim)) if adjusted_recall_sim.size else 0.0
    f1_las = _calculate_f1(precision_las, recall_las)
    
    return {
        "Precision LAS": precision_las,
        "Recall LAS": recall_las,
        "LAS": f1_las 
    }

def _calculate_provider_loads(indices: np.ndarray, provider_count: int) -> np.ndarray:
    """
    Calculate how many chunks each provider (target) chunk serves.
    
    Args:
        indices: Array of matched indices (-1 for no match)
        provider_count: Total number of provider chunks
    
    Returns:
        Array where loads[i] = number of chunks that matched to provider i
    """
    loads = np.zeros(provider_count, dtype=int)
    valid_indices = indices[indices >= 0]
    if valid_indices.size > 0:
        unique_indices, counts = np.unique(valid_indices, return_counts=True)
        loads[unique_indices] = counts
    return loads

def _apply_load_sharing_penalty(
    sim_values: np.ndarray,
    indices: np.ndarray, 
    provider_loads: np.ndarray,
    I: float
) -> np.ndarray:
    """
    Apply the imbalance shrinker gate function to penalize load-shared matches.
    
    The gate function φ_I(s) = max(0, (s - (1-I)) / I) is applied only to
    matches where the provider chunk serves multiple chunks (load > 1).
    
    Args:
        sim_values: Original similarity values
        indices: Matched provider indices
        provider_loads: Load count for each provider chunk
        I: Global chunk imbalance factor (0 < I <= 1)
    
    Returns:
        Adjusted similarity values with penalties applied
    """
    if sim_values.size == 0:
        return sim_values
    
    adjusted_sim = sim_values.copy()
    
    for i, (sim_val, provider_idx) in enumerate(zip(sim_values, indices)):
        if provider_idx >= 0 and provider_loads[provider_idx] > 1:
            # Apply gate function: φ_I(s) = max(0, (s - (1-I)) / I)
            threshold = 1 - I
            if sim_val > threshold:
                adjusted_sim[i] = (sim_val - threshold) / I
            else:
                adjusted_sim[i] = 0.0
        # If load == 1, keep original similarity (no penalty)
    
    return adjusted_sim