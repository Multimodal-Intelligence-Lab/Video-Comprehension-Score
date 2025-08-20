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
    Compute LAS metrics with bidirectional semantic load adequacy penalty.
    
    This approach measures semantic adequacy by comparing demand vs supply
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
        Dict with Precision LAS, Recall LAS, and combined LAS scores
    """
    
    # Apply semantic load adequacy penalty to both directions
    adjusted_precision_sim = _apply_semantic_load_penalty_precision(
        precision_sim_values, precision_indices, ref_len
    )
    adjusted_recall_sim = _apply_semantic_load_penalty_recall(
        recall_sim_values, recall_indices, gen_len
    )
    
    # Calculate final LAS scores
    precision_las = float(np.mean(adjusted_precision_sim)) if adjusted_precision_sim.size else 0.0
    recall_las = float(np.mean(adjusted_recall_sim)) if adjusted_recall_sim.size else 0.0
    f1_las = _calculate_f1(precision_las, recall_las)
    
    return {
        "Precision LAS": precision_las,
        "Recall LAS": recall_las,
        "LAS": f1_las 
    }

def _apply_semantic_load_penalty_recall(
    recall_sim_values: np.ndarray,
    recall_indices: np.ndarray,
    gen_len: int
) -> np.ndarray:
    """
    Apply semantic load adequacy penalty for recall direction.
    
    Groups reference chunks by their chosen generated chunk and calculates
    semantic adequacy based on demand vs supply ratio.
    """
    if recall_sim_values.size == 0:
        return recall_sim_values
    
    adjusted_sim = recall_sim_values.copy()
    
    # Group refs by chosen gen: G_j^R = {i : j_i* = j}
    for gen_idx in range(gen_len):
        # Find all ref chunks that chose this gen chunk
        sharing_refs = np.where(recall_indices == gen_idx)[0]
        K_j_R = len(sharing_refs)  # Number of refs sharing this gen
        
        if K_j_R >= 2:  # Only apply penalty when sharing occurs
            # Calculate semantic load factor: α_j^R = (1/K_j^R) * Σ r_i
            supply = np.sum(recall_sim_values[sharing_refs])
            demand = K_j_R  # Each ref demands 1 unit
            alpha_j_R = supply / demand  # Semantic adequacy factor
            
            # Apply gate function ψ_α(s) to each sharing ref
            for ref_idx in sharing_refs:
                original_sim = recall_sim_values[ref_idx]
                adjusted_sim[ref_idx] = _apply_gate_function(original_sim, alpha_j_R)
    
    return adjusted_sim

def _apply_semantic_load_penalty_precision(
    precision_sim_values: np.ndarray,
    precision_indices: np.ndarray,
    ref_len: int
) -> np.ndarray:
    """
    Apply semantic load adequacy penalty for precision direction.
    
    Groups generated chunks by their chosen reference chunk and calculates
    semantic adequacy based on demand vs supply ratio.
    """
    if precision_sim_values.size == 0:
        return precision_sim_values
    
    adjusted_sim = precision_sim_values.copy()
    
    # Group gens by chosen ref: H_i^P = {j : i_j* = i}
    for ref_idx in range(ref_len):
        # Find all gen chunks that chose this ref chunk
        sharing_gens = np.where(precision_indices == ref_idx)[0]
        K_i_P = len(sharing_gens)  # Number of gens sharing this ref
        
        if K_i_P >= 2:  # Only apply penalty when sharing occurs
            # Calculate semantic load factor: α_i^P = (1/K_i^P) * Σ p_j
            supply = np.sum(precision_sim_values[sharing_gens])
            demand = K_i_P  # Each gen demands 1 unit
            alpha_i_P = supply / demand  # Semantic adequacy factor
            
            # Apply gate function ψ_α(s) to each sharing gen
            for gen_idx in sharing_gens:
                original_sim = precision_sim_values[gen_idx]
                adjusted_sim[gen_idx] = _apply_gate_function(original_sim, alpha_i_P)
    
    return adjusted_sim

def _apply_gate_function(similarity: float, alpha: float) -> float:
    """
    Apply the affine gate function ψ_α(s).
    
    ψ_α(s) = {
        0,                    if α=0 or s ≤ 1-α
        (s - (1-α)) / α,     if α∈(0,1] and s > 1-α
    }
    
    Key properties:
    - ψ_α(1) = 1 (perfect similarity maps to 1)
    - Threshold at 1-α
    - Monotone and bounded in [0,1]
    - ψ_1(s) = s (no penalty when adequacy is perfect)
    
    Args:
        similarity: Original similarity value
        alpha: Semantic adequacy factor (0 ≤ α ≤ 1)
    
    Returns:
        Penalized similarity value
    """
    if alpha == 0 or similarity <= (1 - alpha):
        return 0.0
    else:
        return (similarity - (1 - alpha)) / alpha