import numpy as np
from typing import Dict, Any, Tuple

from .._scaling_function._scaling_function import _apply_scaling_function

def _apply_semantic_load_penalty_precision(
    precision_sim_values: np.ndarray,
    precision_indices: np.ndarray,
    ref_len: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply semantic load sharing penalty for precision direction.
    
    Groups generated chunks by their chosen reference chunk and calculates
    semantic coverage based on demand vs supply ratio.
    
    Returns:
        Tuple containing adjusted similarities and internal calculation details
    """
    if precision_sim_values.size == 0:
        return precision_sim_values, {"load_sharing_details": []}
    
    adjusted_sim = precision_sim_values.copy()
    load_sharing_details = []
    
    # Group generated chunks by their chosen reference chunk
    for reference_chunk_idx in range(ref_len):
        # Find all generated chunks that chose this reference chunk
        generated_chunks_sharing_this_ref = np.where(precision_indices == reference_chunk_idx)[0]
        num_gens_sharing_this_ref = len(generated_chunks_sharing_this_ref)
        
        if num_gens_sharing_this_ref >= 2:  # Only apply penalty when load sharing occurs
            # Calculate semantic coverage ratio: total_supply / total_demand
            total_supply = np.sum(precision_sim_values[generated_chunks_sharing_this_ref])
            total_demand = num_gens_sharing_this_ref  # Each generated chunk demands 1 unit
            semantic_coverage = total_supply / total_demand
            chunk_details = []
            
            # Apply scaling function to each generated chunk sharing this reference chunk
            for generated_chunk_idx in generated_chunks_sharing_this_ref:
                original_similarity = precision_sim_values[generated_chunk_idx]
                adjusted_similarity = _apply_scaling_function(original_similarity, semantic_coverage)
                adjusted_sim[generated_chunk_idx] = adjusted_similarity
                
                chunk_details.append({
                    "generated_chunk_idx": int(generated_chunk_idx),
                    "original_similarity": float(original_similarity),
                    "adjusted_similarity": float(adjusted_similarity)
                })
            
            load_sharing_details.append({
                "reference_chunk_idx": int(reference_chunk_idx),
                "num_gens_shared": int(num_gens_sharing_this_ref),
                "semantic_coverage": float(semantic_coverage),
                "load_sharing_penalty": float(1 - semantic_coverage),
                "chunk_details": chunk_details
            })
    
    internals = {"load_sharing_details": load_sharing_details}
    return adjusted_sim, internals