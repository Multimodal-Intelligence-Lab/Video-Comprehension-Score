import numpy as np
from typing import Dict, Any, Tuple

from .._scaling_function._scaling_function import _apply_scaling_function

def _apply_semantic_load_penalty_recall(
    recall_sim_values: np.ndarray,
    recall_indices: np.ndarray,
    gen_len: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply semantic load sharing penalty for recall direction.
    
    Groups reference chunks by their chosen generated chunk and calculates
    semantic coverage based on demand vs supply ratio.
    
    Returns:
        Tuple containing adjusted similarities and internal calculation details
    """
    if recall_sim_values.size == 0:
        return recall_sim_values, {"load_sharing_details": []}
    
    adjusted_sim = recall_sim_values.copy()
    load_sharing_details = []
    
    # Group reference chunks by their chosen generated chunk
    for generated_chunk_idx in range(gen_len):
        # Find all reference chunks that chose this generated chunk
        reference_chunks_sharing_this_gen = np.where(recall_indices == generated_chunk_idx)[0]
        num_refs_sharing_this_gen = len(reference_chunks_sharing_this_gen)
        
        if num_refs_sharing_this_gen >= 2:  # Only apply penalty when load sharing occurs
            # Calculate semantic coverage ratio: total_supply / total_demand
            total_supply = np.sum(recall_sim_values[reference_chunks_sharing_this_gen])
            total_demand = num_refs_sharing_this_gen  # Each reference chunk demands 1 unit
            semantic_coverage = total_supply / total_demand
            chunk_details = []
            
            # Apply scaling function to each reference chunk sharing this generated chunk
            for reference_chunk_idx in reference_chunks_sharing_this_gen:
                original_similarity = recall_sim_values[reference_chunk_idx]
                adjusted_similarity = _apply_scaling_function(original_similarity, semantic_coverage)
                adjusted_sim[reference_chunk_idx] = adjusted_similarity
                
                chunk_details.append({
                    "reference_chunk_idx": int(reference_chunk_idx),
                    "original_similarity": float(original_similarity),
                    "adjusted_similarity": float(adjusted_similarity)
                })
            
            load_sharing_details.append({
                "generated_chunk_idx": int(generated_chunk_idx),
                "num_refs_shared": int(num_refs_sharing_this_gen),
                "semantic_coverage": float(semantic_coverage),
                "load_sharing_penalty": float(1 - semantic_coverage),
                "chunk_details": chunk_details
            })
    
    internals = {"load_sharing_details": load_sharing_details}
    return adjusted_sim, internals