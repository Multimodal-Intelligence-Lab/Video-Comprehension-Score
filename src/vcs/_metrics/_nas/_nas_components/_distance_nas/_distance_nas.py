import numpy as np
from typing import List, Tuple, Dict, Any

from ._max_penalty._max_penalty import calculate_max_penalty
from ._actual_penalty._actual_penalty import calculate_actual_penalty

def _calculate_distance_based_nas(
    best_indices: np.ndarray,
    mapping_windows: List[Tuple[int, int]],
    length: int,
    direction: str,
    ref_len: int = None,
    gen_len: int = None,
    lct: int = 0
) -> Tuple[float, Dict[str, Any]]:

    penalties, internals = calculate_actual_penalty(
        best_indices, mapping_windows, length, direction, lct, ref_len, gen_len
    )
    
    max_total_penalty = calculate_max_penalty(mapping_windows, length)
    
    total_penalty = np.sum(penalties)
    
    nas = 1 - (total_penalty / max_total_penalty) if max_total_penalty else 0
    
    internals.update({
        "max_penalty": max_total_penalty,
        "total_penalty": total_penalty,
        "value": nas
    })
    
    return nas, internals