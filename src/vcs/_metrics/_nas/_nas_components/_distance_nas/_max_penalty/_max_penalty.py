import numpy as np
from typing import List, Tuple

def calculate_max_penalty(mapping_windows: List[Tuple[int, int]], y_max: int) -> float:
    windows = np.array(mapping_windows)
    start_indices = windows[:, 0]
    end_indices = windows[:, 1]
    
    dist_down = start_indices
    dist_up = y_max - end_indices
    
    max_distances = np.maximum(dist_down, dist_up)
    sum_max_dist = np.sum(max_distances)
    
    return (sum_max_dist / float(y_max)) if y_max > 0 else 0.0