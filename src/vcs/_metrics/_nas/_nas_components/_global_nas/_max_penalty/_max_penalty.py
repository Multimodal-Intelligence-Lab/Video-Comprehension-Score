import numpy as np
from typing import List, Tuple

def calculate_max_penalty(alignment_windows: List[Tuple[int, int]], y_max: int) -> float:
    windows = np.array(alignment_windows)
    start_indices = windows[:, 0]
    end_indices = windows[:, 1]

    distance_to_start = start_indices
    distance_to_end = y_max - end_indices

    max_distances = np.maximum(distance_to_start, distance_to_end)
    sum_max_dist = np.sum(max_distances)

    return (sum_max_dist / float(y_max)) if y_max > 0 else 0.0