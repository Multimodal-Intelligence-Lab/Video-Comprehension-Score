import numpy as np
from typing import List, Tuple, Dict, Any

from ._ideal_line_band._ideal_line_band import _compute_ideal_narrative_line_band
from ._actual_line_length._actual_line_length import _compute_actual_line_length

def _calculate_line_based_nas(
    aligned: List[Tuple],
    mapping_windows,
    ref_len: int, 
    gen_len: int, 
    swap: bool = False,
    lct: int = 0
) -> Tuple[float, Dict[str, Any]]:
    if not aligned:
        return 0.0, {"message": "No aligned segments"}
    
    if not swap:
        source_len = ref_len
        target_len = gen_len
        sort_key = lambda point: point[0]
        sx_idx, sy_idx = 0, 1
    else:
        source_len = gen_len
        target_len = ref_len
        sort_key = lambda point: point[1]
        sx_idx, sy_idx = 1, 0
    
    sorted_aligned = sorted(aligned, key=sort_key)
    sx = np.array([point[sx_idx] for point in sorted_aligned])
    sy = np.array([point[sy_idx] for point in sorted_aligned])

    floor_ideal_line_length, ceil_ideal_line_length, floor_path, ceil_path = _compute_ideal_narrative_line_band(mapping_windows, source_len, target_len)

    floor_path_dy_map = {}
    if len(floor_path) > 1:
        for i in range(len(floor_path) - 1):
            x_pos = floor_path[i][0]
            dy = floor_path[i+1][1] - floor_path[i][1]
            floor_path_dy_map[x_pos] = dy
    
    actual_line_length, segments = _compute_actual_line_length(sx, sy, source_len, target_len, lct, floor_path_dy_map)
    average_ideal_line_length = (floor_ideal_line_length + ceil_ideal_line_length) / 2

    if floor_ideal_line_length <= actual_line_length <= ceil_ideal_line_length:
        line_nas = 1.0
    elif actual_line_length < floor_ideal_line_length:
        line_nas = actual_line_length / floor_ideal_line_length if floor_ideal_line_length else 0.0
    else:
        line_nas = ceil_ideal_line_length / actual_line_length if actual_line_length else 0.0
    
    actual_path = [(int(x), int(y)) for x, y in zip(sx, sy)]
    
    internals = {
        "actual_line_length": actual_line_length,
        "floor_ideal_line_length": floor_ideal_line_length,
        "ceil_ideal_line_length": ceil_ideal_line_length,
        "average_ideal_line_length": average_ideal_line_length,
        "segments": segments,
        "actual_path": actual_path,
        "floor_path": floor_path,
        "ceil_path": ceil_path
    }
    
    return line_nas, internals