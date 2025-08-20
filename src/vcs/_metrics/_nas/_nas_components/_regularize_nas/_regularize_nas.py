from typing import List, Tuple, Dict, Any

def _calculate_window_regularizer(
    ref_len: int,
    gen_len: int,
    prec_map_windows: List[Tuple[int, int]],
    rec_map_windows: List[Tuple[int, int]]
) -> Tuple[float, Dict[str, Any]]:
    if ref_len < gen_len:
        mapping_windows = rec_map_windows
        max_len = gen_len
    else:
        mapping_windows = prec_map_windows
        max_len = ref_len
    
    total_mapping_window_area = 0
    for start, end in mapping_windows:
        mapping_window_height = end - start
        window_area = mapping_window_height * 1
        total_mapping_window_area += window_area
    
    timeline_area = ref_len * gen_len
    
    if ref_len > gen_len:
        min_area = 1 / ref_len
    else:
        min_area = 1 / gen_len
    
    if timeline_area > 0 and min_area < 1: 
        window_regularizer = (total_mapping_window_area / timeline_area - min_area) / (0.5 - min_area)
        window_regularizer = max(0, min(1, window_regularizer))
    else:
        window_regularizer = 0
    
    internals = {
        "total_mapping_window_area": total_mapping_window_area,
        "timeline_area": timeline_area,
        "min_area": min_area,
    }
    
    return window_regularizer, internals

def _regularize_nas(nas_f1: float, window_regularizer: float) -> float:

    nas_regularized = nas_f1 - window_regularizer
    
    return (nas_regularized / (1 - window_regularizer)) if (nas_regularized > 0) else 0.0