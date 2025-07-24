import numpy as np
import math
from typing import List, Tuple, Union, Dict, Any

def _compute_actual_line_length(
    x: Union[List[int], np.ndarray], 
    y: Union[List[float], np.ndarray], 
    y_axis: int, 
    x_axis: int,
    lct: int = 0,
    floor_path_dy_map: Dict[int, int] = None

) -> Tuple[float, List[Dict[str, Any]]]: 
    x_arr = np.array(x) if not isinstance(x, np.ndarray) else x
    y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
    
    if len(x_arr) <= 1:
        return 0.0, [] 
    
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    
    mapping_window_height = math.ceil(y_axis / x_axis) if x_axis else 0
    lengths = np.zeros_like(dx, dtype=float)
    
    ratio = y_axis / x_axis if x_axis else 0
    ratio_decimal_part = ratio - math.floor(ratio)
    
    segments = []
    
    if y_axis <= x_axis:
        lct_window = mapping_window_height
        expanded_lct_window = lct_window + (mapping_window_height * lct) if lct > 0 else lct_window

    else:
        if 0 < ratio_decimal_part <= 0.5:
            lct_window = (2 * mapping_window_height) - 2
            expanded_lct_window = lct_window + ((mapping_window_height - 1) * lct) if lct > 0 else lct_window
        else:
            lct_window = (2 * mapping_window_height) - 1
            expanded_lct_window = lct_window + (mapping_window_height * lct) if lct > 0 else lct_window

    for i in range(len(dx)):

        dy_value = abs(dy[i]) if lct > 0 else dy[i]
        
        is_calculable = False
        segment_length = 0.0
        calculation_method = "none"
        
        # CASE 1: Normal segments (reasonable vertical change)
        if dy_value <= lct_window and dy_value >= 0:
            is_calculable = True
            segment_length = math.sqrt(dx[i]**2 + dy[i]**2)
            lengths[i] = segment_length
            calculation_method = "standard"
            
        # CASE 2: Large vertical jumps but within LCT range
        elif lct > 0 and dy_value > lct_window and dy_value <= expanded_lct_window:
            is_calculable = True
            floor_dy = None
            if floor_path_dy_map and x_arr[i] in floor_path_dy_map:
                floor_dy = abs(floor_path_dy_map[x_arr[i]])
            segment_length = math.sqrt(dx[i]**2 + floor_dy**2)
            lengths[i] = segment_length
            calculation_method = "lct_capped"
            
        # CASE 3: Beyond LCT range or negative slopes
        # Length remains 0, segment not calculable
        
        # Store segment details for visualization
        segments.append({
            "start": (int(x_arr[i]), int(y_arr[i])),
            "end": (int(x_arr[i+1]), int(y_arr[i+1])),
            "dx": int(dx[i]),
            "dy": int(dy[i]),
            "threshold": float(lct_window),
            "threshold_with_lct": float(expanded_lct_window),
            "is_calculable": is_calculable,
            "calculation_method": calculation_method,
            "length": float(segment_length)
        })
    
    total_length = np.sum(lengths)
    return total_length, segments