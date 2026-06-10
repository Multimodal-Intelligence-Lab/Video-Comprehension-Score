import numpy as np
import math
from typing import List, Tuple, Union, Dict, Any

def _compute_actual_line_length(
    x: Union[List[int], np.ndarray], 
    y: Union[List[float], np.ndarray], 
    y_axis: int, 
    x_axis: int,
    Rn: int = 0,
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
        Rn_window = mapping_window_height
    else:
        if 0 < ratio_decimal_part <= 0.5:
            Rn_window = (2 * mapping_window_height) - 2
        else:
            Rn_window = (2 * mapping_window_height) - 1

    for i in range(len(dx)):

        dy_value = abs(dy[i]) if Rn > 0 else dy[i]
        
        is_calculable = False
        segment_length = 0.0
        calculation_method = "none"
        
        # CASE 1: Normal segments (reasonable vertical change)
        if dy_value <= Rn_window and dy_value >= 0:
            is_calculable = True
            segment_length = math.sqrt(dx[i]**2 + dy[i]**2)
            lengths[i] = segment_length
            calculation_method = "standard"
            
        # CASE 2: Large vertical jumps but within LCT range
        elif Rn > 0 and dy_value > Rn_window and dy_value <= Rn_window+Rn:
            is_calculable = True
            floor_dy = None
            if floor_path_dy_map and x_arr[i] in floor_path_dy_map:
                floor_dy = abs(floor_path_dy_map[x_arr[i]])
            if floor_dy is None:
                raise RuntimeError(
                    f"Rn-capped segment at x={int(x_arr[i])} has no floor-path dy "
                    f"to substitute (floor_path_dy_map keys: "
                    f"{sorted(floor_path_dy_map) if floor_path_dy_map else []}). "
                    "This indicates an internal inconsistency between the ideal "
                    "line band and the actual path."
                )
            segment_length = math.sqrt(dx[i]**2 + floor_dy**2)
            lengths[i] = segment_length
            calculation_method = "Rn_capped"
            
        # CASE 3: Beyond LCT range or negative slopes
        # Length remains 0, segment not calculable
        
        # Store segment details for internals
        segments.append({
            "start": (int(x_arr[i]), int(y_arr[i])),
            "end": (int(x_arr[i+1]), int(y_arr[i+1])),
            "dx": int(dx[i]),
            "dy": int(dy[i]),
            "threshold": float(Rn_window),
            "threshold_with_Rn": float(Rn_window+Rn),
            "is_calculable": is_calculable,
            "calculation_method": calculation_method,
            "length": float(segment_length)
        })
    
    actual_line_length = np.sum(lengths)
    return actual_line_length, segments