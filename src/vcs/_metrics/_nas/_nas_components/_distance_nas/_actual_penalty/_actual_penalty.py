import numpy as np
import math
from typing import List, Tuple, Dict, Any

def calculate_Rn_window(y_axis: int, x_axis: int) -> int:
    mapping_window_height = math.ceil(y_axis / x_axis) if x_axis else 0
    
    ratio = y_axis / x_axis if x_axis else 0
    ratio_decimal_part = ratio - math.floor(ratio)
    
    if y_axis <= x_axis:
        Rn_window = mapping_window_height
    else:
        if 0 < ratio_decimal_part <= 0.5:
            Rn_window = mapping_window_height - 1
        else:
            Rn_window = mapping_window_height
            
    return Rn_window

def calculate_actual_penalty(
    best_indices: np.ndarray,
    mapping_windows: List[Tuple[int, int]],
    length: int,
    direction: str,
    Rn: int = 0,
    ref_len: int = None,
    gen_len: int = None
) -> Tuple[np.ndarray, Dict[str, Any]]:

    if direction == "precision":
        y_axis = ref_len if ref_len is not None else length
        x_axis = gen_len if gen_len is not None else len(best_indices)
    else:
        y_axis = gen_len if gen_len is not None else length
        x_axis = ref_len if ref_len is not None else len(best_indices)
    
    Rn_window = calculate_Rn_window(y_axis, x_axis)
    
    valid_indices = best_indices >= 0
    valid_mask = valid_indices
    
    in_window = np.zeros_like(best_indices, dtype=bool)
    for i, idx in enumerate(best_indices):
        if idx >= 0:
            start, end = mapping_windows[i]
            in_window[i] = start <= idx < end
    
    penalties = np.zeros(len(best_indices), dtype=float)
    for i, (idx, is_valid, is_in_window) in enumerate(zip(best_indices, valid_mask, in_window)):
        if not is_valid:
            continue
        if is_in_window:
            continue
            
        start, end = mapping_windows[i]
        dist = start - idx if idx < start else idx - (end - 1)
        dist = 0 if dist <= Rn else dist
        penalties[i] = dist / float(length) if length else 0
    
    in_Rn_zone = np.zeros_like(best_indices, dtype=bool)
    for i, idx in enumerate(best_indices):
        if idx >= 0:
            start, end = mapping_windows[i]
            original_in_window = start <= idx < end
            if not original_in_window and Rn > 0:
                Rn_in_zone = (start - Rn <= idx < end + Rn)
                in_Rn_zone[i] = Rn_in_zone
    
    inteRnals = {
        "mapping_window_height": Rn_window,
        "penalties": penalties.tolist() if isinstance(penalties, np.ndarray) else penalties,
        "in_window": in_window.tolist() if isinstance(in_window, np.ndarray) else in_window,
        "in_Rn_zone": in_Rn_zone.tolist() if isinstance(in_Rn_zone, np.ndarray) else in_Rn_zone,
    }
    
    return penalties, inteRnals