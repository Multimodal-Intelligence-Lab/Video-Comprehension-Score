import math
import numpy as np
from typing import List, Tuple

def _get_mapping_windows(ref_len: int, gen_len: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    is_ref_longer = ref_len >= gen_len
    longer_len = ref_len if is_ref_longer else gen_len
    shorter_len = gen_len if is_ref_longer else ref_len
    
    slope = longer_len / shorter_len if shorter_len else 0
    mapping_window_height = math.ceil(slope)
    
    indices = np.arange(shorter_len)
    idx_points = indices * slope
    starts = np.maximum(np.floor(idx_points).astype(int), 0)
    ends = np.minimum(starts + mapping_window_height, longer_len)
    
    direct_windows = list(zip(starts, ends))
    
    reverse_windows = []
    for long_idx in range(longer_len):
        short_indices = []
        for short_idx, (start, end) in enumerate(direct_windows):
            if start <= long_idx < end:
                short_indices.append(short_idx)
        
        if short_indices:
            reverse_windows.append((min(short_indices), max(short_indices) + 1))
        else:
            if long_idx < direct_windows[0][0]:
                reverse_windows.append((0, 1))
            else:
                reverse_windows.append((shorter_len - 1, shorter_len))
    
    if is_ref_longer:
        precision_windows = direct_windows
        recall_windows = reverse_windows
    else:
        recall_windows = direct_windows
        precision_windows = reverse_windows
    
    return precision_windows, recall_windows