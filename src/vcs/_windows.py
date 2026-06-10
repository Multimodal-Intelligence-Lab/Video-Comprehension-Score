import math
import numpy as np
from typing import List, Tuple

def _get_alignment_windows(ref_len: int, gen_len: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    is_ref_longer = ref_len >= gen_len
    longer_len = ref_len if is_ref_longer else gen_len
    shorter_len = gen_len if is_ref_longer else ref_len

    slope = longer_len / shorter_len if shorter_len else 0
    alignment_window_height = math.ceil(slope)
    
    indices = np.arange(shorter_len)
    idx_points = indices * slope
    starts = np.maximum(np.floor(idx_points).astype(int), 0)
    ends = np.minimum(starts + alignment_window_height, longer_len)
    
    direct_windows = list(zip(starts, ends))

    # The set {i : starts[i] <= j < ends[i]} is a contiguous run because both
    # starts and ends are non-decreasing: it equals [lo, hi) with
    #   lo = first i whose end exceeds j, hi = first i whose start exceeds j.
    # Replaces the O(longer_len * shorter_len) membership scan; output is
    # exactly equal to v1's (fuzz-verified against the vendored original).
    long_indices = np.arange(longer_len)
    lo_bounds = np.searchsorted(ends, long_indices, side="right")
    hi_bounds = np.searchsorted(starts, long_indices, side="right")
    first_start = direct_windows[0][0] if direct_windows else 0

    reverse_windows = []
    for long_idx in range(longer_len):
        lo, hi = int(lo_bounds[long_idx]), int(hi_bounds[long_idx])
        if lo < hi:
            reverse_windows.append((lo, hi))
        else:
            if long_idx < first_start:
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