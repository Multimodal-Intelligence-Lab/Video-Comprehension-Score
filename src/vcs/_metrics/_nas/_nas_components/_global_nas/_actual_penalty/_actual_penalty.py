import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_actual_penalty(
    best_indices: np.ndarray,
    alignment_windows: List[Tuple[int, int]],
    length: int,
    direction: str,
    Rn: int = 0,
    ref_len: int = None,
    gen_len: int = None
) -> Tuple[np.ndarray, Dict[str, Any]]:

    valid_indices = best_indices >= 0

    in_window = np.zeros_like(best_indices, dtype=bool)
    for i, idx in enumerate(best_indices):
        if idx >= 0:
            start, end = alignment_windows[i]
            in_window[i] = start <= idx < end

    penalties = np.zeros(len(best_indices), dtype=float)
    for i, (idx, is_valid, is_in_window) in enumerate(zip(best_indices, valid_indices, in_window)):
        if not is_valid:
            continue
        if is_in_window:
            continue

        start, end = alignment_windows[i]
        dist = start - idx if idx < start else idx - (end - 1)
        dist = 0 if dist <= Rn else dist
        penalties[i] = dist / float(length) if length else 0

    in_rn_zone = np.zeros_like(best_indices, dtype=bool)
    for i, idx in enumerate(best_indices):
        if idx >= 0:
            start, end = alignment_windows[i]
            original_in_window = start <= idx < end
            if not original_in_window and Rn > 0:
                rn_in_zone = (start - Rn <= idx < end + Rn)
                in_rn_zone[i] = rn_in_zone

    internals = {
        "penalties": penalties.tolist() if isinstance(penalties, np.ndarray) else penalties,
        "in_window": in_window.tolist() if isinstance(in_window, np.ndarray) else in_window,
        "in_rn_zone": in_rn_zone.tolist() if isinstance(in_rn_zone, np.ndarray) else in_rn_zone,
    }
    
    return penalties, internals