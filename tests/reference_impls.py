"""Verbatim v1 reference implementations, vendored for equivalence fuzzing.

When the live implementations are rewritten for speed (plan C12/C13), the
fuzz suites compare them against these frozen copies at EXACT equality.
Do not edit these functions — they are the spec.

Provenance (commit 3c6a2718):
  _get_alignment_windows   <- src/vcs/_alignment_windows/_alignment_windows.py
  calculate_rn_window,
  calculate_actual_penalty <- src/vcs/_metrics/_nas/_nas_components/_global_nas/
                              _actual_penalty/_actual_penalty.py
"""
import math
from typing import Any, Dict, List, Tuple

import numpy as np


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


def calculate_rn_window(y_axis: int, x_axis: int) -> int:
    alignment_window_height = math.ceil(y_axis / x_axis) if x_axis else 0

    ratio = y_axis / x_axis if x_axis else 0
    ratio_decimal_part = ratio - math.floor(ratio)

    if y_axis <= x_axis:
        rn_window = alignment_window_height
    else:
        if 0 < ratio_decimal_part <= 0.5:
            rn_window = alignment_window_height - 1
        else:
            rn_window = alignment_window_height

    return rn_window


def calculate_actual_penalty(
    best_indices: np.ndarray,
    alignment_windows: List[Tuple[int, int]],
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

    rn_window = calculate_rn_window(y_axis, x_axis)

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
        "alignment_window_height": rn_window,
        "penalties": penalties.tolist() if isinstance(penalties, np.ndarray) else penalties,
        "in_window": in_window.tolist() if isinstance(in_window, np.ndarray) else in_window,
        "in_rn_zone": in_rn_zone.tolist() if isinstance(in_rn_zone, np.ndarray) else in_rn_zone,
    }

    return penalties, internals
