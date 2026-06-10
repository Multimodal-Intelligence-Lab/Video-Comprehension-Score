import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from ._utils import _calculate_f1


def _calculate_max_penalty(alignment_windows: List[Tuple[int, int]], y_max: int) -> float:
    windows = np.array(alignment_windows)
    start_indices = windows[:, 0]
    end_indices = windows[:, 1]

    distance_to_start = start_indices
    distance_to_end = y_max - end_indices

    max_distances = np.maximum(distance_to_start, distance_to_end)
    sum_max_dist = np.sum(max_distances)

    return (sum_max_dist / float(y_max)) if y_max > 0 else 0.0


def _calculate_actual_penalty(
    best_indices: np.ndarray,
    alignment_windows: List[Tuple[int, int]],
    length: int,
    direction: str,
    Rn: int = 0,
    ref_len: int = None,
    gen_len: int = None,
    collect_details: bool = True,
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


def _calculate_global_nas(
    best_indices: np.ndarray,
    alignment_windows: List[Tuple[int, int]],
    length: int,
    direction: str,
    ref_len: int = None,
    gen_len: int = None,
    Rn: int = 0,
    collect_details: bool = True,
) -> Tuple[float, Dict[str, Any]]:

    penalties, internals = _calculate_actual_penalty(
        best_indices, alignment_windows, length, direction, Rn, ref_len, gen_len,
        collect_details,
    )

    max_total_penalty = _calculate_max_penalty(alignment_windows, length)
    
    total_penalty = np.sum(penalties)
    
    global_nas = 1 - (total_penalty / max_total_penalty) if max_total_penalty else 0

    if collect_details:
        internals.update({
            "max_penalty": max_total_penalty,
            "total_penalty": total_penalty,
            "value": global_nas
        })

    return global_nas, internals


def _compute_ideal_narrative_line_band(
    alignment_windows: List[Tuple[int, int]],
) -> Tuple[float, float, List[Tuple[int, int]], List[Tuple[int, int]]]:

    n_windows = len(alignment_windows)
    if n_windows <= 1:
        return 0.0, 0.0, [], []

    max_window_height = max(end - start for start, end in alignment_windows)
    
    dp_min_list = []
    dp_max_list = []
    pred_min_list = []
    pred_max_list = []
    
    for i in range(n_windows):
        start, end = alignment_windows[i]
        window_height = end - start
        
        dp_min_window = np.full(window_height, np.inf)
        dp_max_window = np.full(window_height, -np.inf)
        
        pred_min_window = np.full(window_height, -1, dtype=int)
        pred_max_window = np.full(window_height, -1, dtype=int)
        
        dp_min_list.append(dp_min_window)
        dp_max_list.append(dp_max_window)
        pred_min_list.append(pred_min_window)
        pred_max_list.append(pred_max_window)
    
    start0, end0 = alignment_windows[0]
    for y in range(end0 - start0):
        dp_min_list[0][y] = 0
        dp_max_list[0][y] = 0

    for i in range(1, n_windows):
        curr_x = i + 1
        curr_start, curr_end = alignment_windows[i]

        prev_x = i
        prev_start, prev_end = alignment_windows[i-1]
        
        for y_curr in range(curr_end - curr_start):
            curr_y = curr_start + y_curr
            
            for y_prev in range(prev_end - prev_start):
                prev_y = prev_start + y_prev
                
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if dp_min_list[i-1][y_prev] + distance < dp_min_list[i][y_curr]:
                    dp_min_list[i][y_curr] = dp_min_list[i-1][y_prev] + distance
                    pred_min_list[i][y_curr] = y_prev
                
                if dp_max_list[i-1][y_prev] + distance > dp_max_list[i][y_curr]:
                    dp_max_list[i][y_curr] = dp_max_list[i-1][y_prev] + distance
                    pred_max_list[i][y_curr] = y_prev
    
    last_window_idx = n_windows - 1
    last_start, last_end = alignment_windows[last_window_idx]
    last_window_height = last_end - last_start
    
    shortest_end_idx = 0
    shortest_line = dp_min_list[last_window_idx][0]
    for y in range(1, last_window_height):
        if dp_min_list[last_window_idx][y] < shortest_line:
            shortest_line = dp_min_list[last_window_idx][y]
            shortest_end_idx = y
    
    longest_end_idx = 0
    longest_line = dp_max_list[last_window_idx][0]
    for y in range(1, last_window_height):
        if dp_max_list[last_window_idx][y] > longest_line:
            longest_line = dp_max_list[last_window_idx][y]
            longest_end_idx = y
    
    floor_path = []
    curr_idx = shortest_end_idx
    for i in range(n_windows - 1, -1, -1):
        x = i + 1
        y = alignment_windows[i][0] + curr_idx
        floor_path.insert(0, (x, y))
        if i > 0:
            curr_idx = pred_min_list[i][curr_idx]

    ceil_path = []
    curr_idx = longest_end_idx
    for i in range(n_windows - 1, -1, -1):
        x = i + 1
        y = alignment_windows[i][0] + curr_idx
        ceil_path.insert(0, (x, y))
        if i > 0:
            curr_idx = pred_max_list[i][curr_idx]
    
    return shortest_line, longest_line, floor_path, ceil_path


def _compute_actual_line_length(
    x: Union[List[int], np.ndarray], 
    y: Union[List[float], np.ndarray], 
    y_axis: int, 
    x_axis: int,
    Rn: int = 0,
    floor_path_dy_map: Dict[int, int] = None,
    collect_details: bool = True,
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
        if not collect_details:
            continue
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


def _calculate_local_nas(
    aligned: List[Tuple],
    alignment_windows,
    ref_len: int,
    gen_len: int,
    swap: bool = False,
    Rn: int = 0,
    collect_details: bool = True,
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

    floor_ideal_line_length, ceil_ideal_line_length, floor_path, ceil_path = _compute_ideal_narrative_line_band(alignment_windows)

    floor_path_dy_map = {}
    if len(floor_path) > 1:
        for i in range(len(floor_path) - 1):
            x_pos = floor_path[i][0]
            dy = floor_path[i+1][1] - floor_path[i][1]
            floor_path_dy_map[x_pos] = dy
    
    actual_line_length, segments = _compute_actual_line_length(sx, sy, source_len, target_len, Rn, floor_path_dy_map, collect_details)
    average_ideal_line_length = (floor_ideal_line_length + ceil_ideal_line_length) / 2

    if floor_ideal_line_length <= actual_line_length <= ceil_ideal_line_length:
        local_nas = 1.0
    elif actual_line_length < floor_ideal_line_length:
        local_nas = actual_line_length / floor_ideal_line_length if floor_ideal_line_length else 0.0
    else:
        local_nas = ceil_ideal_line_length / actual_line_length if actual_line_length else 0.0

    if not collect_details:
        return local_nas, {}

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

    return local_nas, internals


def _compute_nas_metrics(
    sim_matrix: np.ndarray,
    ref_len: int,
    gen_len: int,
    precision_matches: List[Tuple],
    precision_indices: np.ndarray,
    precision_sim_values: np.ndarray,
    recall_matches: List[Tuple],
    recall_indices: np.ndarray,
    recall_sim_values: np.ndarray,
    precision_alignment_windows: List[Tuple[int, int]],
    recall_alignment_windows: List[Tuple[int, int]],
    ref_chunks: List[str],
    gen_chunks: List[str],
    Rn: int = 0,
    collect_details: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:

    precision_global_nas, precision_global_nas_internals = _calculate_global_nas(
        precision_indices, precision_alignment_windows, ref_len, "precision",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn, collect_details=collect_details
    )

    recall_global_nas, recall_global_nas_internals = _calculate_global_nas(
        recall_indices, recall_alignment_windows, gen_len, "recall",
        ref_len=ref_len, gen_len=gen_len, Rn=Rn, collect_details=collect_details
    )

    global_nas = _calculate_f1(precision_global_nas, recall_global_nas)

    precision_aligned_segments = []
    for g_idx, r_idx in precision_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            precision_aligned_segments.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))

    recall_aligned_segments = []
    for g_idx, r_idx in recall_matches:
        if g_idx >= 0 and r_idx >= 0 and g_idx < len(gen_chunks) and r_idx < len(ref_chunks):
            recall_aligned_segments.append((g_idx + 1, r_idx + 1, gen_chunks[g_idx], ref_chunks[r_idx]))
    
    precision_local_nas, precision_local_nas_internals = _calculate_local_nas(precision_aligned_segments, precision_alignment_windows, ref_len, gen_len, Rn=Rn, collect_details=collect_details)
    recall_local_nas, recall_local_nas_internals = _calculate_local_nas(recall_aligned_segments, recall_alignment_windows, ref_len, gen_len, swap=True, Rn=Rn, collect_details=collect_details)

    local_nas = _calculate_f1(precision_local_nas, recall_local_nas)

    nas = _calculate_f1(global_nas, local_nas)

    metrics = {
        "Precision Global NAS": precision_global_nas,
        "Recall Global NAS": recall_global_nas,
        "Global NAS": global_nas,
        "Precision Local NAS": precision_local_nas,
        "Recall Local NAS": recall_local_nas,
        "Local NAS": local_nas,
        "NAS": nas
    }

    internals = {
        "precision_global_nas_internals": precision_global_nas_internals,
        "recall_global_nas_internals": recall_global_nas_internals,
        "precision_local_nas_internals": precision_local_nas_internals,
        "recall_local_nas_internals": recall_local_nas_internals,
        "aligned_precision": precision_aligned_segments,
        "aligned_recall": recall_aligned_segments,
    }
    
    return metrics, internals
