import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def _find_best_match_with_context(
    similarity_array: np.ndarray, 
    mapping_windows: Optional[Tuple[int, int]] = None,
    context_cutoff_value: float = 0.6,
    context_window_ctrl: float = 5.0
) -> Tuple[int, Dict[str, Any]]:  # Return type changed to include details

    if similarity_array.size == 0:
        return -1, {}
    
    max_val = np.max(similarity_array)
    max_idx = np.argmax(similarity_array)
    context_range = 1 - context_cutoff_value
    
    # Determine if context window is applied
    context_window_applied = max_val > 0 and (context_range - (1 - max_val)) > 0
    
    context_window = ((context_range - (1 - max_val)) / max_val) / context_window_ctrl if context_window_applied else 0.0
    context_threshold = max_val - context_window
    
    # Get all candidates within threshold
    candidate_indices = np.where(similarity_array >= context_threshold)[0]
    candidate_values = similarity_array[candidate_indices]
    
    # Initialize selection data
    selection_details = {
        "max_value": float(max_val),
        "max_index": int(max_idx),
        "context_range": float(context_range),
        "context_window": float(context_window),
        "context_threshold": float(context_threshold),
        "context_window_applied": bool(context_window_applied),
        "candidates": []
    }
    
    # If there's only one candidate or no mapping window
    if len(candidate_indices) == 1 or mapping_windows is None:
        selection_details["selected_index"] = int(max_idx)
        selection_details["selection_reason"] = "max_similarity" if len(candidate_indices) == 1 else "no_mapping_window"
        return max_idx, selection_details
    
    # Process all candidates
    start, end = mapping_windows
    selection_details["mapping_window"] = {"start": int(start), "end": int(end)}
    
    in_window = (candidate_indices >= start) & (candidate_indices < end)
    left_dist = np.maximum(start - candidate_indices, 0)
    right_dist = np.maximum(candidate_indices - (end - 1), 0)
    distances = np.where(in_window, 0, np.maximum(left_dist, right_dist))
    
    # Find the candidate with minimum distance, using similarity as tie-breaker
    min_distance = np.min(distances)
    min_dist_mask = distances == min_distance
    
    if np.sum(min_dist_mask) == 1:
        # Only one candidate with minimum distance
        min_dist_idx = np.argmax(min_dist_mask)
        selected_idx = candidate_indices[min_dist_idx]
    else:
        # Multiple candidates with same minimum distance - use similarity as tie-breaker
        tied_candidates = candidate_indices[min_dist_mask]
        tied_similarities = candidate_values[min_dist_mask]
        best_similarity_idx = np.argmax(tied_similarities)
        selected_idx = tied_candidates[best_similarity_idx]
        min_dist_idx = np.where(candidate_indices == selected_idx)[0][0]
    
    # Record all candidate details
    for i, (cand_idx, cand_val, is_in_win, dist) in enumerate(zip(
            candidate_indices, candidate_values, in_window, distances)):
        candidate_info = {
            "index": int(cand_idx),
            "similarity": float(cand_val),
            "in_window": bool(is_in_win),
            "distance": int(dist),
            "is_selected": cand_idx == selected_idx
        }
        selection_details["candidates"].append(candidate_info)
    
    # Record selection reason
    if in_window[min_dist_idx]:
        selection_reason = "in_mapping_window"
    else:
        selection_reason = "closest_to_mapping_window"
    selection_details["selection_reason"] = selection_reason
    selection_details["selected_index"] = int(selected_idx)
    
    return selected_idx, selection_details

def _calculate_row_col_matches_context(
    sim_matrix: np.ndarray,
    mapping_windows: List[Tuple[int, int]],
    direction: str,
    context_cutoff_value: float,
    context_window_ctrl: float
) -> Tuple[List[Tuple], np.ndarray, np.ndarray, Dict[str, Any]]:  # Added Dict to return type

    ref_len, gen_len = sim_matrix.shape
    match_details = {}  # Store all detailed information
    
    if direction == "precision":
        length = gen_len
        best_indices = np.full(length, -1, dtype=int)
        sim_values = np.zeros(length, dtype=float)
        match_details["direction"] = "precision"
        match_details["segments"] = []
        
        for g_idx in range(gen_len):
            column = sim_matrix[:, g_idx]
            if column.size == 0:
                match_details["segments"].append({
                    "index": g_idx,
                    "valid": False,
                    "reason": "empty_column"
                })
                continue
            
            start_ref, end_ref = mapping_windows[g_idx]
            r_idx, selection_details = _find_best_match_with_context(
                column, (start_ref, end_ref),
                context_cutoff_value, context_window_ctrl
            )
            
            segment_details = {
                "index": g_idx,
                "mapping_window": {"start": start_ref, "end": end_ref},
                "selection": selection_details,
                "valid": r_idx >= 0
            }
            match_details["segments"].append(segment_details)
            
            if r_idx >= 0:
                best_indices[g_idx] = r_idx
                sim_values[g_idx] = sim_matrix[r_idx, g_idx]
        
        matches = [(g_idx, best_indices[g_idx]) 
                  for g_idx in range(gen_len) 
                  if best_indices[g_idx] >= 0]
            
    elif direction == "recall":
        length = ref_len
        best_indices = np.full(length, -1, dtype=int)
        sim_values = np.zeros(length, dtype=float)
        match_details["direction"] = "recall"
        match_details["segments"] = []
        
        for r_idx in range(ref_len):
            row = sim_matrix[r_idx, :]
            if row.size == 0:
                match_details["segments"].append({
                    "index": r_idx,
                    "valid": False,
                    "reason": "empty_row"
                })
                continue
            
            start_gen, end_gen = mapping_windows[r_idx]
            g_idx, selection_details = _find_best_match_with_context(
                row, (start_gen, end_gen),
                context_cutoff_value, context_window_ctrl
            )
            
            segment_details = {
                "index": r_idx,
                "mapping_window": {"start": start_gen, "end": end_gen},
                "selection": selection_details,
                "valid": g_idx >= 0
            }
            match_details["segments"].append(segment_details)
            
            if g_idx >= 0:
                best_indices[r_idx] = g_idx
                sim_values[r_idx] = sim_matrix[r_idx, g_idx]
        
        matches = [(best_indices[r_idx], r_idx) 
                  for r_idx in range(ref_len) 
                  if best_indices[r_idx] >= 0]
    
    return matches, best_indices, sim_values, match_details