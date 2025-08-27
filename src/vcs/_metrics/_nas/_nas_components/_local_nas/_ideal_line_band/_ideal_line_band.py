import numpy as np
import math
from typing import List, Tuple, Dict, Any

def _compute_ideal_narrative_line_band(
    mapping_windows: List[Tuple[int, int]], 
    source_len: int, 
    target_len: int
) -> Tuple[float, float, List[Tuple[int, int]], List[Tuple[int, int]]]:

    n_windows = len(mapping_windows)
    if n_windows <= 1:
        return 0.0, 0.0, [], []
    
    max_window_height = max(end - start for start, end in mapping_windows)
    
    dp_min_list = []
    dp_max_list = []
    pred_min_list = []
    pred_max_list = []
    
    for i in range(n_windows):
        start, end = mapping_windows[i]
        window_height = end - start
        
        dp_min_window = np.full(window_height, np.inf)
        dp_max_window = np.full(window_height, -np.inf)
        
        pred_min_window = np.full(window_height, -1, dtype=int)
        pred_max_window = np.full(window_height, -1, dtype=int)
        
        dp_min_list.append(dp_min_window)
        dp_max_list.append(dp_max_window)
        pred_min_list.append(pred_min_window)
        pred_max_list.append(pred_max_window)
    
    start0, end0 = mapping_windows[0]
    for y in range(end0 - start0):
        dp_min_list[0][y] = 0
        dp_max_list[0][y] = 0
    
    for i in range(1, n_windows):
        curr_x = i + 1
        curr_start, curr_end = mapping_windows[i]
        
        prev_x = i
        prev_start, prev_end = mapping_windows[i-1]
        
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
    last_start, last_end = mapping_windows[last_window_idx]
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
        y = mapping_windows[i][0] + curr_idx
        floor_path.insert(0, (x, y))
        if i > 0:
            curr_idx = pred_min_list[i][curr_idx]
    
    ceil_path = []
    curr_idx = longest_end_idx
    for i in range(n_windows - 1, -1, -1):
        x = i + 1
        y = mapping_windows[i][0] + curr_idx
        ceil_path.insert(0, (x, y))
        if i > 0:
            curr_idx = pred_max_list[i][curr_idx]
    
    return shortest_line, longest_line, floor_path, ceil_path